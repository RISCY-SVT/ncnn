# YOLO + NCNN Notes (K1X / BPI-F3)

_Last updated: 2026-01-03_

This document records:

- how we convert YOLO models to ncnn format in a reproducible way
- how we benchmark YOLO11n deterministically on Banana-Pi BPI-F3 (K1X)
- the INT8 workflow (quantization + validation) and the current status of INT8 runtime on K1X

---

## 0. Canonical locations

- ncnn repo: `/data/ncnn`
- K1X tooling:
  - `/data/ncnn/tools/k1x/convert_yolo_to_ncnn.py`
  - `/data/ncnn/tools/k1x/make_coco2017_calib_for_ncnn.py`
- Calibration dataset (example):
  - `/data/datasets/coco_calib2K/images`
  - `/data/datasets/coco_calib2K/imagelist.txt`
  - `/data/datasets/coco_calib2K/selection.json`
- Logs:
  - `/data/ncnn-logs/YYYY-MM-DD/`
- Board smoke workspace (example):
  - `/home/svt/ncnn-k1x-int8-smoke`
  - binaries: `/home/svt/ncnn-k1x-int8-smoke/bin/`
  - models: `/home/svt/ncnn-k1x-int8-smoke/models/`

---

## 1. Deterministic board benchmark (FP16)

### 1.1. One-line template (board)

```bash
cd /home/svt/ncnn-k1x-int8-smoke

unset OMP_NUM_THREADS OMP_PROC_BIND OMP_PLACES OMP_SCHEDULE OMP_MAX_ACTIVE_LEVELS OMP_NESTED OMP_STACKSIZE OMP_CANCELLATION OMP_DISPLAY_ENV
unset GOMP_CPU_AFFINITY GOMP_STACKSIZE GOMP_SPINCOUNT

export LD_LIBRARY_PATH=/home/svt/opencv-install-k1x-gtk3/lib:${LD_LIBRARY_PATH:-}

./bin/yolo11_fp16 \
  --param ./models/yolo11n.ncnn.param \
  --bin   ./models/yolo11n.ncnn.bin \
  --image ./models/photo_2024-10-11_10-04-04.jpg \
  --no-gui \
  --bench-only --pin cluster0 --threads 4 \
  --warmup 10 --runs 100 --repeats 5 \
  --strict-omp-env 1 --quiet \
  2>&1 | tee fp16_bench.log
```

### 1.2. Reference baseline (recent)

A stable FP16 baseline on cluster0 pinned, threads=4, strict env:

- mean: ~566 ms (≈ 1.77 FPS)
- stddev across repeats: ~0.9–1.2 ms (n=5)

(Exact numbers depend on board load and OS image; always trust the log produced by the harness.)

---

## 2. Per-layer benchmark (hotspots)

Per-layer requires the ncnn benchmark build (`-DNCNN_BENCHMARK=ON`) and a benchmark-enabled binary
on the board (example naming: `yolo11_bench`).

Typical output includes a per-layer time table and a final total.

**Note:** per-layer mode is significantly slower than the forward-only loop. Use it primarily to
rank hotspots, not to compare absolute FPS.

### 2.1. Hotspots (example Top-10)

From a recent on-board per-layer run (single run, single repeat):

- 80.42 ms  Convolution  conv_4
- 39.06 ms  Slice        split_0
- 32.07 ms  Convolution  conv_3
- 31.63 ms  Concat       cat_0
- 31.51 ms  Convolution  conv_83
- 28.82 ms  Split        splitncnn_0
- 27.32 ms  Concat       cat_20
- 26.83 ms  Convolution  conv_9
- 24.19 ms  Slice        split_10
- 22.20 ms  Convolution  conv_6

Per-layer total for that run: ~776 ms (≈ 1.29 FPS).

---

## 3. INT8 calibration dataset (COCO2017)

We use a deterministic helper to build a calibration set from COCO2017.

Example (host/container):

```bash
cd /data/ncnn/tools/k1x

python3 make_coco2017_calib_for_ncnn.py \
  --out-dir /data/datasets/coco_calib2K \
  --n 2000 \
  --seed 123 \
  --split val2017 \
  --jobs 8
```

### 3.1. Determinism note: imagelist ordering

For strict determinism, keep `imagelist.txt` sorted:

```bash
sort -o /data/datasets/coco_calib2K/imagelist.txt /data/datasets/coco_calib2K/imagelist.txt
```

If `imagelist.txt` is not sorted but `selection.json` contains a fixed seed, the sample selection can
still be deterministic, but we treat it as WARN in tooling.

---

## 4. INT8 quantization workflow (host/container)

We convert and quantize using the K1X helper:

- `tools/k1x/convert_yolo_to_ncnn.py`

This script orchestrates:

- model conversion to ncnn (param/bin)
- table generation (`ncnn2table`) with a deterministic imagelist
- INT8 quantization (`ncnn2int8`)
- fail-fast validation of table values (NaN/Inf/<=0)

The script supports both KL and ACIQ workflows (depending on your configuration).

**Important:** current K1X runtime performance for INT8 is poor (see section 6); treat INT8 as a
correctness/coverage workflow until kernels are implemented.

---

## 5. Board run template (INT8)

Assuming INT8 artifacts exist in `./models/` and a corresponding binary exists as `yolo11_int8`:

```bash
cd /home/svt/ncnn-k1x-int8-smoke

unset OMP_NUM_THREADS OMP_PROC_BIND OMP_PLACES OMP_SCHEDULE OMP_MAX_ACTIVE_LEVELS OMP_NESTED OMP_STACKSIZE OMP_CANCELLATION OMP_DISPLAY_ENV
unset GOMP_CPU_AFFINITY GOMP_STACKSIZE GOMP_SPINCOUNT

export LD_LIBRARY_PATH=/home/svt/opencv-install-k1x-gtk3/lib:${LD_LIBRARY_PATH:-}

./bin/yolo11_int8 \
  --param ./models/yolo11n-int8.ncnn.param \
  --bin   ./models/yolo11n-int8.ncnn.bin \
  --image ./models/photo_2024-10-11_10-04-04.jpg \
  --no-gui \
  --bench-only --pin cluster0 --threads 4 \
  --warmup 10 --runs 100 --repeats 5 \
  --strict-omp-env 1 --quiet \
  2>&1 | tee int8_bench.log
```

---

## 6. INT8 runtime status on K1X (why it is slow)

As of the current triage, the RISC-V backend does not provide optimized INT8 convolution kernels for
K1X, so INT8 execution can fall back to generic/scalar paths with packing disabled, producing
catastrophic slowdowns (seconds per frame).

This is expected until we implement RVV INT8 kernels in the RISC-V backend.

---

## 7. Codex skills (recommended path)

Instead of manually typing long sequences, use the project skills:

- `$k1x_env_sanity`
- `$k1x_cross_build_ncnn`
- `$k1x_deploy_and_run_yolo11`
- `$k1x_layer_hotspots_report`
- `$k1x_int8_quantize_yolo11`
- `$k1x_git_hygiene`

All skills log under `/data/ncnn-logs/YYYY-MM-DD/` with Task Summary blocks.

See `k1x-env-overview.md` for the authoritative list and acceptance-test run.

