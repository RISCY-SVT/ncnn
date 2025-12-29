## Analysis of yolo11n failure
- Reproduced with `python tools/k1x/convert_yolo_to_ncnn.py yolo11n -k`; re-export in `yolo11n_pnnx.py` failed during `torch.jit.trace` with `RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 6400 but got size 1600 for tensor number 1 in the list.` from `torch.cat` in the modified script.
- `modify_pnnx_script()` changed `torch.cat((v_235, v_236, v_237), dim=2)` to `dim=1`, so tensors shaped `(1,144,6400)`, `(1,144,1600)`, `(1,144,400)` no longer matched on dim=2, causing the mismatch. It also rewrote fixed `view/reshape(...,20,20)` ops to use `v_95.size(2/3)`.
- The failure happens in the “re-export modified PNNX script” step when tracing the patched script, before any NCNN files are produced by the second PNNX pass.

## Fix for YOLO11 detection
- Converter now first tries Ultralytics’ native NCNN export: `yolo export model=<base>.pt format=ncnn` inside `ncnn_conversion_temp`, then copies the produced `*.ncnn.param/bin` to `models/` with the expected filenames.
- If direct export fails, it falls back to a single PNNX conversion of the original TorchScript with `inputshape=[1,3,640,640]`, skipping `modify_pnnx_script()` and `re_export_torchscript()` for YOLO11 detect models to avoid the fragile dim edits.
- The YOLO11 detect path is treated as `requires_modification = False`, so only the static PNNX run is used in the fallback path. Ultralytics 8.3.232 (current environment) succeeds with the direct NCNN export.

## Mapping to C++ examples
- yolo11.cpp → `yolo11n.ncnn.param/bin` (model_name: `yolo11n`)
- yolo11_cls.cpp → `yolo11n_cls.ncnn.param/bin` (model_name: `yolo11n-cls`)
- yolo11_obb.cpp → `yolo11n_obb.ncnn.param/bin` (model_name: `yolo11n-obb`)
- yolo11_pose.cpp → `yolo11n_pose.ncnn.param/bin` (model_name: `yolo11n-pose`)
- yolo11_seg.cpp → `yolo11n_seg.ncnn.param/bin` (model_name: `yolo11n-seg`)
- yolov8.cpp → `yolov8n.ncnn.param/bin` (model_name: `yolov8n`)
- yolov8_cls.cpp → `yolov8n_cls.ncnn.param/bin` (model_name: `yolov8n-cls`)
- yolov8_obb.cpp → `yolov8n_obb.ncnn.param/bin` (model_name: `yolov8n-obb`)
- yolov8_pose.cpp → `yolov8n_pose.ncnn.param/bin` (model_name: `yolov8n-pose`)
- yolov8_seg.cpp → `yolov8n_seg.ncnn.param/bin` (model_name: `yolov8n-seg`)
- yoloworld.cpp → `yolov8s_worldv2.ncnn.param/bin` (model_name: `yolov8s-worldv2`)
- yolov5_pnnx.cpp → `yolov5s.ncnn.param/bin` (model_name: `yolov5s`)
- These mappings are encoded in `EXAMPLE_MODEL_MAP` and drive the optional `--for-example` flag so converter outputs match the filenames each C++ sample loads.

## Smoke tests
- `python tools/k1x/convert_yolo_to_ncnn.py yolo11n-cls` → produced `models/yolo11n_cls.ncnn.param/bin`, verified with `ncnn.Net`.
- `python tools/k1x/convert_yolo_to_ncnn.py yolo11n` → used Ultralytics direct NCNN export; produced `models/yolo11n.ncnn.param/bin`, verified with `ncnn.Net`.
- On Banana Pi BPI-F3, running the example with a local test image (e.g., `/path/to/image.jpg`) logs `out0 shape: w=8400 h=84 c=1` and renders detections. Use the deterministic harness for timing.

## INT8 workflow (K1X)
### Host quantization (x86_64)
```
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DNCNN_BUILD_TOOLS=ON -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_VULKAN=OFF -S /data/ncnn -B /data/ncnn/build-host-tools
cmake --build /data/ncnn/build-host-tools --target ncnn2table ncnn2int8

python3 tools/k1x/convert_yolo_to_ncnn.py \
  --for-example yolo11 \
  --output-dir ./models-int8 \
  --work-dir ./tmp_int8_work \
  --imgsz 640 \
  --int8 1 \
  --calib-dir /data/opencv \
  --calib-count 200 \
  --int8-method kl \
  --calib-threads 8 \
  --force
```

### INT8 workflow robustness
- Table validation is fail-fast by default. Non-finite or non-positive scales abort the conversion and print the offending line.
- Sanitization is opt-in only. Use `--allow-sanitize-nonfinite 1` to write a `*.sanitized.table` copy while preserving the original table.
- Mixed precision is supported by commenting out table lines with `#` and the validator ignores commented and blank lines.

### Reproducible INT8 conversion (KL / ACIQ)
```
python3 tools/k1x/convert_yolo_to_ncnn.py \
  --for-example yolo11 \
  --output-dir ./models-int8 \
  --work-dir ./tmp_int8_work_kl \
  --imgsz 640 \
  --int8 1 \
  --imagelist /data/datasets/coco_calib2K/imagelist.txt \
  --calib-count 2000 \
  --int8-method kl \
  --calib-threads 8 \
  --failfast-nonfinite 1

python3 tools/k1x/convert_yolo_to_ncnn.py \
  --for-example yolo11 \
  --output-dir ./models-int8 \
  --work-dir ./tmp_int8_work_aciq \
  --imgsz 640 \
  --int8 1 \
  --imagelist /data/datasets/coco_calib2K/imagelist.txt \
  --calib-count 2000 \
  --int8-method aciq \
  --calib-threads 8 \
  --failfast-nonfinite 1
```

### Board runs (Banana Pi BPI-F3)
```
export LD_LIBRARY_PATH=/home/svt/opencv-install-k1x-gtk3/lib:${LD_LIBRARY_PATH:-}
unset OMP_NUM_THREADS OMP_PROC_BIND OMP_PLACES OMP_SCHEDULE OMP_DYNAMIC OMP_WAIT_POLICY \
  OMP_MAX_ACTIVE_LEVELS OMP_NESTED OMP_STACKSIZE OMP_CANCELLATION OMP_DISPLAY_ENV
unset GOMP_CPU_AFFINITY GOMP_STACKSIZE GOMP_SPINCOUNT

# FP16 baseline
/home/svt/ncnn-k1x-int8-smoke/bin/yolo11_fp16 \
  --param /home/svt/ncnn-k1x-int8-smoke/models/yolo11n.ncnn.param \
  --bin /home/svt/ncnn-k1x-int8-smoke/models/yolo11n.ncnn.bin \
  --image /home/svt/ncnn-k1x-int8-smoke/models/photo_2024-10-11_10-04-04.jpg \
  --bench-only --pin cluster0 --threads 4 --warmup 10 --runs 100 --repeats 5 --strict-omp-env 1 --quiet --no-gui

# INT8
/home/svt/ncnn-k1x-int8-smoke/bin/yolo11_int8 \
  --param /home/svt/ncnn-k1x-int8-smoke/models/yolo11n-int8.ncnn.param \
  --bin /home/svt/ncnn-k1x-int8-smoke/models/yolo11n-int8.ncnn.bin \
  --image /home/svt/ncnn-k1x-int8-smoke/models/photo_2024-10-11_10-04-04.jpg \
  --bench-only --pin cluster0 --threads 4 --warmup 10 --runs 100 --repeats 5 --strict-omp-env 1 --quiet --no-gui
```

## Performance note (historical)
- Early GUI runs were noisy and included I/O and post-processing. Use the deterministic forward-only harness for benchmark numbers.
