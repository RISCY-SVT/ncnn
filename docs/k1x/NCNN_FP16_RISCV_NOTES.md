# NCNN FP16 + RVV Notes for RISC-V (SpacemiT K1X)

_Last updated: 2026-01-03_

This document records **stability and quality fixes** required to build and run ncnn on
SpacemiT K1X (RV64GCV + RVV 1.0 + FP16 `zfh`/`zvfh`) with the SpacemiT GCC cross-toolchain.

It is intentionally focused on:

- FP16 type compatibility (`__fp16` vs `_Float16`)
- RISC-V backend warning/UB fixes we validated
- the deterministic YOLO11 harness used for performance work
- how this maps to our Codex skills and reproducible workflows

---

## 1. FP16 type compatibility: `__fp16` vs `_Float16`

### 1.1. Symptom

Some upstream ncnn code paths use `__fp16` (Arm-centric), which can be unavailable or behave
differently on RISC-V GCC, while `_Float16` is available when FP16 extensions are enabled.

### 1.2. Fix (canonical)

On RISC-V builds with FP16 enabled (`__riscv_zfhmin` and/or related macros), we provide a guarded
typedef that maps `__fp16` to `_Float16` in a way that:

- preserves source compatibility with existing ncnn code
- keeps compilation portable across targets
- avoids violating ABI assumptions

**File(s):**

- `src/mat.h`

---

## 2. RISC-V backend: padding safety fix

### 2.1. Symptom

`pad_value` in the RISC-V padding implementation could trigger warnings like:

- `-Wmaybe-uninitialized`

### 2.2. Fix

Initialize `pad_value` safely (zero-initialization) in the RISC-V padding path.

**File(s):**

- `src/layer/riscv/padding_riscv.cpp`

---

## 3. Warning/UB cleanup (quality gate)

We aim for **0 warnings** in both K1X cross-build trees:

- `build-riscv`
- `build-riscv-bench`

Key changes that were validated in our K1X environment:

### 3.1. Portable unused/fallthrough helpers

Introduce portable macros for:

- intentionally unused variables / return values
- intentional `switch` fallthrough

**File(s):**

- `src/platform.h.in` (`NCNN_UNUSED`, `NCNN_FALLTHROUGH`)

### 3.2. Silence intentional fallthrough warnings (RISC-V)

Annotate intentional fallthrough in RISC-V-specific layer code.

**File(s):**

- `src/layer/riscv/shufflechannel_riscv.cpp`

### 3.3. FP16 narrowing warnings

Some RVV/FP16 paths may emit narrowing warnings when converting between float and fp16 storage
types. We silence these only with explicit casts that preserve semantics.

**Representative file(s):**

- `src/layer/riscv/convolution_winograd_transform_packn_fp16s.h`
- `src/layer/riscv/interp_riscv_zfh.cpp`

### 3.4. RUAPU portable `sigaction` initialization

Avoid partial/uninitialized `sigaction` structs by using:

- `memset(&sa, 0, sizeof(sa))`
- `sigemptyset(&sa.sa_mask)`

**File(s):**

- `src/ruapu.h`

### 3.5. GRU accumulation correctness (non-ZVFH path)

Preserve float accumulators in scalar (non-ZVFH) accumulation paths to avoid precision regressions.

**File(s):**

- `src/layer/riscv/gru_riscv_zfh.cpp`

---

## 4. Deterministic YOLO11 harness (canonical baseline)

The canonical reproducible benchmark harness lives at:

- `examples/yolo11.cpp`

Key features:

- in-program CPU pinning:
  - `--pin cluster0|none|list:<cpu_list>`
- strict environment mode:
  - `--strict-omp-env 1` fails fast if `OMP_*` / `GOMP_*` env vars are set
  - strict mode also forbids `--desired-*` overrides
- deterministic benchmarking:
  - `--warmup <N> --runs <N> --repeats <N>`
  - prints mean + stddev across repeats
- quiet mode for performance runs:
  - `--quiet` reduces log noise
- explicit model paths:
  - `--model-dir`, `--model-name`, `--param`, `--bin`
- prints build/runtime INT8 flags for diagnostics

Canonical baseline settings on the board:

- `--pin cluster0 --threads 4`
- `--warmup 10 --runs 100 --repeats 5`
- `--strict-omp-env 1 --quiet`

---

## 5. Build knobs (K1X cross-build)

We build ncnn with:

- `-DNCNN_RVV=ON`
- `-DNCNN_OPENMP=ON`
- `-DNCNN_XTHEADVECTOR=OFF`
- `-DCMAKE_BUILD_TYPE=Release`

Benchmark build adds:

- `-DNCNN_BENCHMARK=ON`

INT8 coverage build adds:

- `-DNCNN_INT8=ON`

Build orchestration is typically done via `/data/build_scripts/05-build-opencv-ncnn.sh`, or via
Codex skills (see next section).

---

## 6. Codex skills mapping (recommended operator path)

This project maintains custom Codex skills (installed to `/data/.codex/skills` and `/etc/codex/skills`)
to keep builds and runs reproducible:

- `$k1x_env_sanity` – toolchain/sysroot/board readiness
- `$k1x_cross_build_ncnn` – cross-build + warning scan
- `$k1x_deploy_and_run_yolo11` – deterministic FP16 benchmark on the board
- `$k1x_layer_hotspots_report` – per-layer benchmark + Top-N extraction
- `$k1x_int8_quantize_yolo11` – INT8 table + int8 conversion with fail-fast validation
- `$k1x_git_hygiene` – repo hygiene / `.gitignore` / secret-like scan

See `k1x-env-overview.md` for acceptance-test references.

