# NCNN on K1X (SpacemiT K1) — FP16 Compatibility Notes

_Last updated: 2025-12-19_

This note documents a build portability issue we hit when cross-compiling **ncnn** for the Banana Pi BPI-F3 (SpacemiT K1/K1X) with RVV + FP16 enabled, and the minimal fix that made the build reliable across clean containers.

---

## 1. Problem: `__fp16` is used but not provided by the toolchain (RISC-V)

In the ncnn codebase, `__fp16` appears in:

- RISC-V RVV + FP16 (ZFH/ZVFH) code paths under `src/layer/riscv/…` (fp16 packn kernels, rvv fp16 math helpers),
- ARM/AArch64 fp16 code paths (ASIMDHP / fp16 vector arithmetic),
- a few shared headers.

On **RISC‑V**, `_Float16` is the standard FP16 type, but `__fp16` is not standard and is not guaranteed to exist.  
The SpacemiT GCC we use provides `_Float16` but does not define `__fp16` for RISC‑V, which causes compile failures when RVV+FP16 paths are enabled.

---

## 2. Design goal

Fix `__fp16` compilation on RISC‑V **with minimal changes**:

- do not touch dozens of RVV layer source files,
- do not affect ARM (where `__fp16` is a compiler-provided type),
- do not require global `-D__fp16=_Float16` injection (easy to forget, can conflict).

---

## 3. Fix: a centralized RISC‑V-only typedef in `src/mat.h`

We introduced a small compatibility block in a central header (`src/mat.h`) that is reached by essentially all layer code (`layer.h` → `mat.h`), guarded so it only triggers on RISC‑V when FP16 is actually enabled:

- only on `__riscv`
- only if one of these macros is present:
  - `__riscv_zfh`, `__riscv_zfhmin` (scalar FP16)
  - `__riscv_zvfh`, `__riscv_zvfhmin` (vector FP16)
- only if `__fp16` is not already defined
- only if `__FP16_TYPE__` is not present (avoids collisions with toolchains that already provide a native fp16 type macro)

Pseudo-logic:

```c
#if defined(__riscv)
#  if (defined(__riscv_zfh) || defined(__riscv_zvfh) || ...) && !defined(__fp16)
typedef _Float16 __fp16;
#  endif
#endif
```

This resolves the build without touching the per-layer sources.

---

## 4. Verification steps

### 4.1. Compile-time checks (host)

Ensure your `-march` defines the expected macros:

```bash
echo | riscv64-unknown-linux-gnu-gcc ${K1_ARCH_FLAGS} -dM -E - | rg '__riscv_(vector|zvfh|zfh)'
```

### 4.2. Runtime checks (board)

- Ensure your deployed runtime library directory is visible:

```bash
export LD_LIBRARY_PATH=/home/<user>/<runtime>/lib:$LD_LIBRARY_PATH
```

- If your sample app opens a GUI window and blocks, use `timeout` for headless smoke tests:

```bash
timeout 8s ./your_app <args...>
```

Non-zero exit codes may simply be `timeout` terminating a blocked GUI wait; verify that inference logs are printed before termination.

---

## 5. Related cleanup: `pad_value` initialization warning (RISC-V padding)

During RVV builds we also saw GCC warnings about `pad_value` potentially being used uninitialized in the RISC‑V padding layer (`src/layer/riscv/padding_riscv.cpp`) in the “packn uint16” code.

Fix strategy:

- initialize `vuint16m1_t pad_value` to zero at definition (safe default),
- keep the existing branches that overwrite it for fp16/bf16 paths.

This eliminates the warning and removes a possible UB footgun if future refactors introduce a new path.

---

## 6. Benchmarking note (why timings vary)

One-off GUI runs are noisy because they include:

- image I/O, preprocessing and postprocessing,
- window creation / display sync,
- DVFS and cache state.

Preferred method:

- prepare input once,
- run 10 warmups,
- time 100 “forward-only” iterations,
- report mean and variance.

(Our YOLO11 example was refactored accordingly; see `examples/yolo11.cpp` in our working tree.)

---

## 7. Key takeaways

- On RISC‑V, prefer `_Float16` in new code.  
- For existing codebases that use `__fp16`, a **central guarded typedef** is often the least invasive solution.
- Always confirm RVV flags are applied globally and benchmark forward-only loops for meaningful performance comparisons.
