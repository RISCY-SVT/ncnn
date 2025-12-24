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

## Performance note (historical)
- Early GUI runs were noisy and included I/O and post-processing. Use the deterministic forward-only harness for benchmark numbers.
