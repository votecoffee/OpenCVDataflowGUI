# Sources, model quantities and reference depth

- Ramp/hole images, arithmetic ONNX models and graphs are original MIT controls
  by OpenCVDataflowGUI contributors. The normal head computes `1 + 2 * R/255`;
  the hole head computes `R/255`. These are numerical/geometry controls, not
  trained depth estimators. Their declared metric quantities and inverse-depth
  conversions have independent known values.
- Optional pretrained model: [Intel ISL MiDaS v2.1 release](https://github.com/isl-org/MiDaS/releases/tag/v2_1),
  `model-small.onnx`, 66,764,249 unchanged bytes, SHA256
  `2d8c6cb8f415229daf1eb041024208e2608c9f98e17c81cc7c6ecb449c56fd58`.
  MIT; retained `MODEL-LICENSE.txt` is the exact 1,098-byte upstream notice,
  SHA256 `99ec0b9f9bcc9234b649787b8f03b07dbece95764b7879e1e72fb76cb0f96876`.
  Source/license revision `454597711a62eabcbf7d1e89f3fb9f569051ac9b`.
  [Author's ONNX input pipeline](https://github.com/isl-org/MiDaS/blob/454597711a62eabcbf7d1e89f3fb9f569051ac9b/tf/run_onnx.py)
  and [RGB unit conversion](https://github.com/isl-org/MiDaS/blob/454597711a62eabcbf7d1e89f3fb9f569051ac9b/tf/utils.py)
  inform explicit 256x256 cubic RGB preparation. The actual export contains
  its mean/std normalization; wrapper mean zero/std one avoid applying it twice.
  Native input `0` is [1,3,256,256], output `797` is [1,256,256]. The output is
  relative inverse depth, not meters. Our source-grid interpolation is explicit
  bilinear rather than the author's cubic display resize.
- Real RGB and measured first-return depth: **T-LESS**, scene 6/frame 2,
  original registered 720x540 planes, CC BY 4.0. Credit T. Hodaň, P. Haluza,
  Š. Obdržálek, J. Matas, M. Lourakis and X. Zabulis,
  *T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects*,
  WACV 2017. [Dataset](https://cmp.felk.cvut.cz/t-less/),
  [license](https://creativecommons.org/licenses/by/4.0/).
  These reuse the already attributed public dataset bytes in this repository;
  no private plugin/model/customer assets are copied. Color SHA256
  `04b19c87d60aaabe98c6be9655a58c5cc620ac31dc356ede55e89dc02cd2a166`;
  depth SHA256 `ffc9e6045fbb77da82f9620b4d5e8d086b7d13a74a2360f7d6c7a446db20cc56`.
  Depth samples scale by 0.0001 meters; zero is invalid. Camera parameters and
  unit conversion are retained in `assets/tless_camera.json`. Image pixels are
  unchanged; previews, inference and the diagnostic benchmark are derived work.

`truth.json` retains all public image/model/contract hashes. `benchmark.py`
independently runs native MiDaS and uses a separate bilinear sum to verify
source-coordinate samples. Its near/far test uses measured pairs separated by
at least 20 mm. Its inverse-depth scale/shift fit uses the left image half at
eight-pixel stride; it scores the right half of the same frame and compares a
constant median-depth baseline. This is **evaluation-only alignment**, not a
runtime calibration, independent acquisition or proof of physical accuracy.
Reference depth is never supplied to inference. All graphs embed attribution.
