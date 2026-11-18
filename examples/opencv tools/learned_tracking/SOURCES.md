# Sources and qualified tracking scope

- **ViTTrack**, contributed by Pengyu Liu as part of *Realtime object tracking
  models*, distributed by OpenCV Zoo under Apache 2.0.
  [Publisher's model, implementation references and license](https://github.com/opencv/opencv_zoo/tree/25f423d0e04c31a17254620e58febd7386da523b/models/object_tracking_vittrack).
  Unchanged `object_tracking_vittrack_2023sep.onnx`, **714,726 bytes**, SHA-256
  `2990f0b7cd44d92afa48cd97db6de7be113fc1d9594fddb74e2725c10478e91d`.
  The small native ONNX is included for offline execution; each sealed variant
  contains identical weight bytes. Exact upstream `LICENSE` is retained as
  `MODEL-LICENSE.txt` (11,358 bytes, SHA-256
  `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`).
  There is no conversion, training, remote Python execution or framework install.
- OpenCV's native `TrackerVit` owns template/search crops, forward execution,
  Hanning-window response, box decoding and its score threshold. The wrapper
  exposes native mean/std, score threshold and DNN import/backend/target controls.
  [Native implementation](https://github.com/opencv/opencv/blob/5.x/modules/video/src/tracking/tracker_vit.cpp).
  Native defaults are retained; software acceptance is measured against the
  installed OpenCV 5 build, not inferred from a different source revision.
- Public image: **T-LESS**, T. Hodaň, P. Haluza, Š. Obdržálek, J. Matas,
  M. Lourakis and X. Zabulis, *T-LESS: An RGB-D Dataset for 6D Pose Estimation
  of Texture-less Objects*, WACV 2017.
  [Dataset](https://cmp.felk.cvut.cz/t-less/),
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  Original scene 6/frame 2 color SHA-256
  `04b19c87d60aaabe98c6be9655a58c5cc620ac31dc356ede55e89dc02cd2a166`
  is reused from the attributed adjacent monocular-depth example. Here it is
  resized to 360x270 and shifted by (2*i,i), with black outside support. The
  declared evaluation ROI [92,133,43,43] follows one published part box after
  rescaling/rounding; its translated boxes are exact artificial annotations,
  not pixel segmentation or physical dimensional ground truth. Blank frames are
  deliberately inserted, and the cropped template is used for explicit reseeding.
  These are derived artificial clips, not recorded camera motion/occlusion data.
  No customer images, private plugins or measured depth are included.

Original graph/build/benchmark authoring and annotations are MIT work by
OpenCVDataflowGUI contributors. `truth.json` retains all image/model/contract hashes.
Each project contains attribution on the canvas. `benchmark.py` independently
calls the native tracker and evaluates half-open box IoU/center errors against
annotations; scores are native window-weighted responses, not probabilities.
