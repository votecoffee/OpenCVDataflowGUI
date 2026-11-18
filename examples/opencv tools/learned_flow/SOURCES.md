# Sources, model identity and artificial motion

- Original affine ramps, arithmetic ONNX controls and graphs: MIT,
  OpenCVDataflowGUI contributors. For these ramps, translation follows
  `dx = -delta_red/4 + delta_blue/10`, `dy = -delta_blue/5`, with differences
  measured in original uint8 channel units. These are exact arithmetic controls,
  not learned matching models.
- **RAFT**, Zachary Teed and Jia Deng, Princeton University,
  [original method/code](https://github.com/princeton-vl/RAFT), BSD-3-Clause.
  ONNX conversion by Jeong-gi Kwak / PINTO0309, MIT.
  [OpenCV Zoo publication](https://github.com/opencv/opencv_zoo/tree/25f423d0e04c31a17254620e58febd7386da523b/models/optical_flow_estimation_raft)
  retains both exact notices as `BSD-3-LICENSE.txt` and `MITLICENSE.txt` here.
  BSD notice: 1,512 bytes, SHA-256
  `399af7e243d625e8dea2b3f81fec676ea19bacf7d9ce8cfd4ce15530d8a66173`;
  MIT notice: 1,070 bytes, SHA-256
  `7a1e5596775ef6ceb89fe7bc92d71882864641f839ca83c0d5fe97b14937f440`.
  Unchanged Sintel-trained `optical_flow_estimation_raft_2023aug.onnx`:
  **64,119,337 bytes**, SHA-256
  `3b479e0447894245b822bf0b7392155d31ccd22bb9b0ee55503066bf6f6f869a`.
  Publisher revision `25f423d0e04c31a17254620e58febd7386da523b`.
  Inputs `0`/`1` are [1,3,360,480] raw RGB float values in [0,255]. The wrapper
  converts full-scale integer/unit-float images to unit RGB and divides by
  1/255. Output `12006` is [1,2,360,480] first-to-second dx/dy in network pixels;
  `12007` is a separate low-resolution head and is not used by these projects.
  Native engines can return terminal heads in different orders: selection is by
  name, never list position. Inference uses OpenCV; no training framework or remote
  Python code is installed/executed. Model bytes are neither edited nor converted.
- Public industrial texture: **T-LESS**, T. Hodaň, P. Haluza, Š. Obdržálek,
  J. Matas, M. Lourakis and X. Zabulis, *T-LESS: An RGB-D Dataset for 6D Pose
  Estimation of Texture-less Objects*, WACV 2017.
  [Dataset](https://cmp.felk.cvut.cz/t-less/),
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  Scene 6/frame 2 original color SHA-256
  `04b19c87d60aaabe98c6be9655a58c5cc620ac31dc356ede55e89dc02cd2a166`
  is reused from the attributed adjacent monocular-depth example. Here it is area
  resized to 240x180, then nearest-neighbor translated by (4,-2), (-3,2) or (0,0),
  with black outside support. These are artificial motion controls, not recorded
  motion, physical part displacement or occlusion ground truth. No private
  plugin, customer images or measured-depth assets are included.

All projects include an attribution block. `truth.json` records exact asset and
toy contract/model hashes. `benchmark.py` independently prepares/runs the native
model, selects the named full-size head and samples half-pixel centers into source
units. `benchmark.json` separates endpoint error from retained support on an
explicit interior. Flow-file metadata binds exact image/field content and direction;
it does not validate true optical motion or scene visibility.
