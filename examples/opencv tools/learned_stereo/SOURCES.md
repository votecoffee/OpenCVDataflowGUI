# Sources and changes

- Original paired ramp images, arithmetic ONNX controls, masks and graphs are MIT
  work by OpenCVDataflowGUI contributors. Their red-channel difference represents
  known disparity. They verify contracts, not learned matching accuracy.
- Pretrained **HITNET ETH3D**, Google Research, converted to ONNX by **PINTO0309**.
  [Original model and method](https://github.com/google-research/google-research/tree/master/hitnet),
  [publisher's conversion package](https://github.com/PINTO0309/PINTO_model_zoo/tree/d2bd6ff715f7b3b9afcc6d3141bce2dd6e701597/142_HITNET).
  The conversion package publishes an Apache 2.0 license; its exact notice is
  retained in `MODEL-LICENSE.txt` (11,358 bytes, SHA-256
  `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`).
  Unchanged archive member
  `eth3d/saved_model_480x640/model_float32.onnx`, 3,548,840 bytes, SHA-256
  `8c5505ab77f83c7e9aee6901b2a6647fa1dcfc57e5775c5a903c6b555c210cb2`.
  Archive: `https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/142_HITNET/resources.tar.gz`.
  No model conversion, training or remote Python execution is performed here.
  Input `input` is [1,2,480,640], unit grayscale left then right; output
  `reference_output_disparity` is [1,480,640,1], in network-image pixels.
  The installed New CPU engine executes it. Classic's importer rejects GatherND
  for this exact artifact; the example explicitly requests New without retrying.
- Public industrial texture: **T-LESS**, T. Hodaň, P. Haluza, Š. Obdržálek,
  J. Matas, M. Lourakis and X. Zabulis, *T-LESS: An RGB-D Dataset for 6D Pose
  Estimation of Texture-less Objects*, WACV 2017.
  [Dataset](https://cmp.felk.cvut.cz/t-less/),
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  Scene 6/frame 2 original 720x540 color image SHA-256
  `04b19c87d60aaabe98c6be9655a58c5cc620ac31dc356ede55e89dc02cd2a166` is
  already attributed in the adjacent monocular-depth example. Here it is resized
  to 320x240 with area interpolation. Right images are copies shifted left by
  0, 8 or 16 pixels with a black uncovered strip. These are artificial known
  disparity planes using real texture, **not actual stereo acquisitions**.
  The original measured depth is not used; the artificial shifts do not represent
  the photographed objects' real depths. No customer assets or private plugin
  work are included.

`truth.json` records exact generated asset/model/contract hashes. `benchmark.json`
reports native errors against the known artificial disparities on a fixed
interior, excluding introduced boundaries. Preview normalization affects display
only. The synthetic rig in `plane.json` has a declared 0.1 m baseline and known
intrinsics for independent XYZ checks. All saved projects embed attribution.
