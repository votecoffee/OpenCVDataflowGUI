# Sources and attribution

## Shipped controls

The images, image-dependent arithmetic feature/matcher models, known translation,
projects, authoring scripts and tests are original OpenCVDataflowGUI work under
the repository's [MIT license](../../../LICENSE). No training images, learned
weights, customer assets or upstream implementation source are copied into them.
Each project contains an attribution comment. Four assets are pinned in
[truth.json](truth.json); the builder regenerates their exact bytes.

## Native API reference

- OpenCV **5.0.0**, commit `40738fb16ceddb5fb3fea747585f7ce6abb0605b`:
  [ALIKED source](https://github.com/opencv/opencv/blob/40738fb16ceddb5fb3fea747585f7ce6abb0605b/modules/features/src/feature2d_aliked.cpp),
  [LightGlue source](https://github.com/opencv/opencv/blob/40738fb16ceddb5fb3fea747585f7ce6abb0605b/modules/features/src/matchers_lightglue.cpp),
  [native regression tests](https://github.com/opencv/opencv/blob/40738fb16ceddb5fb3fea747585f7ce6abb0605b/modules/features/test/test_aliked_lightglue.cpp).
  These establish the ONNX names, native normalization, DMatch score convention
  and CPU reference workflow. Our wrapper and arithmetic fixtures are original.
- [LightGlue](https://github.com/cvg/LightGlue), Lindenberger, Sarlin and Pollefeys,
  ICCV 2023. Its own matcher code/weights have Apache-2.0 terms; extractor terms
  are separate. This increment does not distribute a SuperPoint detector.

## Optional local-only pretrained qualification

`prepare_reference.py` uses the exports referenced by OpenCV-extra's
[download manifest](https://github.com/opencv/opencv_extra/blob/c9cdf09e4d4e5e73effeabce97ef52a5c2789afc/testdata/dnn/download_models.py)
and the image pair from its native ALIKED/LightGlue regression. The
[reference-generation note](https://github.com/opencv/opencv_extra/blob/c9cdf09e4d4e5e73effeabce97ef52a5c2789afc/testdata/dnn/generate_aliked_lightglue_references.md)
warns that resizing and engines affect reference values. Our qualification uses
the installed native New CPU path and independent direct ONNX preparation.

- Export snapshot:
  [YangGuanyuhan/lightglue_opencv_project at 9bf886bf](https://github.com/YangGuanyuhan/lightglue_opencv_project/tree/9bf886bf93a505ab1cf841152ede7fe18bd2351a/model),
  `aliked-n16rot-top1k-640.onnx` (6,438,045 bytes) and
  `aliked_lightglue.onnx` (45,837,864 bytes), unchanged.
  That export repository has no license file; no license grant for redistributing
  those exact exported files is inferred from its public availability.
- Local-only images:
  [OpenCV-extra box.png](https://github.com/opencv/opencv_extra/blob/c9cdf09e4d4e5e73effeabce97ef52a5c2789afc/testdata/cv/shared/box.png)
  and [box_in_scene.png](https://github.com/opencv/opencv_extra/blob/c9cdf09e4d4e5e73effeabce97ef52a5c2789afc/testdata/cv/shared/box_in_scene.png),
  unchanged. The exports and these image bytes remain outside tracked examples.

The public [qualification receipt](qualification.json) contains only hashes and
numerical parity measurements. Downloads are explicit and bounded, with differing
existing files preserved. Runtime blocks never download assets. Model rights and
real-scene accuracy remain the application's responsibility.
