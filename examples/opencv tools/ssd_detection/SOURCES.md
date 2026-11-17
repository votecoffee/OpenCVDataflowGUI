# Sources and licenses

## Pretrained SSD ResNet-34

ONNX Model Zoo contributors; upstream training/export work from MLPerf and the
contributors credited in the [pinned model documentation](https://github.com/onnx/models/blob/c5cae0f1942c8d7727372c7b1f6b485aa39e4421/validated/vision/object_detection_segmentation/ssd/README.md).
This export is distributed under **Apache 2.0**, as stated by that model's own
license section; a copy is in [MODEL-LICENSE.txt](MODEL-LICENSE.txt).

- Repository revision: `c5cae0f1942c8d7727372c7b1f6b485aa39e4421`.
- Artifact: `validated/vision/object_detection_segmentation/ssd/model/ssd-12.onnx`.
- Bytes: `80366315`.
- SHA-256: `5ec1e058e666ef980a6213ac40088877865aafb2b6676b9af204b3828e4bb9d5`.
- [Pinned upstream weights](https://media.githubusercontent.com/media/onnx/models/c5cae0f1942c8d7727372c7b1f6b485aa39e4421/validated/vision/object_detection_segmentation/ssd/model/ssd-12.onnx).
- Changes: none. Weights are downloaded explicitly into ignored local storage,
  without retraining, graph rewriting, simplification or conversion.

SSD architecture: Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
Scott Reed, Cheng-Yang Fu, Alexander C. Berg, *SSD: Single Shot MultiBox Detector*,
ECCV 2016. The export uses a ResNet-34 backbone and COCO object classes.

## Coffee photograph

Rachel Michetti, courtesy of Pikolo Espresso Bar; distributed by scikit-image
under **CC0 1.0**. The projects reuse unchanged `examples/assets/coffee.png`.
SHA-256: `cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7`.
See the [scikit-image attribution](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.coffee)
and [pinned source image](https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/data/coffee.png).

The original source-image annotation is `[170,18,245,292]` (XYWH pixel edges),
hand marked independently for this integration check. It is approximate and
covers the cup body and handle, excluding the saucer. Other classes are excluded
from the decoder and this validation set. No training/held-out accuracy claim is made.

## Original project material

Graphs, generation/preparation scripts and annotations are original
OpenCVDataflowGUI material under the repository's
[MIT license](https://github.com/votecoffee/OpenCVDataflowGUI/blob/main/LICENSE).
The unchanged black-image control comes from
[dnn_detection](https://github.com/votecoffee/OpenCVDataflowGUI/tree/main/examples/opencv%20tools/dnn_detection),
also original MIT material. No upstream
training or inference implementation is copied. Source credits and the model
license are declared program assets and accompany a portable Save As.
