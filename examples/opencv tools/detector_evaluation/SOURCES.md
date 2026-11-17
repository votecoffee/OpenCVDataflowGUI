# Sources and licenses

- Creator: OpenCVDataflowGUI contributors. License: repository MIT license.
- Source: original `proposal_heads.onnx`, `color.png` and `black.png` in
  [dnn_detection](../dnn_detection/README.md). Copied unchanged and checked against
  its asset SHA-256 manifest. No third-party image, tutorial code or model weights.
- The ONNX fixture supplies fixed proposals with image-dependent scores. It is
  not a trained detector. No OpenCV Zoo model is copied or downloaded here.
- Labels, graphs, comments, build code and count/AP references are original work.
  Source truth uses independent exact scaling of declared network box coordinates.
  Deliberate offset/class/negative controls are identified in each project.
- [truth.json](truth.json) records hashes and expected metrics. Every native project
  includes a source/license/scope comment on its layout.

AP/IoU definitions are standard mathematics. The documentation links primary COCO
references to explain the difference from that complete benchmark protocol; no
COCO code, images, annotations or additional package is included.
