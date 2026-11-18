# Sources, licenses and modifications

## VisA PCB2 photographs

Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang and Onkar Dabeer,
**SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and
Segmentation**, ECCV 2022. Visual Anomaly (VisA) was accessed through its
[AWS Open Data entry](https://registry.opendata.aws/visa/); its
[CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/) was checked
again on 2026-11-18. [Author project](https://github.com/amazon-research/spot-diff).

Prepared assets reuse `pcb2/Data/Images/Normal/0000.JPG` and
`pcb2/Data/Images/Normal/0621.JPG` from `VisA_20220922.tar`. The original
1404 × 1070 photographs were resized to 702 × 535 with area interpolation and
JPEG encoded at quality 94. The reference PNG retains the exact prepared
grayscale training pixels. Existing approximate mounting-hole annotations are
example annotations, not independent metrology. No new dataset download or pose
ground truth was added for this increment.

The refined recipe, retained grayscale refinement image, edge/pose overlays,
comparison images and contact sheet are modifications. Derived photographs
retain CC BY 4.0. The dataset authors do not endorse this tool.

## Generated bracket controls and code

The asymmetric reference, tray and partial-tray images are authored generated
controls reused from the existing example. They are not camera acquisitions.
Authored example code, controls and the bundled matcher retain the included
[MIT license](LICENSE); the wheel also includes its MIT notice.

Original and refined recipes use the same existing search settings. Scores from
the refinement image describe adaptation on that image, not independent accuracy
or generalization. No private implementation-repository attribution is included.
