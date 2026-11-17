# Sources and changes

## Original controls

`reference.png`, `query.png`, their doubled-size versions and RGB16 versions are
original images by OpenCVDataflowGUI contributors, under the repository's
[MIT license](../../../LICENSE). RNG-seeded texture is blurred, then translated
by exactly +8/+6 pixels, or +16/+12 in the doubled grid. No model outputs or
prescribed feature rows are baked into these images. The native networks extract
their features during execution.

## VisA PCB photographs

**Visual Anomaly (VisA)** by Yang Zou, Jongheon Jeong, Latha Pemula,
Dongqing Zhang and Onkar Dabeer accompanies *SPot-the-Difference Self-Supervised
Pre-training for Anomaly Detection and Segmentation*, ECCV 2022.
[Dataset](https://registry.opendata.aws/visa/),
[project](https://github.com/amazon-science/spot-diff),
[dataset license](https://github.com/amazon-science/spot-diff/blob/main/LICENSE-DATASET),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

| Image | Original archive member |
| --- | --- |
| `assets/pcb2-0000.jpg` | `VisA_20220922.tar`: `pcb2/Data/Images/Normal/0000.JPG` |
| `assets/pcb2-0621.jpg` | `VisA_20220922.tar`: `pcb2/Data/Images/Normal/0621.JPG` |

These are unchanged copies of the repository's openly attributed prepared views.
The earlier preparation resized 1404 by 1070 color originals to 702 by 535 with
OpenCV area interpolation and JPEG quality 94. This example performs no further
image changes. The original-file hashes are retained in the
[shared source record](../../assets/photograph-sources.md#visual-anomaly-visa-pcb-photographs).
The new folder's exact bytes are pinned in `truth.json`. The
[existing visual annotations](<../../Simple Plugins/pattern_matching/visa_pcb2/landmarks.json>)
provide the four approximate mounting-hole centers, in upper-left, upper-right,
lower-left and lower-right order. They are visual checkpoints, not calibrated
ground truth or customer part images.

## Learned models and runtime

- Michał Tyszkiewicz, Pascal Fua and Eduard Trulls, **DISK: Learning Local Features
  with Policy Gradient**, NeurIPS 2020. [Source and weights](https://github.com/cvlab-epfl/disk),
  [Apache-2.0 license](https://github.com/cvlab-epfl/disk/blob/master/LICENSE.txt).
  Native extraction uses [Kornia](https://github.com/kornia/kornia), under its
  [Apache-2.0 license](https://github.com/kornia/kornia/blob/main/LICENSE).
- Philipp Lindenberger, Paul-Edouard Sarlin and Marc Pollefeys,
  **LightGlue: Local Feature Matching at Light Speed**, ICCV 2023.
  [Source and weights](https://github.com/cvg/LightGlue),
  [Apache-2.0 license](https://github.com/cvg/LightGlue/blob/main/LICENSE).

The explicit preparation script downloads the depth detector, epipolar detector
and matching LightGlue checkpoint. Their individual SHA-256 values are recorded
in `truth.json` and checked by the projects. They are not redistributed in this
folder. Capturing reference descriptors does not fine-tune these weights.
