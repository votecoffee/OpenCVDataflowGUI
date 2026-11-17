# Sources and changes

## Images and geometry

The eight image files are byte-for-byte copies of the openly attributed
[DISK control collection](../disk_matching/SOURCES.md). Their hashes are checked
before copying and retained in `truth.json`; this example changes no pixels.
The synthetic textures, exact translations and RGB16 versions are original work
by OpenCVDataflowGUI contributors under the repository's
[MIT license](../../../LICENSE).

`pcb2-0000.jpg` and `pcb2-0621.jpg` originate from
`VisA_20220922.tar`, members `pcb2/Data/Images/Normal/0000.JPG` and `0621.JPG`.
**Visual Anomaly (VisA)** is by Yang Zou, Jongheon Jeong, Latha Pemula,
Dongqing Zhang and Onkar Dabeer, accompanying *SPot-the-Difference Self-Supervised
Pre-training for Anomaly Detection and Segmentation*, ECCV 2022.
[Dataset](https://registry.opendata.aws/visa/),
[project](https://github.com/amazon-science/spot-diff),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The earlier preparation resized 1404 by 1070 originals to 702 by 535 with OpenCV
area interpolation and JPEG quality 94. No further changes occur here.
The shared [original-file record](../../assets/photograph-sources.md#visual-anomaly-visa-pcb-photographs)
retains source hashes. The four approximate mounting-hole checkpoints come from
the [existing public annotations](<../../Simple Plugins/pattern_matching/visa_pcb2/landmarks.json>).
They are visual checks, not calibrated ground truth or customer part data.

## Learned models and runtime

- Xiaoming Zhao, Xingming Wu, Weihai Chen, Peter C.Y. Chen, Qingsong Xu and
  Zhengguo Li, **ALIKED: A Lighter Keypoint and Descriptor Extraction Network via
  Deformable Transformation**, IEEE Transactions on Instrumentation and
  Measurement, 2023. [Source/weights](https://github.com/Shiaoming/ALIKED),
  [BSD-3-Clause license](https://github.com/Shiaoming/ALIKED/blob/main/LICENSE).
  The wrapper calls the installed Kornia ALIKED implementation; no upstream
  inference source is copied into this example.
- [Kornia](https://github.com/kornia/kornia) provides the qualified 0.9.0 native
  API and retains its Apache-2.0 and included ALIKED BSD notices.
- Philipp Lindenberger, Paul-Edouard Sarlin and Marc Pollefeys,
  **LightGlue: Local Feature Matching at Light Speed**, ICCV 2023.
  [Source/weights](https://github.com/cvg/LightGlue),
  [Apache-2.0 license](https://github.com/cvg/LightGlue/blob/main/LICENSE).

The explicit preparation script fetches n16, n16rot, n32 and matching LightGlue
checkpoints. Independent SHA-256 values are pinned in `truth.json` and the saved
projects. Model files remain local rather than being redistributed here.
Reference capture records descriptors and does not fine-tune pretrained weights.
