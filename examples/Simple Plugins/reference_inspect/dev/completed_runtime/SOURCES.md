# Sources, preparation and licenses

## VisA

**Visual Anomaly (VisA)** by Yang Zou, Jongheon Jeong, Latha Pemula,
Dongqing Zhang and Onkar Dabeer, *SPot-the-Difference Self-Supervised
Pre-training for Anomaly Detection and Segmentation*, ECCV 2022.
[Primary dataset/license/citation](https://registry.opendata.aws/visa/) ·
[Author project](https://github.com/amazon-science/spot-diff) ·
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
License/citation verified **2026-11-17**. No new photographs were downloaded.
Photographs, masks, retained learned pixels and derived views/arrays remain CC BY 4.0.

The limited control reuses `pcb2/Data/Images/Normal/0000.JPG` and
`pcb2/Data/Images/Normal/0621.JPG` from `VisA_20220922.tar`: prepared 1404×1070
images resized to 702×535 with OpenCV area interpolation, JPEG quality 94, then
grayscale inspection. Its one-reference model and unchanged default gates are
intentionally insufficient for the independent normal photo; that is a prerequisite
rejection, not a new dataset defect label.

The posed case reuses eight aligned training-normal references and the first
held-out normal/anomaly in the publisher's pinned split. [Retained source manifest](visa_posed/source_manifest.json)
records every original member, download hash, split assignment, publisher revision
`2a692ab575001cbde74d402d897a7286086c6199` and mirror revision
`912ae1aa35ac0791e64524a42342850af1605363`. Its selected inputs are:

- `test-normal-0011.jpg`: `pcb2/Data/Images/Normal/0011.JPG`.
- `test-anomaly-000.jpg`: `pcb2/Data/Images/Anomaly/000.JPG`.
- `truth-anomaly-000.png`: publisher-provided evaluation mask (exact original
  member and source hash in the retained manifest).

[Original preparation/source notes](visa_posed/SOURCES.md) detail the resized
photographs, nearest-neighbor evaluation masks, aligned good-sample support and
application masks. The retained `upstream-evaluation.json` describes the original
eight-frame study, not a new eight-frame evaluation by this review increment.
`native-verification.json` records this increment's two selected frames and their
labels/evaluation-only mask metrics. No test labels/masks teach or drive inference.
Other original assets remain available for the retained training/evaluation set.

The existing false alarms, unrelated difference flags and incomplete pixel recall
remain errors. Review adds color maps, source/reference comparisons, diagnostic
tables, native arrays and re-encoded models; no accuracy improvement or endorsement
by dataset authors is claimed. Source/reference/display grids and original pins
remain explicit in the JSON exports.

## Generated geometry and code

Generated nominal/changed/translated parts and authored code retain the supplied
[MIT license](LICENSE). The material-change control has a known 754 reference-pixel
rectangle. Fixture translations are generated software controls, not measured
optical pose ground truth. Bundled wheels retain their license notices and require
no private source-repository connection.
