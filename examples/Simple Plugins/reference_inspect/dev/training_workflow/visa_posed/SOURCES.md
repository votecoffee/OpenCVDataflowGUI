# VisA attribution and source pins

**Visual Anomaly (VisA)** by Yang Zou, Jongheon Jeong, Latha Pemula,
Dongqing Zhang and Onkar Dabeer, *SPot-the-Difference Self-Supervised
Pre-training for Anomaly Detection and Segmentation*, ECCV 2022.
[Publisher, citation and CC BY 4.0 license](https://registry.opendata.aws/visa/) ·
[CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/).

Photographs, masks, retained training pixels and derived maps retain CC BY 4.0.
Builder code uses this repository's MIT license. No endorsement is implied.

[source_manifest.json](source_manifest.json) records every original member,
URL, SHA-256 and train/test assignment. The publisher's split is pinned at
[spot-diff revision 2a692ab575001cbde74d402d897a7286086c6199](https://github.com/amazon-science/spot-diff/blob/2a692ab575001cbde74d402d897a7286086c6199/split_csv/1cls.csv).
The selected raw files come from the unmodified
[VisA mirror](https://huggingface.co/datasets/imaadd05/visa-anomaly-detection/tree/912ae1aa35ac0791e64524a42342850af1605363)
at revision 912ae1aa35ac0791e64524a42342850af1605363. The publisher's dataset
license remains authoritative. Original Normal/0000.JPG also matches the
previously pinned original photograph in the host's shared assets.

Changes: resize 1404×1070 photographs to 702×535 using area interpolation,
encode JPEG quality 94, convert to grayscale for training/inspection, align
good samples using measured matcher poses, and add application masks,
predictions, assessment panels and measurement annotations. Evaluation masks
are resized with nearest-neighbor interpolation. Both source and reference
pixel grids are identified in the result exports.
