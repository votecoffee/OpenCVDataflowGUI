# Source photograph and modifications

Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang and Onkar Dabeer,
*SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and
Segmentation*, ECCV 2022. VisA PCB2 normal photograph 0000.
[Publisher/license](https://registry.opendata.aws/visa/),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

We reuse the [bundled 480×320 photograph](../../assets/showcase/visa-pcb2-0000.png).
Its [showcase source record](../../assets/showcase/SOURCES.md) and
[original archive/member/hash record](../../assets/photograph-sources.md)
retain upstream provenance.

OpenCVDataflowGUI contributors crop 260×320 views at x = 0, 110 and 220,
multiply their uint8 BGR values by 1, 0.85 and 0.70 respectively, round, and
save lossless PNGs. Additional copies multiply the resulting samples by 257
and store uint16 PNGs. No extra sensor information is introduced. Original and
derived file hashes and formulas are in [truth.json](truth.json).

Photographs and these derivatives retain CC BY 4.0. Python/project code is MIT.
Dataset creators do not endorse this example. Source overlap and intensity
changes are controlled software fixtures, not independent acquisitions or
physical calibration ground truth. No customer images are included.
