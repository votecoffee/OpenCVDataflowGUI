# Camera-health image source and changes

**Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang and Onkar Dabeer**,
*SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and
Segmentation*, ECCV 2022. VisA PCB2 normal photograph 0000.
[Publisher and license](https://registry.opendata.aws/visa/),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

This example reuses the bundled
[480×320 resized/letterboxed image](../../assets/showcase/visa-pcb2-0000.png).
The [showcase source record](../../assets/showcase/SOURCES.md) and
[original archive/member/hash record](../../assets/photograph-sources.md)
retain the upstream provenance. No additional download is needed.

OpenCVDataflowGUI contributors add synthetic colored moving parts to three
training captures and ten replay cases. Selected cases additionally translate
the photo by an exact number of pixels, multiply intensity by a known gain,
apply Gaussian blur, or externally mark the entire image as occluded.
[truth.json](truth.json) records formulas, regions, all edited PNG hashes,
the bundled source hash and the reference archive hash. The pickle-free
`reference.npz` retains the normalized three-image intensity history, median
image, stable-pixel mask, region statistics and this attribution.

These edited images and normalized reference intensities are derivatives under
CC BY 4.0. Example Python/project code is MIT. No endorsement by the dataset
creators is implied. Changes are controlled software fixtures, **not** measured
physical camera movement, focus calibration or VisA-supplied health labels.
