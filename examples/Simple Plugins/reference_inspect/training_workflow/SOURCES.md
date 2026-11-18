# VisA attribution and changes

**Visual Anomaly (VisA)** by Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing
Zhang and Onkar Dabeer, *SPot-the-Difference Self-Supervised Pre-training for
Anomaly Detection and Segmentation*, ECCV 2022.
[Publisher dataset, citation and license](https://registry.opendata.aws/visa/) ·
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The primary registry's license and attribution were checked on **2026-11-18**.
No new dataset archive or photographs were downloaded for this increment.

The posed PCB example reuses eight aligned normal training samples and the
existing publisher-held-out normal/anomaly photographs and evaluation masks.
[Source manifest](visa_posed/source_manifest.json) retains source members, hashes,
publisher split revision `2a692ab575001cbde74d402d897a7286086c6199` and mirror
revision `912ae1aa35ac0791e64524a42342850af1605363`.
[Earlier preparation notes](visa_posed/SOURCES.md) describe resizing 1404 × 1070
photographs to 702 × 535, JPEG quality 94, grayscale conversion, fixture alignment
and nearest-neighbor evaluation masks.

Additional derivatives are an authored translated reference-image TIFF, an
aligned-support graphic, cached inspection comparison/map/assessment exports,
native arrays and the gallery. The translation is a software alignment control,
not measured camera-pose truth. It is removed before training. The trained arrays,
original health, masks and settings are unchanged; only source-description metadata
is added. All photograph, mask, retained learned-pixel and derived-image content
remains CC BY 4.0. No endorsement is implied.

The PCB false alarms, unrelated flags and incomplete pixel recall remain current
limitations. Held-out inputs and ground-truth masks do not teach the tool. The
retained `upstream-evaluation.json` is the earlier evaluation, not a new full-set
accuracy study. Current receipts describe the native workflow and inspected image.

The generated part/material-change data, translation controls and authored code
use the included [MIT license](LICENSE). The bundled wheels include their own
license notices. Generated controls are not physical process qualification.
