# VisA PCB2 photograph attribution

**Visual Anomaly (VisA)** by Yang Zou, Jongheon Jeong, Latha Pemula,
Dongqing Zhang, and Onkar Dabeer accompanies *SPot-the-Difference
Self-Supervised Pre-training for Anomaly Detection and Segmentation*,
ECCV 2022. [Dataset and citation](https://registry.opendata.aws/visa/) ·
[Project](https://github.com/amazon-science/spot-diff) ·
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

| Example image | Member of `VisA_20220922.tar` | Use |
| --- | --- | --- |
| `assets/pcb2-0000.jpg` | `pcb2/Data/Images/Normal/0000.JPG` | Training reference and first replay frame |
| `assets/pcb2-0621.jpg` | `pcb2/Data/Images/Normal/0621.JPG` | Separate replay frame |

The example resizes both color photographs from 1404 × 1070 to 702 × 535
with OpenCV area interpolation and re-encodes them as JPEG at quality 94.
The shared assets retain the original JPEG bytes. The recipe uses an
OpenCV grayscale conversion of the prepared reference, and the example's
mounting-hole annotations were made for this project.
[Source files and their SHA-256 hashes](../../../assets/photograph-sources.md#visual-anomaly-visa-pcb-photographs)
