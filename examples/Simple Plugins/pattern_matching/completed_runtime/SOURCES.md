# Image sources and licenses

## Real VisA PCB2 photographs

**Visual Anomaly (VisA)**, Yang Zou, Jongheon Jeong, Latha Pemula,
Dongqing Zhang and Onkar Dabeer, *SPot-the-Difference Self-Supervised
Pre-training for Anomaly Detection and Segmentation*, ECCV 2022.

- [Dataset, license and citation](https://registry.opendata.aws/visa/)
- [Author project](https://github.com/amazon-science/spot-diff)
- [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

License/citation checked against the AWS dataset entry on **2026-11-17**.
These copies reuse previously prepared public assets; no new dataset download
was needed for this review increment. Source archive: `VisA_20220922.tar`.

| Prepared file under `visa_pcb2/assets` | Original archive member | Use |
| --- | --- | --- |
| `pcb2-0000.jpg` | `pcb2/Data/Images/Normal/0000.JPG` | Reference and first replay |
| `pcb2.reference.png` | `pcb2/Data/Images/Normal/0000.JPG` | Exact prepared grayscale training pixels |
| `pcb2-0621.jpg` | `pcb2/Data/Images/Normal/0621.JPG` | Separate replay frame |

Original photographs were resized from 1404 × 1070 to 702 × 535 with OpenCV area
interpolation, then JPEG-encoded at quality 94. The PNG preserves the prepared
grayscale training pixels. New exports add grayscale fits, colored edges/pose,
inspection coverage, display-only crops and diagnostic text. Exact consumed
prepared pixels are retained in each `consumed_input.npy`; decoded-pixel hashes
and dimensions are in `review.json` and `native-verification.json`.

The pre-existing mounting-hole pose annotations in the saved project are
approximate project annotations, not independent metrology. This increment
adds no new pose ground truth, acceptance labels or accuracy claims. Dataset
authors do not endorse this tool. Derived image exports retain CC BY 4.0.

## Generated controls and authored code

`brackets/assets` and `wired/assets` reuse the authored asymmetric-bracket
reference and generated tray/partial-tray images. `blank/assets/blank.png` is
a constant-value 40 image with the tray dimensions. Unused copied assets retain
their original status. These images, the authored code and bundled tool use
the supplied [MIT license](LICENSE). They are controls, not acquired-camera data.

The bundled wheel has its own MIT notice. [wheel-verification.json](wheel-verification.json)
records its source revision and checksum without requiring a source repository
connection or environment installation.
