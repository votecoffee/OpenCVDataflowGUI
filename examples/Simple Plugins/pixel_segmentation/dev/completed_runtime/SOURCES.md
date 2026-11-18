# Image sources, labels and licenses

## ARMBench tote photographs

Photographs by **Amazon.com, Inc.**, from **ARMBench Sparrow Object Segmentation**,
under **CC BY 4.0**. Chaitanya Mitash, Fan Wang, Shiyang Lu, Vikedo Terhuja,
Tyler Garaas, Felipe Polido and Manikantan Nambi, *ARMBench: An Object-centric
Benchmark Dataset for Robotic Manipulation*, ICRA 2023.
[Publisher](https://www.armbench.com/) · [paper](https://arxiv.org/abs/2303.16382) ·
[pinned distribution license](https://huggingface.co/datasets/amazon/armbench-segmentation/blob/4444a1cc2c592afc3eea2b7da63aff11328e9aea/License-CC-BY.txt) ·
[retained complete license](ARMBENCH-License-CC-BY.txt).
Pinned publisher-distribution license retrieved and checked **2026-11-17**.

Tiny archive SHA-256: `30f2539a2c8ae32ae25000ee9025657e2c49529cf9e513196597f9a94e7ca0d7`.
Original archive members under `armbench-segmentation-tiny/`:

- `P2S0835EOA.jpg`, `9TARTKRF1X.jpg`: original training photographs.
- `OAR1KXNIME.jpg`: separate loaded validation tote.
- `EZ5895NGB8.jpg`: separate empty validation tote.

The existing examples resize 2448×2048 to 612×512 with area interpolation and
JPEG quality 93. A saved blue-dominance channel transform creates the model input.
Application masks/labels are approximate manual polygons drawn for those examples;
the tiny archive contains photographs only. [Retained preparation details](tote/SOURCES.md).
Tote, heatmap-only and unobserved projects reuse the same trained model/assets.
The unobserved project deliberately supplies a source-aligned all-zero validity
mask through a standard threshold block. It is a software unknown-data control,
not a publisher-labeled sensor failure.

## VisA PCB photographs

Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang and Onkar Dabeer,
*SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and
Segmentation*, ECCV 2022. **CC BY 4.0**.
[Dataset/citation/license](https://registry.opendata.aws/visa/) ·
[author project](https://github.com/amazon-science/spot-diff) ·
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Primary license/citation checked **2026-11-17**.

The existing prepared inputs derive from `VisA_20220922.tar`:

| Original member | Original SHA-256 | Role |
| --- | --- | --- |
| `pcb2/Data/Images/Normal/0000.JPG` | `b3f01fc1c1bd43df2c5a1afa63348097f7ecbcc9ebe2fd43bed6f2d074531a37` | Training photo / first replay |
| `pcb2/Data/Images/Normal/0621.JPG` | `d258d60936443d62745e0c208a78ada21227df6b1405bda0967e105b045c4720` | Independent replay |

Original 1404×1070 photographs were resized to 650×495 with area interpolation,
then converted to a grayscale blue-excess channel using saturating `B - R`.
Labels for exposed laminate/background were painted for this application; they
are not official defect masks. The validation labels cover selected pixels only.
PCB and votes-unavailable projects reuse the original model/settings/preparation.

## Review derivations

No new photographs or model training were needed. New outputs add selected
component bounding footprints/centers, colored vote/coverage maps, display crops,
statistics and exact arrays/model bytes with frame metadata. Source image grids,
sampled component grids and assessment display crops remain distinct. Scores are
uncalibrated forest votes. Foreground labels here represent inventory/material,
not defects. Dataset authors do not endorse the tool. Image/prepared-pixel and
derived-review attribution remains CC BY 4.0; authored code is [MIT](LICENSE).
