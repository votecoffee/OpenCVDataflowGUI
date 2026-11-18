# Image attribution and changes

Licenses were rechecked at the primary publisher pages on 2026-11-18. These
examples reuse prepared public assets; no new dataset download was required.

## VisA PCB2

Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang and Onkar Dabeer,
**SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and
Segmentation**, ECCV 2022. Visual Anomaly (VisA) was accessed from the
[AWS Open Data registry](https://registry.opendata.aws/visa/), which identifies
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) as its license.
[Author project](https://github.com/amazon-research/spot-diff).

The source members are `pcb2/Data/Images/Normal/0000.JPG` (training) and
`pcb2/Data/Images/Normal/0621.JPG` (validation) from `VisA_20220922.tar`.
Existing preparation resized photographs from 1404 × 1070 to 702 × 535 and
encoded JPEGs at quality 94. The blue-excess grayscale PNGs and masks are the
previous example's derived training inputs and example-drawn annotations.

## ARMBench Sparrow Object Segmentation

Photographs by **Amazon.com, Inc.** Chaitanya Mitash, Fan Wang, Shiyang Lu,
Vikedo Terhuja, Tyler Garaas, Felipe Polido and Manikantan Nambi,
**ARMBench: An Object-centric Benchmark Dataset for Robotic Manipulation**,
ICRA 2023. The [publisher](https://www.armbench.com/) identifies CC BY 4.0.
The archive's [distribution license](ARMBench-License-CC-BY.txt) is included.

The tiny archive members `P2S0835EOA.jpg` and `9TARTKRF1X.jpg` provide training;
`OAR1KXNIME.jpg` and `EZ5895NGB8.jpg` provide loaded/empty validation scenes.
The prepared photographs were resized from 2448 × 2048 to 612 × 512 using area
interpolation and JPEG encoded at quality 93. Native derived grayscale PNGs use
the existing blue-dominance transform. This tiny archive contains photos only;
the foreground/background polygons and masks were drawn for the example.
The retained [case credits](tote/SOURCES.md) identify the original pinned archive.

## Additional modifications and code

Retained native PNGs, copied labels/validity masks, trained models, colored
annotation/validation/vote views and the gallery are derived materials. Image
derivatives retain CC BY 4.0. No endorsement is implied. These few annotated
examples are workflow evidence, not a broad accuracy benchmark.

Authored code and the bundled tool retain the included [MIT license](LICENSE).
The wheel includes its own MIT notice. No private implementation-repository
attribution is included in these public examples.
