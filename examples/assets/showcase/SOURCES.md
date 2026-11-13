# Real showcase captures

All nine Demos, the Lab color applications and the PLC photo replay use real
captures. No objects or defects were painted into these photos. Code is MIT;
the images and derived previews retain **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**. Licenses were checked at
the publisher pages below on November 13, 2026.

## VisA PCB photographs

Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang and Onkar Dabeer,
*SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and
Segmentation*, ECCV 2022. [Publisher/license](https://registry.opendata.aws/visa/).
The inputs are the already bundled original PCB2 normal photographs 0000 and
0621. Their archive members and original hashes are in
[photograph sources](../photograph-sources.md). We area-resize while preserving
aspect ratio and add black letterbox padding to 480×320. Palette/region choices
are example-authored. Photographic Lab measurements have no colorimeter ground truth.

## ARMBench warehouse photographs

Amazon.com, Inc.; Chaitanya Mitash, Fan Wang, Shiyang Lu, Vikedo Terhuja,
Tyler Garaas, Felipe Polido and Manikantan Nambi. *ARMBench: An object-centric
benchmark dataset for robotic manipulation*, ICRA 2023.
[Publisher/license](https://www.armbench.com/).
Inputs are the prepared JPEG copies from the public
[clear-tote example](<../../Simple Plugins/reference_inspect/armbench_tote/SOURCES.md>).
Its archive/member/original/prepared hash chain remains available there.
We additionally area-resize and letterbox to 480×320. Two empty and three loaded
labels were assigned for these examples; the tiny source archive does not supply
these occupancy labels. A blue-plastic signal and selected cavity demonstrate
clear/occupied verdicts. Small, blue or excluded objects can be missed; touching
inventory can merge. The small fixed-view replay is not an accuracy benchmark.

## TUM recorded camera sequence

Jürgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram Burgard and Daniel Cremers.
*A Benchmark for the Evaluation of RGB-D SLAM Systems*, IROS 2012.
[Publisher/license](https://cvg.cit.tum.de/data/datasets/rgbd-dataset).
The sequence-memory demo reuses the bundled
[freiburg1_rpy excerpt](<../../opencv tools/recorded_camera_stabilization/README.md>):
original RGB movie frames 384–443, resized to 320×240, converted to grayscale
and stored losslessly as FFV1. This demo selects its first twelve frames.
Camera motion is real and contributes to the flow/background outputs.

## T-LESS measured RGB-D

Tomáš Hodaň, Pavel Haluza, Štěpán Obdržálek, Jiří Matas, Manolis Lourakis and
Xenophon Zabulis. *T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less
Objects*, WACV 2017. [Publisher/license](https://cmp.felk.cvut.cz/t-less/).
The cloud demo reuses the already bundled Primesense scene 1/frame 220, original
RGB/depth arrays and publisher camera intrinsics from the
[pinned source](<../../Simple Plugins/3d_locate/real_tless/source/ATTRIBUTION.md>).
Depth scale is 0.1 mm per encoded sample. The demo samples, filters, voxelizes
and estimates normals; it does not claim a detected part pose or robot calibration.

## Reproduce and audit

[sources.json](sources.json) records source/derived image hashes, source sizes,
resize and padding details, dataset credits, and reused video/RGB-D hashes.
The source originals and upstream provenance remain intact in their existing
collections. [build_real_showcases.py](../build_real_showcases.py) rebuilds the
derived images and graphs without downloading data or contacting hardware.
Every showcase graph includes a source/scope Comment on its canvas.

```powershell
python examples/assets/build_real_showcases.py
```

Original generated graphs are retained separately under
[synthetic_graphs](../synthetic_graphs/README.md) for independent regressions.
Redistributed images/derivatives require the creator attribution and license
above; the dataset authors do not endorse these application examples.
