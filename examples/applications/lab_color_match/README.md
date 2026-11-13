# Lab color matching on a real PCB photograph

Open [lab_color_match.json](lab_color_match.json) and run once. The input is a real VisA PCB2
photograph, with reference and inspected substrate regions on the board.
The measured CIEDE2000 difference is approximately **5.353**, below the illustrative
maximum of 15. The match graph exposes the two measured Lab values. The companion
[palette example](../lab_color_classification/README.md) classifies the blue substrate.

Edit the regions to inspect another material on the board. Illumination and camera
response affect these values; the image has no colorimeter calibration or
manufacturing pass/fail ground truth. The original synthetic color graph is retained
under [regression fixtures](../../assets/synthetic_graphs/README.md).

VisA, CC BY 4.0: Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang and
Onkar Dabeer, ECCV 2022. The image was resized/letterboxed; source/hash/change
details are in [showcase attribution](../../assets/showcase/SOURCES.md).
