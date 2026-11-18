# T-LESS measurement workflow data

T. Hodaň, P. Haluza, Š. Obdržálek, J. Matas, M. Lourakis and X. Zabulis. T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects. WACV 2017.

[Publisher, paper and license](https://cmp.felk.cvut.cz/t-less/) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [Official mirror](https://huggingface.co/datasets/bop-benchmark/tless).

Pinned mirror revision `5fd309a04476a842d93abfb584fba9ee7caecdf1`. Original scene 2 RGB/depth images and calibration/pose metadata for frames 3, 13 and 24 are preserved under `source/`; SHA-256 provenance is included. The full upstream ZIP was not downloaded.

Derived XYZ, trained reference recipes, result clouds and numerical reports backproject measured first returns. No object mask, CAD or hidden surface points are used. Published annotations supply the measurement fixture in all three graphs; they are not detected poses. Frame 3 is a training replay control. Frames 13/24 are independent captures of the same scene. Rounded rotations are projected onto a proper rotation. The same trained regions/bands and 5-degree parallel guard are reused. No acceptance limits or independent gauge truth are supplied.

Raw and derived dataset material retain CC BY 4.0. Preserve this attribution on redistribution. No endorsement is implied. Builder code uses MIT.
