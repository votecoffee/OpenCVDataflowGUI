# Calibrated training-data and label review

Open [review/review.html](review/review.html) for filtered filmstrips, original
images, 25–800% pixel zoom and **Center part**. Open `review_project.json` in
OpenCVDataflowGUI for four colored fly-around clouds beside their label images.
No optional plugin or neural model is needed to read that saved project.

These are **ground-truth label reviews**, not predicted poses. No network was
trained. Yellow image contours show the full camera-facing CAD silhouette;
cyan contours show the visible annotation. White/RGB markers show the original
CAD-origin pose. Yellow cloud points are clipped to the visible label, so hidden
CAD surfaces do not fill the scene. RGB cloud axes are annotation graphics.

The slice retains 16 original captures: eight synthetic PBR images for training
and eight real Primesense images from separate official training acquisitions
for validation. Objects 1 and 6 each have six usable training labels and four
validation labels. Two fully hidden instances are retained as unusable. The
check status is `needs_review` because these are small example counts.
Benchmark test data is excluded. This does not qualify model accuracy or picking.

`dataset.json` is the portable definition. `source/` retains 69 unchanged source
members and `provenance.json` records the publisher revision, archive metadata,
selected-member CRC checks and SHA-256 hashes. Whole archives were not downloaded
or claimed to be locally rehashed. `review/report.json` records every checked
label, group/scene/RGB leakage checks, units, source hashes and raw viewpoint bins.
The original BOP box spans and inclusive-pixel crop rectangles are separate.

## Replay without a neural runtime

Create a separate Python 3.13 environment. The included wheel's `data` extra
uses NumPy, OpenCV, Pillow and trimesh. It does not install Torch or FoundationPose.
The checked versions are retained in `validation.json`; the older inference
runtime examples keep their original locks and wheels.

```powershell
py -3.13 -m venv .local/data_review
$dataPython = ".local/data_review/Scripts/python.exe"
& $dataPython -m pip install "wheel/opencv_simple_gdrnpp-0.1.0.dev0-py3-none-any.whl[data]"
& $dataPython -m opencv_simple_gdrnpp.cli check-data dataset.json --output .local/new_review
```

Choose a fresh output directory. Existing source files and recipes are preserved.
The native saved-project PASS means files were read, not that a pose was learned.
The next training milestones are symmetry-aware dense targets, actual training
and held-out validation, checkpoint export, and a native training workbench.

## Attribution

T. Hodaň, P. Haluza, Š. Obdržálek, J. Matas, M. Lourakis and X. Zabulis,
**T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects**,
WACV 2017. [Publisher and CC BY 4.0 license](https://cmp.felk.cvut.cz/t-less/).
The [BOP mirror](https://huggingface.co/datasets/bop-benchmark/tless) also declares
CC BY 4.0. Original selected RGB/depth/mask/CAD and annotation bytes retain those
terms. Role selections, metric conversions, review rasters and diagnostic clouds
are modifications. No endorsement is implied. The wheel carries its source
licenses and third-party notices; no weights are bundled.
