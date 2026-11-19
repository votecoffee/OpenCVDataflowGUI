# Trained SuperPoint v6 and LightGlue

These projects use a trained detector on original synthetic board images. The
query board is translated by 8 pixels right and 6 pixels down, with a small
contrast and brightness change. The blank image tests a normal inspection
failure.

## Prepare the model

From the repository root, install the ONNX preparation extra and a CPU PyTorch
build if those packages are missing. See [ML setup](../../../docs/ml.md) for
the optional LightGlue installation.

```powershell
.\.venv\Scripts\python -m pip install -e ".[ml-test]"
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python scripts/prepare_superpoint_v6.py
```

This writes a pinned ONNX model, its license, and a receipt to ignored
`.local/models`. Extraction then runs on the OpenCV 5 CPU engine. For
**translation.json**, also install the LightGlue runtime and run the following
once to prepare its matcher weights.

```powershell
.\.venv\Scripts\python scripts/prepare_ml_assets.py --superpoint-matcher
```

## Open a project and run once

| Project | Expected result |
| --- | --- |
| [extract.json](extract.json) | About 190 keypoints. Inspect the feature markers, confidence, and 256-value descriptors. |
| [translation.json](translation.json) | A planar registration with at least 20 inliers. The four board corners agree with the known shift within 1.5 pixels. Match graphics appear in the filmstrip in Debug. |
| [resized_region.json](resized_region.json) | The 320-by-240 source is downsized to 193 by 145 and padded to 200 by 152. Returned keypoints are in source coordinates and inside the selected region. |
| [blank.json](blank.json) | Expected count failure with fewer than four points. The count, diagnostics and source image remain available. |

Lean skips optional drawing and retains numeric features. The repeatability
and alignment checks use the controlled images in this folder. Industrial
detection accuracy depends on the image set and geometry.

The [block documentation](../../../docs/local-features.md) describes the
dense-head layout, preprocessing and selection. Run `build_example.py --output`
with an empty destination to regenerate the source images and graphs.

## Attribution

Original synthetic images and graphs by OpenCVDataflowGUI contributors,
[MIT](../../../LICENSE). The detector checkpoint is the independently trained
SuperPoint v6 release by Rémi Pautrat and Paul-Edouard Sarlin. Its
[release notes](https://github.com/rpautrat/SuperPoint/blob/1411bbd68c50163555d39c1b26e9e046ebd48f27/README.md)
identify the [MIT terms](../../../docs/third-party/superpoint-v6-MIT.txt).
The optional [LightGlue matcher](https://github.com/cvg/LightGlue) is by
Philipp Lindenberger, Paul-Edouard Sarlin and Marc Pollefeys, with
[Apache-2.0 terms](https://github.com/cvg/LightGlue/blob/main/LICENSE).
