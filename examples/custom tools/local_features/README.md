# External local features and learned matching

These are **interface and geometry controls**, using original synthetic images
and prescribed keypoints/descriptors. The included ONNX file is not a SuperPoint
detector. It exposes fixed coordinates and scores plus random reference
descriptors with a small image-mean-dependent perturbation. This verifies real
tensor flow and learned matching without making a detection-accuracy claim.

Open [pack_only.json](pack_only.json) and run once with the base app environment.
For projects containing a matcher, first install the optional runtime described
in [ML setup](../../../docs/ml.md), then prepare the matcher explicitly:

```powershell
.\.venv\Scripts\python scripts/prepare_ml_assets.py --superpoint-matcher
```

Run that command from the repository root. It downloads **only LightGlue's
matcher**, not a SuperPoint detector. The projects pin its SHA-256. Runtime never
downloads weights. The comparison project additionally needs the existing
`--lightglue` SIFT-matcher preparation.

| Project | What to inspect |
| --- | --- |
| [pack_only.json](pack_only.json) | Base OpenCV ONNX inference, explicit tensor packing, row identities and feature graphics; no optional ML runtime. |
| [translation.json](translation.json) | Matched SuperPoint-family LightGlue weights recover the prescribed +8/+6 transform. |
| [masked.json](masked.json) | Corresponding source-grid selections filter both descriptor sets. |
| [score_filter.json](score_filter.json) | Inclusive score filtering retains corresponding raw indices. |
| [empty.json](empty.json) | Empty retained sets intentionally fail matching without loading weights or emitting a pose. |
| [minimum_matches.json](minimum_matches.json) | Valid correspondences intentionally fail a high count requirement; no pose is emitted. |
| [comparison.json](comparison.json) | The external-feature contract and genuine SIFT/RootSIFT extraction use the same textured image pair and +8/+6 geometry. |

Debug shows match markers and feeds the graphics overlay to the filmstrip. Lean
retains the numeric results and original image while skipping optional painting.
The source images are the same RNG-seeded 200-by-160 pair used by the existing
SIFT regression. Contract-control pose origins are checked within 0.002 pixels;
the genuine SIFT comparison uses 0.2 pixels. Neither tolerance qualifies a real
SuperPoint detector, industrial images, nonplanar scenes or physical metrology.

`truth.json` records judgments and three original asset hashes. The generator
refuses existing asset files. Tests regenerate in a temporary directory, compare
assets/projects and preserve all source files. JSON/XML/Save As rebases local
weight references; pretrained weights are not copied into the public example.

Original images, prescribed feature controls, ONNX arithmetic and graphs are by
OpenCVDataflowGUI contributors under the repository's [MIT license](../../../LICENSE).
Learned matching uses Philipp Lindenberger, Paul-Edouard Sarlin and Marc Pollefeys,
**LightGlue: Local Feature Matching at Light Speed**, ICCV 2023:
[project](https://github.com/cvg/LightGlue),
[Apache-2.0 license](https://github.com/cvg/LightGlue/blob/main/LICENSE).
The [upstream licensing note](https://github.com/cvg/LightGlue#license) identifies
separate restrictive terms for SuperPoint's detector, inference file and weights.
Those assets are not included or downloaded here. Attribution and this limitation
also appear on every project canvas.

The [feature contract](../../../docs/local-features.md) explains shapes,
coordinate conventions, masks, descriptor normalization, provenance and limits.
S2-15 remains **Partial** until a real permitted extractor fixture is qualified.
