# DISK extraction and LightGlue matching

These projects run real learned DISK extraction on each image, followed by
matching LightGlue weights. Install the optional runtime through
[ML setup](../../../docs/ml.md), then explicitly prepare local weights from the
repository root:

```powershell
.\.venv\Scripts\python scripts/prepare_ml_assets.py --disk
```

Use the existing `--lightglue` preparation as well for the SIFT comparison.
Existing weight files are preserved. No model is downloaded during execution.
The projects pin the depth, epipolar and matcher hashes independently.

| Project | What to inspect |
| --- | --- |
| [translation.json](translation.json) | Known +8/+6 pixels, with actual detector/descriptor inference. |
| [resized.json](resized.json) | Downscaled inference maps results back to a +16/+12 source translation. |
| [masked.json](masked.json) | Source-grid rectangles filter corresponding features after native top-N. |
| [epipolar.json](epipolar.json) | The alternate detector checkpoint uses the same descriptor family. |
| [rgb16.json](rgb16.json) | Tagged RGB16 source pixels stay intact while inference uses normalized RGB. |
| [empty.json](empty.json) | A high raw detector threshold yields empty sets and an intentional matching failure. |
| [minimum_matches.json](minimum_matches.json) | A high required match count intentionally fails and withholds the pose. |
| [pose_region.json](pose_region.json) | The fitted pose places a local inspection rectangle. |
| [comparison.json](comparison.json) | Genuine SIFT and DISK extraction use the same textured image pair. |
| [pcb_photos.json](pcb_photos.json) | Two separate VisA PCB photographs exercise real industrial imagery. |

Inspect features, matches, pose and report. Debug overlays mark geometric
inliers/outliers green/red and feed the filmstrip. Lean keeps numeric judgments
and the original image while skipping optional painting. Review images may
expand grayscale to color; the primary Image preserves its original channel
layout and depth. Detector scores are raw values, distinct from learned match
confidence. Mask/Region filters centers after top-N; descriptors see surrounding
image context.

Synthetic pose origins have declared 0.5-pixel tolerances (1 pixel for the
masked control). Tests also compare matched displacements against the independent
known shift. The PCB case uses four approximate visual mounting-hole annotations
per photograph with a 4-pixel maximum projection-error check at 702 by 535. These
are image-pair controls, not physical measurement tolerances. The initially
qualified PCB run had 298 matches/270 inliers and a 3.624-pixel maximum checkpoint
error. Counts can vary across backend versions.

`truth.json` records expected outcomes, geometry, model hashes and eight image
hashes. `build_example.py` refuses existing assets. Automated checks regenerate
in a temporary folder, compare assets/projects and preserve the source collection.
JSON/XML/Save As rebase local weight references without publishing model files.

The synthetic images/graphs are original MIT work. PCB images are CC BY 4.0;
DISK/Kornia/LightGlue retain their Apache-2.0 terms. Full creator, source, license
and change records are in [SOURCES.md](SOURCES.md) and on every canvas.
See the [block contract](../../../docs/disk-features.md) for controls and limits.
