# ALIKED extraction and LightGlue matching

These projects run real ALIKED detector/descriptor inference with local weights.
Install the optional environment in [ML setup](../../../docs/ml.md), including
Kornia 0.9 or newer, then prepare the four pinned files from the repository root:

```powershell
.\.venv\Scripts\python scripts/prepare_ml_assets.py --aliked
```

The command prepares n16, n16rot, n32 and the matching LightGlue weights, preserving
existing artifacts. No weights download during execution. The SIFT comparison
also needs the existing `--lightglue` preparation.

| Project | What to inspect |
| --- | --- |
| [translation.json](translation.json) | Native n16 features recover the known +8/+6 source translation. |
| [n16rot.json](n16rot.json) | The rotation-trained preset uses its own pinned checkpoint. |
| [n32.json](n32.json) | The alternate descriptor-head architecture uses its matching local weights. |
| [resized.json](resized.json) | Downscaled inference returns a +16/+12 translation in the original source grid. |
| [masked.json](masked.json) | Source-grid rectangles filter features after native detection. |
| [rgb16.json](rgb16.json) | Tagged RGB16 pixels remain intact while inference uses normalized RGB. |
| [threshold_fallback.json](threshold_fallback.json) | Native threshold 1 still allows mean-score fallback. |
| [strict_score.json](strict_score.json) | A separate hard score floor rejects fallback results; intentional failure. |
| [empty.json](empty.json) | An empty selection intentionally fails matching without a stale pose. |
| [minimum_matches.json](minimum_matches.json) | A high required match count intentionally withholds the pose. |
| [pose_region.json](pose_region.json) | A fitted frame positions a local inspection rectangle. |
| [comparison.json](comparison.json) | Genuine SIFT and ALIKED extraction use the same textured pair. |
| [pcb_photos.json](pcb_photos.json) | Two separate VisA PCB photographs exercise industrial imagery. |

Debug markers color geometric inliers/outliers green/red and feed the final
filmstrip image. Lean retains numeric results/judgments while skipping painting.
Review images can expand grayscale to color; the primary Image keeps its source
layout and depth. Detection probabilities and learned matching confidence are
separate values. Masks/regions filter centers; descriptors still see surrounding
image context.

Synthetic pose origins have 0.5-pixel tolerances, or 1 pixel for the masked case.
Tests also check matched displacements against the independently known shift.
The PCB case checks four approximate visual mounting-hole annotations at 702 by
535, with a 4-pixel maximum projection-error limit. Its initial CPU run has
212 matches/194 inliers and a 1.781-pixel maximum checkpoint error. These are
image-pair controls, not calibrated physical tolerances or general accuracy
claims. Counts can differ across backend versions.

`truth.json` pins all eight image hashes, the four model hashes, geometry and
expected outcomes. The builder copies existing openly attributed images unchanged,
verifies their source hashes and refuses existing destination assets. Tests
regenerate in a temporary folder, compare projects/assets and preserve originals.
JSON/XML/Save As rebase local model references; weights are not published here.

Synthetic images/graphs are original MIT work; PCB photographs retain CC BY 4.0.
ALIKED retains BSD-3-Clause terms and Kornia/LightGlue retain their applicable
notices. Full credits and changes are in [SOURCES.md](SOURCES.md) and on each canvas.
See the [block contract](../../../docs/aliked-features.md) for controls and limits.
