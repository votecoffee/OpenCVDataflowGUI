# SimplePatternMatch selected-result review

Requires **SimplePatternMatch 0.1.0.dev1**. These samples use the existing
generated bracket and real VisA PCB2 recipes without changing their training,
search parameters or acceptance settings.

| Frame | Selected accepted fit | Selected score rejection | Accepted instances |
| --- | --- | --- | --- |
| Generated tray | [Assessment](samples/asymmetric_brackets_0_accepted/assessment.png) | [Assessment](samples/asymmetric_brackets_0_rejected/assessment.png) | 6 |
| Partial tray and decoy | [Assessment](samples/asymmetric_brackets_1_accepted/assessment.png) | [Assessment](samples/asymmetric_brackets_1_rejected/assessment.png) | 4 |
| VisA training frame | [Assessment](samples/visa_pcb2_0_accepted/assessment.png) | [Assessment](samples/visa_pcb2_0_rejected/assessment.png) | 1 |
| Separate VisA frame | [Assessment](samples/visa_pcb2_1_accepted/assessment.png) | [Assessment](samples/visa_pcb2_1_rejected/assessment.png) | 1 |

The accepted view selects the lowest-score returned instance to expose
remaining missing edges. The rejected view selects the first retained
`score_cutoff` candidate. Rejected candidates are not detections or training
labels. Their poses are not known to be correct. Scores are geometric fit
fractions, not calibrated probabilities.

Each sample folder contains the full working-grid **fit.png**, a cropped
display-only **assessment.png**, and **review.json** with pose, source-pixel
mapping, available measurements, actual score/contrast limits and edge
counts. The images distinguish found (green), missing (red), outside or
unmeasured (amber), residuals (dark green) and the pose (cyan). Expected gray
probes are an optional layer. Fit RMS and coverage remain diagnostic values;
this display does not introduce new gates.

Open either saved source project, select the matcher and open **Train
geometric recipe…**. In **Inspect**, capture/open the source image and run
preview. Filter by accepted/rejection reason and select one result. Scroll
to zoom, drag to pan, use **100%** for working pixels or **Focus selected**
for that footprint. Toggle layers without rerunning matching. **Export
selected review…** saves a new folder; it does not alter or stage the recipe.

The original source projects retain their default result images/graphics.
These review samples add assessment views; they do not improve recognition
accuracy. All four frames retained their accepted counts and their previous
default core and host overlays in a sequential before/after comparison.

[samples/receipt.json](samples/receipt.json) records recipe and decoded-image
hashes, source/working dimensions and frame positions. To rebuild to a **new**
directory from the host repository root:

```powershell
python "examples/Simple Plugins/pattern_matching/result_review/build_views.py" .local/new-pattern-review
```

## Attribution

Generated geometry and builder code use this repository's MIT license.
Real VisA photographs and derived fit/assessment images retain **CC BY 4.0**.
Authors: Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang and Onkar
Dabeer, *SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection
and Segmentation*, ECCV 2022.
[Dataset](https://registry.opendata.aws/visa/) ·
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) ·
[full source attribution and preparation](../visa_pcb2/SOURCES.md).
This example adds grayscale selected-edge overlays, a resized crop and
measurement annotations. No dataset-author endorsement is implied.
