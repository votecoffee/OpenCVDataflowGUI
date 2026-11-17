# Completed SimpleClassify review

Three portable projects exercise completed-image review with unchanged trained
recipes and real, labeled VegQual photographs. Load **plugin.py first** through
File → Load Python plugin, then open a project. Restart before loading if another
classifier version was already imported. The checksum-pinned 0.3.2 wheel loads
directly; no environment installation is needed. Use a current host with paired
completed snapshots and an independent Results review action, plus Debug retention.

| Project | Scope |
| --- | --- |
| [Surface texture](vegqual/surface_texture.json) | 16 held-out images, four classes; texture-exemplar evidence. |
| [Color and layout](vegqual/color_and_layout.json) | The same 16 images/split with the original color/layout forest. |
| [Rendering disabled](vegqual/no_overlay.json) | One real test image, unchanged numerical scores and no runtime chart; Results builds its display on request. |

Run once to step through each sequence, select the classifier and open
**Review completed classification…** from Results. The exact consumed input,
full candidate table, winning/runner-up separation and confusion are frozen
together. Switch image/chart tabs, zoom/pan, select a candidate and export.
Training filmstrips and unapplied settings remain independent. Save As includes
the exact model and its labeled training images. See [workflow](COMPLETED_REVIEW.md).

The saved held-out results remain **10/16 correct for Surface texture and 8/16
for Color and layout**. This small subset is a demonstration, not independent
accuracy qualification. It retains misclassifications and high-confusion cases.
All scores sum to 100% but are uncalibrated; an unrelated input can still receive
a winner. Confusion is `100 * runner-up / winner`, not probability of error.
No class rejection or product-quality acceptance gate is inferred.

`native-verification.json` records all 33 completed frames; `expected.json` keeps
their full numerical evidence. Nine [frozen exports](snapshots) cover the first
test sample of every class for each profile plus rendering disabled. Each has
native arrays, original executed recipe bytes, full JSON and a readable PNG.
Exports preserve historical training-image paths; use the project, not an export,
to reopen a complete training bundle. Ground truth is evaluation-only.

The final wheel is verified against archived committed source. Native Qt offscreen
Results checks, numerical/default-chart comparison to 0.3.1, and portable Save As
replay receipts are included. These checks do not demonstrate hands-on acceptance.

```powershell
python verify_examples.py . path/to/new-verification --native
python build_examples.py path/to/original-vegqual path/to/new-example
```

Run these scripts with the host Python from this folder. Rebuild outputs must be
new directories. Code is [MIT](LICENSE); images/labels are [CC BY 4.0](SOURCES.md).
