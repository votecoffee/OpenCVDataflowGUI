# Completed Simple3DLocate review

Four saved projects use an explicit checksum-pinned **0.1.0.dev21** plugin.
Load `plugin.py` first through File → Load Python plugin, then open a project.
Restart if another locator version is already imported. No environment package
is installed. Use a current host with paired completed snapshots and independent
Results review, with Debug input/output retention.

| Project | Input and purpose |
| --- | --- |
| [Real accepted pose](../../tless_match/project.json) | T-LESS object 29, measured PrimeSense capture 230 after publisher-supplied segmentation, with camera-depth verification. |
| [Real localization failure](../../tless_no_match/project.json) | T-LESS object 6, measured capture 3 from scene 2, supplied-mask search plus full measured first-return depth. Retains the existing missed localization. |
| [Generated complete observations](generated_complete/project.json) | Two partial/noisy asymmetric instances and clutter. Explicit complete/fused observation control, not a first-surface sensor simulation. |
| [No observed points](empty_control/project.json) | The unchanged object-29 model with an intentionally empty generated scene; must remain an insufficient-data failure. |

Run once, select **Simple3DLocate** and choose **Review completed 3D localization…**
on Results. Native `fit` cloud thumbnails remain primary; the popup uses the same
host CloudViewer with RGB point colors, orbit and Flyaround. Select accepted or
retained rejected evidence, toggle layers, inspect the exact consumed scene/depth
and export. Rejected/alternative orange geometry is not a detection or measured
return. Original camera first-surface observations and units/frame stay explicit.
See [workflow and export semantics](COMPLETED_REVIEW.md).

Recipes, settings, thresholds, model budgets and workers are copied unchanged.
Test truth is evaluation-only and never passed as a matching seed. The empty
control changes only its source cloud. The generated control deliberately retains
complete mode; the two real scenes preserve first-surface acquisition. No hidden
scene surfaces are invented. This improves review, not matching accuracy; an
accepted geometric pose can still be wrong and the real miss remains visible.

`expected.json` retains numerical reports/poses excluding execution seconds.
`native-verification.json` records source/model/default cloud/chart hashes and
every frozen export. The examples exercise exact source/expected-model separation,
accepted/rejected selection, native fly-around and untouched primary outputs.
Baseline, source/host native, Save As/reopen and Windows checkout receipts accompany
the final publication. Hands-on operation and independent accuracy remain open.

```powershell
python examples.py verify . path/to/new-verification --native
python examples.py build path/to/public-3d-locate-examples path/to/new-example
```

Run with the host Python from this folder. Output directories must be new.
Code/generated controls are [MIT](LICENSE). Real clouds/recipes/derived displays
retain [T-LESS CC BY 4.0 attribution](SOURCES.md).
