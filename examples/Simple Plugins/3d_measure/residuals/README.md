# Residual review examples

Requires Simple3DMeasure **0.1.0.dev5**. These projects retain the earlier
trained recipes and measured source assets byte for byte, with unchanged
numerical reports. They enable `residual_graphics` and select the native
**Cloud / residuals** flyaround thumbnail/popout.

| Project | Expected evidence |
| --- | --- |
| [Good step](good/project.json) | Valid 5 mm step; small signed plane residuals. |
| [Raised feature](raised/project.json) | Failed surface range; excluded raised points remain visible, with 76 samples above the top region's color range. |
| [Missing surface](missing/project.json) | Top residuals remain unavailable, with no fabricated zero histogram. |
| [Line traces](line_good/project.json) | Unsigned radial residuals and the original guarded 4 mm separation. |
| [Nonparallel traces](line_nonparallel/project.json) | Small line-fit residuals coexist with unavailable guarded separation; good fit is not measurement acceptance. |
| [T-LESS frame 13](tless_frame13/project.json) | Real first-return plane data; part-region tails remain visible in counts. |
| [T-LESS frame 24](tless_frame24/project.json) | Same reference bands and regions on another real capture. |

Open training and choose **Review trained / completed run… → Residuals / all
selected**. Select a region, or double-click its table row, to show the physical
histogram, fit-band markers and all-selected statistics. Capture a completed
run to compare it with the reference at the same taught-band scale. Layer/view
changes reuse frozen fits. Fit evidence and source appearance remain separate
views; Cancel preserves active parameters.

Plane colors use -3/-1/0/+1/+3 fit-band anchors: blue/cyan/gray/orange/red.
Line colors use unsigned 0/+1/+3 anchors: gray/orange/red. Endpoint colors
saturate; histogram tails count values outside the color range. Magenta means
unavailable selected samples. Counts and exports include fitting exclusions.
Residuals are distances to the current fitted primitive, rather than deviations
from the original reference surface or dimensional uncertainty. Fit bands and
acceptance limits have not been changed to produce these views.

Each folder contains native RGB PLY, a computed cloud PNG, per-region histogram
PNGs, full source-row residual NPZ, raw summary and hashes identifying its
original project/recipe. Display sampling preserves extrema; exported arrays
retain every selected row. Invalid fits have selected row IDs and no value
array. [Definitions, export format and limits](../RESIDUALS.md).

The five generated controls contain known visible surface/line samples. The
two real projects use the pinned T-LESS captures and supplied fixture poses.
They have no independent dimensional gauge truth or configured acceptance
limits; successful execution does not establish production accuracy.

T. Hodaň, P. Haluza, Š. Obdržálek, J. Matas, M. Lourakis and X. Zabulis,
*T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects*, WACV 2017.
[Official source/license](https://cmp.felk.cvut.cz/t-less/) ·
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) ·
[Attribution and pinned raw sources](../real_tless/ATTRIBUTION.md).
Real source and derived dataset material retain CC BY 4.0. No endorsement is
implied. Generated fixtures and builder code use MIT.

Rebuild into a fresh folder with the host/plugin and an offscreen Qt font available:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python build_residuals.py . new_residuals
```
