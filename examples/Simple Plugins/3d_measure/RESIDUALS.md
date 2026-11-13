# All-selected fit residual review

Simple3DMeasure **0.1.0.dev5** adds native residual color clouds, full-population
histograms and portable source-row residual exports. The numerical measurement
report, trained fit bands and archive formats remain unchanged.

## Meaning and color scale

Plane residuals are signed distances along the taught normal to the **current
fitted plane**. Flipping the normal reverses their sign. Line residuals are
unsigned radial distances to the current fitted line. Both use every selected
point, including points excluded from robust fitting. A residual is not a
deviation from the original reference surface, a minimum-zone flatness value,
a defect score, or a measurement-uncertainty estimate. Uniform plane motion can
have small fit residuals; use step/probe measurements to assess its position.

Colors use the retained fit band, without frame-dependent rescaling:

| Region | Color anchors, in multiples of its taught band |
| --- | --- |
| Plane | -3 blue, -1 cyan, 0 gray, +1 orange, +3 red. |
| Line | 0 gray, +1 orange, +3 red. |
| Invalid fit | Magenta selected samples, no residual values. |

Endpoint colors saturate outside the displayed range. Counts below and above
that range remain explicit. Regions can have different physical fit bands;
the table and histogram show each region's actual `mm` or `m` values. Reference
and completed runs of one recipe use identical per-region scales.

Histograms have 24 linear bins, count all selected residuals and retain both
tails. They show counts, not probabilities; the final bin includes its upper
edge. Statistics include all-selected min/median/max, RMSE and absolute p95,
fit inlier/exclusion counts and unavailable counts. Invalid fits have `null`
statistics/histograms rather than zero-valued residuals.

## Training and native review

Open **Review trained / completed run… → Residuals / all selected**. The native
flyaround viewer keeps the cached grids, ROI boxes, axes, zones and dimensions.
Choose a region, or double-click its table row, to inspect a physical histogram.
Fit inlier/exclusion toggles are disabled in this mode because its population
always includes both. Source appearance and the original fit view remain
separate choices. Histograms are built once per frozen result; changing views,
regions or graphic layers does not refit. Explicit capture replaces the frozen
completed result and its residual cache together.

The cloud display samples at most 4,000 points per region, reserving part of
that budget for extreme residuals and always preserving its min/max witnesses.
Statistics and exports use every selected row. Overlapping regions can color
the same source row differently against different datums; select one region
to interpret it. Display sampling is not an inspection or measurement filter.

**Export visible review…** in residual mode adds `residuals.npz`, per-region
histogram PNGs and hashes/context in `report.json`. NPZ contains a Unicode JSON
manifest, zero-based original scene-row IDs and full physical residual arrays.
An invalid region retains its selected row IDs with no residual-value array.
The manifest binds those rows to the full scene XYZ SHA-256 and declares frames,
units, fit axes, fit bands, definitions and unavailable state. Read with
`numpy.load(path, allow_pickle=False)`; no original source path is required to
interpret the retained arrays. Match the scene hash before relating rows to a
separate capture. Fresh export files are never overwritten.

## Runtime and API

Runtime residual work is **opt-in** through `residual_graphics`. With both it
and `render_graphics` enabled, select **Cloud / residuals** for the native
flyaround thumbnail/popout; `residual_report` exposes regional summaries.
Existing `fit` output and default preview remain unchanged. Host graphics-off
mode or `render_graphics=false` skips residual statistics and cloud composition
entirely; both optional residual outputs then remain unavailable. No new
dependency or training parameter is required.

```python
from opencv_simple_3d_measure import ResidualReview

review = ResidualReview(measurement_result)
summary = review.report("surface")
cloud = review.cloud(region="surface", layers=("planes", "axes"))
review.export("residuals.npz", region="surface")
```

The builder `examples/build_residuals.py <existing-examples> <fresh-output>`
reuses pinned generated controls and real T-LESS projects. It never retrains or
changes their numerical reports, fit bands or acceptance guards. T-LESS data
and derived graphics retain their separate CC BY 4.0 attribution. Physical
gauges, sensor repeatability and independent dimensional qualification remain
roadmap work.
