# Line fitting and measurements

Simple3DMeasure **0.1.0.dev2** adds robust 3D line regions alongside plane
regions. Select samples that represent a line or narrow section; the tool does
not automatically extract a physical edge from a broad surface patch. A recipe
can contain 1–16 total plane/line regions and 1–64 measurements, with the same
unit, fixture, point-budget and worker contracts as plane recipes.

```python
from opencv_simple_3d_measure import LineRegion, Measurement, Measurer, train

lines = [
    LineRegion("datum", (-12, -.5, -.5), (12, .5, .5)),
    LineRegion("target", (-12, 3.4, -.5), (12, 4.6, .5)),
]
recipe = train(reference_xyz, (), [
    Measurement("gap", "line_separation", "datum", "target", lower=3.9, upper=4.1),
    Measurement("angle", "line_angle", "datum", "target"),
    Measurement("span", "line_span", "datum"),
], lines=lines, units="mm", frame="part")
report = Measurer(recipe).measure(test_xyz, units="mm", frame="part").report()
```

| Kind | Definition |
| --- | --- |
| `line_separation` | Nonnegative perpendicular distance from the target's **inlier centroid** to the datum's infinite fitted axis. A maximum angle guard, 5 degrees by default, rejects nonparallel lines. This is not the shortest distance between two arbitrarily intersecting axes. |
| `line_angle` | Acute angle between fitted axis directions, in degrees. |
| `point_line_distance` | Nonnegative perpendicular distance from a declared reference-frame probe to the fitted infinite line. |
| `line_span` | Maximum minus minimum axial coordinate of **observed inliers**. It is not hidden object length. Missing/occluded endpoints can shorten it. |

For compatibility, the serialized primary-region field is still named `plane`
for either primitive type. The training UI filters region choices to match the
selected measurement type. Plane measurements require planes; line measurements
require lines. Mixed pairs are rejected rather than silently reinterpreted.

Line training uses bounded deterministic two-point RANSAC and orthogonal
least-squares refinement on all selected inliers. The reference-only automatic
radial band uses a Rayleigh-noise heuristic after coarse consensus:
`3.5 * median(radial_residual) / sqrt(2*ln(2))`, with numerical/scale floors.
This is a fitting heuristic, not calibrated measurement uncertainty or a
probability. Runtime uses the retained physical band and direction orientation;
it never widens the band on a difficult test scene. Explicit bands are optional.

Reports retain origin, signed direction, selected/inlier/excluded counts,
radial RMSE, all-selected maximum radial residual, observed axial bounds and
endpoints. Distances flag projections outside the observed datum interval;
being inside that interval does not establish continuous support or visibility.
Degenerate, sparse, failed-consensus and unstable fits leave dependent values
unavailable. A fitted line with a short observed span can have a poorly
constrained angle even when its inlier percentage is high.

In **Train 3D measurements…**, choose **New line region** before drawing. The
same XYZ handles, focus/orbit/zoom, Undo/Redo, background training and Stage/Apply
workflow apply. **Flip axis** changes the retained line direction or plane
normal, then requires retraining. The line preview shows its observed fitted
span and +X direction. The result cloud adds a separate **Lines** layer,
distance/span dimensions and angle arcs. Line datum red X follows its direction;
plane datum blue Z remains its signed normal. The `lines` output exposes raw
fit evidence separately from `planes` and the full `report`.

Archives with lines use version 2 and require dev2. Existing plane-only
archives remain version 1, load without retraining, and keep their numerical
report structure/values. Existing projects can expose the new `lines` port
when needed. The CLI accepts a `lines` configuration list; `planes` can be empty.

Generated controls cover a 4 mm separation, 30-degree angle, 5 mm probe,
missing target, nonparallel target, changed separation and shortened span.
Real T-LESS examples fit axes to fixed measured table section strips, using
explicit supplied annotation fixtures. They contain no independent dimensional
gauge truth or automatically detected physical edges. Fixed frame 24 retains
a 12.29-degree angle and an unavailable separation; observed spans vary
substantially across views. These limitations are preserved for qualification.
