# Minimum-zone flatness

Simple3DMeasure **0.1.0.dev3** adds `minimum_zone_flatness`. It finds the
smallest separation of two parallel planes enclosing **every selected point**.
The existing `surface_range` remains the residual range about the robustly
fitted least-squares datum. They are different quantities and remain separate
measurement choices.

```python
from opencv_simple_3d_measure import Measurement, PlaneRegion, Measurer, train

region = PlaneRegion("surface", (-11, -11, -1), (11, 11, 1))
recipe = train(reference_xyz, [region], [
    Measurement("flatness", "minimum_zone_flatness", "surface", upper=.45),
    Measurement("datum_range", "surface_range", "surface"),
], units="mm", frame="part")
result = Measurer(recipe).measure(test_xyz, units="mm", frame="part")
```

Install the optional dependency when needed:

```powershell
python -m pip install ".[flatness]"
```

Ordinary plane/line execution keeps its NumPy dependency. SciPy imports lazily
for the nonplanar hull calculation. The host environment already provides it.
No SciPy source or binaries are vendored; see [dependency notes](THIRD_PARTY.md).

The method constructs the original point-set hull `H`, then the hull of its
pairwise vertex differences `H-H`. The nearest supporting facet of that
symmetric body gives its centered inradius and the original set's minimum
width. This covers vertex/face and edge/edge contacts; testing only original
face normals would miss some solutions. This geometric relationship is
described in [Zhu and Ding, 2002](https://doi.org/10.1016/S0010-4485(01)00137-3).
Hull construction uses [SciPy ConvexHull / Qhull](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html).

Each difference-body facet is checked against original hull-vertex widths.
The final normal is independently checked against every original selected
point. No random jitter, source subsampling, clipping or robust-fit exclusion
is applied to the flatness population. A raised or depressed point excluded
from datum fitting can still determine the minimum zone.

The result retains method/status, source/hull/difference counts, a roundoff
resolution estimate, the final enclosure check gap, an initial candidate upper
bound, normal/origin/offsets, support counts and original source row IDs/points.
The two zone grids and orange/violet support markers appear in native RGB
result clouds. Dimensions show their separation. These grids and marker
crosses are derived graphics; support coordinates remain actual source points.
Changing cached review layers does not solve the geometry again.

`zone_max_vertices` defaults to 256, configurable from 4 to 1024. The worst
pair-difference population is its square; difference-body verification also
caps at 131,072 facets. All selected points still enter the original hull.
Exceeding a budget, missing SciPy, a degenerate footprint or failed numerical
verification leaves flatness unavailable. The initial candidate bound is
diagnostic and is never substituted as a minimum-zone result. No fit band is
widened. Repeat measurements on the same region/budget share one solve;
independent regions use the recipe's bounded worker setting.

Near-planar sets resolved within the estimated numeric resolution retain the
actual candidate width with `resolved_at_numeric_resolution`, rather than a
fabricated zero. Floating-point resolution is not instrument accuracy or a
calibrated uncertainty interval. Convex hulls operate on finite observations;
unseen defects and missing surface coverage cannot be inferred. Select a broad
2D surface region and inspect the returned normal; a thin or wrongly chosen
region can measure a different geometric direction. Gauge-based qualification
and appropriate source filtering/region selection remain necessary.

In **Train 3D measurements…**, choose a plane region and the new measurement
kind. The geometry budget appears only for this kind. Training reports
unavailable reference values with status tooltips. Review the zone/support
layers and independent runs before Stage/Apply. The archive uses version 3;
existing plane-only and line archives remain versions 1/2 without retraining
or changes to their numerical report structure.

The generated control has a known 0.4 mm minimum-width slab despite a larger
least-squares range. Its witnesses have distinct XY locations, representing
one visible height at each sample. A deeper witness produces 0.55 mm. A regular
tetrahedron test independently checks an edge/edge solution. Real examples
reuse attributed measured T-LESS points and supplied fixture poses; they have
no independent dimensional gauge truth. Minimum-zone computation alone does
not establish production metrology capability.
