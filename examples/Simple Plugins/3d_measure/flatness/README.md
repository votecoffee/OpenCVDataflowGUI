# Minimum-zone flatness examples

Requires Simple3DMeasure **0.1.0.dev3** and its optional SciPy dependency. The
host environment already includes SciPy. [Definitions and budgets](../FLATNESS.md)
explain the numerical method and its limits.

| Generated project | Minimum zone (mm) | Least-squares range (mm) | Outcome |
| --- | --- | --- | --- |
| [Independent good sample](generated/good/project.json) | 0.4000 | 0.5799 | Within the 0.45 mm flatness limit. |
| [Deep feature](generated/deep_feature/project.json) | 0.5500 | 0.7299 | Outside the same limit. |
| [Geometry budget](generated/geometry_budget/project.json) | Unavailable | 0.5799 | Four hull vertices are deliberately insufficient; no substitute value is reported. |
| [Missing surface](generated/missing_surface/project.json) | Unavailable | Unavailable | Ten observed points do not establish the trained datum. |

The control samples have one height at each distinct XY coordinate. Three
upper witnesses surround the lower witness in projection, with a broad enough
footprint to establish a known 0.4 mm minimum width. The independent interior
samples favor a tilted datum, producing a larger residual range. The deep
witness increases the known minimum zone to 0.55 mm. Extreme points excluded
from robust datum fitting still determine flatness. These are numerical
controls with known finite-point geometry, not calibrated sensor/gauge tests.

| Real T-LESS capture | Table minimum zone / range (mm) | Part region minimum zone / range (mm) |
| --- | --- | --- |
| [Frame 3](real_tless/frame000003/project.json), training replay | 7.2180 / 7.6225 | 16.6079 / 16.9063 |
| [Frame 13](real_tless/frame000013/project.json), independent | 6.5961 / 6.8480 | 20.2581 / 21.0517 |
| [Frame 24](real_tless/frame000024/project.json), independent | 7.4170 / 7.5836 | 21.1625 / 22.0731 |

The real projects reuse measured first-return points, original camera metadata
and supplied annotation fixtures. One reference teaches the same fixed table
and part regions; the geometry budget and bands stay unchanged on test views.
Every selected point counts, including datum-fit exclusions. These regions
contain noise and nonplanarity and have no independent dimensional gauge truth
or acceptance limits. A graph pass means values were available, not that the
physical part passed dimensional inspection. Unobserved surface defects cannot
be inferred from the finite point set.

Select **Cloud / fit** for the native flyaround thumbnail/popout. Orange/violet
grids are the two parallel zone planes; crosses mark the source support points.
Yellow dimensions show separation. The blue grid and green/red samples still
describe the robust datum. These graphics deliberately show both definitions.
Use **Train 3D measurements… → Review trained / completed run…** to turn Scene,
Planes or Regions off and focus on Zones/Supports. Layer changes use cached
results and do not rebuild the hulls.

`measurements` and `report` retain method/status, hull budgets and counts,
numeric resolution, verification gap, original support-row IDs/coordinates and
the final normal/offsets. Numeric resolution is not measurement uncertainty.
Recipes containing flatness use version 3 and survive Save As with their
hashes intact. Plane/line-only recipes remain compatible without retraining.

Raw T-LESS files and publisher hashes are in [the sibling source folder](../real_tless/source/).
The selected file hashes and transformations are recorded in
`real_tless/source_provenance.json`. [Author attribution and CC BY 4.0 terms](../real_tless/ATTRIBUTION.md)
apply to the raw/derived T-LESS clouds, recipes and graphics. Tomáš Hodaň,
Pavel Haluza, Štěpán Obdržálek, Jiří Matas, Manolis Lourakis and Xenophon
Zabulis, *T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects*,
WACV 2017. [Publisher and license](https://cmp.felk.cvut.cz/t-less/).
No endorsement is implied. Generated controls and builder code use MIT;
optional libraries retain their [separate licenses](../THIRD_PARTY.md).

Rebuild into a fresh folder with the host/plugin/SciPy installed:

```powershell
python build_flatness.py new_flatness --source real_tless/source
```
