# Compare aligned point-cloud dimensions

Open [cloud_dimensional_comparison.json](cloud_dimensional_comparison.json)
and run it. The measured scan is in meters and the reference is in
millimeters in the same fixture frame. The scans share 30 correspondences.
The measured scan has one extra point and the reference has two points
missing from the measured scan.

| Branch | Expected outcome |
| --- | --- |
| Allow up to 2.1 mm | Pass with all 30 matched rows in tolerance |
| Inspect inward and outward tolerance | Fail with three measured defects at approximately −2.0, −1.5, and +1.2 mm |
| Require 98% coverage | Fail with measured overlap of 30/31 and reference overlap of 30/32 |

The positive sign points along the oriented reference normal in either
direction. Normal mode applies independent inward and outward limits. It can
require a fraction of the overlap to have reliable signs. Equidistant or
duplicate neighbors withhold the target row and signed distance. When both
clouds carry normals, the normal angle limit also governs sign reliability.
Distance mode evaluates Euclidean deviations and can operate without normals.

Pair arrays follow original source row order. A nearest index of −1 means
there is no unique target row. A null Euclidean distance means the point
is outside overlap. States are 0 for unmatched, 1 for in tolerance, 2 for
out of tolerance, and 3 for overlap without a usable signed evaluation.
The diagnostic preview shows inward defects in blue, outward defects in
red, in-tolerance points in green, unmatched points in amber, and unscored
signs in gray. The cloud outputs preserve original attributes.

Nearest-neighbor searches run in bounded batches. The example uses sampled
surfaces with known corresponding positions. Align real scans to a common
frame and choose the correspondence radius and tolerances for the sampling
density. [SciPy nearest-neighbor query](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query.html)

The scans and [builder](build_example.py) are original OpenCVDataflowGUI
contributions under the repository's MIT license.
