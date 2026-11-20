# Compare a measured scan to a reference mesh

Open [scan_mesh_comparison.json](scan_mesh_comparison.json) and run it. The
nominal panel is a 100 mm square at Z = 500 mm, wound toward +Z. The measured
scan is in meters in the same fixture frame. Five scan rows match usable
surface. Two land on open boundaries and one lies beyond the 4 mm search
radius. Duplicate mesh vertices at the internal UV seam retain one shared
physical edge, so the −0.5 mm measurement on that seam remains supported.

| Branch | Expected outcome |
| --- | --- |
| Allow 2 mm deviation | Pass with all five supported measurements in tolerance |
| Inspect at 1 mm | Fail with signed defects of approximately −1.5 mm and +1.6 mm |
| Require 80% usable scan support | Fail with five of eight rows on usable surface |

The tool computes the closest point on usable triangles. Signed distances
use triangle winding and compatible incident normals at shared features.
Open mesh edges and vertices are reported separately from dimensional
defects. Use unsigned distance when winding is unknown. The support fraction counts usable scan rows. The review colors
show green for in-tolerance rows, blue for inward defects, red for outward
defects, amber for boundary, gray for unsupported rows and purple for
unreliable signs in normal mode.

Correspondence fields align with the scan's original row order. Face index
−1 and null coordinates mean no triangle was within range. State 0 is
unsupported, 1 is boundary, 2 is in tolerance, 3 is a dimensional defect,
and 4 is supported without a reliable sign. The block builds an AABB
hierarchy and limits the number of point-to-box and point-to-triangle
tests. [AABB distance query reference](https://doc.cgal.org/latest/AABB_tree/index.html)

The assets and [builder](build_example.py) are original OpenCVDataflowGUI
contributions under the repository's MIT license.
