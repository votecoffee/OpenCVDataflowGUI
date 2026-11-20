# Merge registered scans in one fixture frame

Open [cloud_scan_merge.json](cloud_scan_merge.json) and run it. Three
synthetic scans carry different point colors. The green side scan supplies
positions in meters and a 90° camera-to-fixture pose with millimeter
translation. Both branches produce millimeter coordinates in the
`fixture` frame.

| Branch | Expected result |
| --- | --- |
| Append all scan rows | 11 points with their original RGB values and rotated normals. |
| Suppress cross-scan overlap | 8 points. The 0.5 mm radius removes two side-view points near the blue reference and one gold point near a retained green point. |

**Prefer earlier** compares each later scan to the retained points from
earlier inputs in port or group order. A close pair from the same scan
survives, and the surviving rows retain their measured positions. Use
**Append** to keep every observation.

Connect up to four individual clouds and supply a pose for each cloud
that needs a frame change. Individual scans with tickets must share the
same ticket. For a group of up to 64 already registered clouds, connect
the group output of **Join parallel branches** or **Batch serial entries**
to **Scans**. Group order determines overlap priority and member tickets
remain available in the per-scan records.

The **Provenance** output gives each scan's half-open output row span,
original row indices, and available source pixels, image grid and ticket.
The merged cloud has one named 3D frame. Source pixel grids belong to
their individual scans. **Require matching** validates RGB and normal
availability. **Common** emits an attribute when all retained scans
provide it. The block uses bounded nearest-neighbor queries for overlap.
[SciPy nearest-neighbor reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query.html)

The three PLY assets and [builder](build_example.py) are original
OpenCVDataflowGUI contributions under the repository's MIT license.
