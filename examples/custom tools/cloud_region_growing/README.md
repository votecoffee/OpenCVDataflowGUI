# Separate touching point-cloud surfaces

Open [cloud_region_growing.json](cloud_region_growing.json) and run it. The
colored PLY interleaves 25 floor and 25 wall measurements. The two faces
share an edge and their stored normals differ by 90°. Every branch uses a
6 mm neighborhood and fits local curvature from the raw XYZ rows.

| Branch | Normal angle | Curvature limit | Result |
| --- | ---: | ---: | --- |
| Join both faces | 90° | 1/3 | One region with 50 rows |
| Separate faces | 20° | 1/3 | Two regions with 25 rows each |
| Leave the seam unassigned | 20° | 0.05 | Two regions with 20 rows each and ten unassigned seam rows |

The block grows regions through pairwise neighbor links. Normal signs are
ignored, so opposite-facing normals on the same plane still connect. Rows
with excessive curvature or an unusable local fit cannot join or bridge
a region. The curvature is the smallest local PCA eigenvalue divided by
the sum of the three eigenvalues. If the cloud has no normals, the fit
estimates them. A separate curvature input can supply one ratio per scan
row, with null marking unusable rows.

The preview colors accepted regions. The cloud outputs preserve scan
order, original colors and other row attributes. Region records give
zero-based source row indices. The labels are one-based in order of each
region's earliest source row, with 0 for unassigned rows.

The pinned Python wheel does not expose the OpenCV C++ region-growing
class. The built-in fallback uses SciPy neighborhood search and sparse
connectivity with NumPy local fits. [OpenCV C++ region-growing reference](https://docs.opencv.org/5.0/main_modules/classcv_1_1RegionGrowing3D.html)

The PLY and [builder](build_example.py) are original OpenCVDataflowGUI
assets under the repository's MIT license.
