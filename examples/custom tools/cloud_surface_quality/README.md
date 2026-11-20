# Measure local point-cloud normals and curvature

Open [cloud_surface_quality.json](cloud_surface_quality.json) and run it.
The input PLY has 321 millimeter rows in shuffled scan order. An 81-point
flat grid and a 232-point sphere have usable surface normals. The seven
collinear rows have ambiguous normals, and the isolated row has too few
neighbors.

**Measure point-cloud normals and curvature** fits the nearest rows within
22 mm for this graph. It orients the normals away from the known sphere
center. Recovered plane normals point along +Z and have zero curvature.
The sphere normals point outward and the estimated curvature is positive.
The preview uses blue for the lowest accepted curvature and yellow for the
highest. It marks rejected rows in red.

Curvature is the smallest local covariance eigenvalue divided by the sum
of the three eigenvalues. Normal quality is the gap between the two
smallest eigenvalues divided by the largest. Both are dimensionless.
Sparse and ambiguous neighborhoods have null curvature in the
source-order diagnostics. The diagnostics also report the number of rows
actually used after the radius and nearest-neighbor limits.

The accepted cloud and its `curvature` output have matching row order.
This graph wires both into region growing, which returns separate plane
and sphere regions. The `valid_indices` output maps each accepted row
back to the original scan. [Point Cloud Library normal estimation reference](https://pointclouds.org/documentation/tutorials/normal_estimation.html)

The pinned Python wheel lacks the OpenCV C++ normal estimation function.
This tool shares a bounded SciPy and NumPy fallback with region growing.
[OpenCV geometry API](https://docs.opencv.org/5.0/main_modules/segment_8hpp.html)

The PLY and [builder](build_example.py) are original OpenCVDataflowGUI
assets under the repository's MIT license.
