# Fit a plane and sphere through point-cloud clutter

Open [cloud_model_fitting.json](cloud_model_fitting.json) and run it. Each
branch reads a millimeter PLY and fits one surface with a 0.5 mm tolerance.

| Scan | Known surface | Expected result |
| --- | --- | --- |
| Plane | (z = 600 + 0.15x - 0.20y) mm | 63 inliers and 17 outliers |
| Sphere | Center ([-33,55,430]) mm, radius 38 mm | 72 inliers and 20 outliers |

Both clouds include up to 0.1 mm of measurement variation and randomly
interleaved clutter. The sphere branch limits the admissible radius to
30–46 mm. The preview colors inliers green and outliers red. Both cloud
outputs retain their original point attributes and ordering.

The model uses the named cloud frame and reports distances in meters. A
plane uses a unit normal and metric offset, with the largest-magnitude
normal component positive. A sphere uses center XYZ and radius. Signed residuals
are aligned with source rows and are positive outside the sphere or on the
plane's positive-normal side. The report includes full-cloud support and
root-mean-square inlier error. A failed fit reports diagnostics and leaves
model outputs empty.

The block uses bounded seeded RANSAC, plane least squares or spherical
geometric refinement, then classifies every input row. The pinned Python
wheel lacks the OpenCV C++ sample-consensus class, so the shipped block uses
NumPy and SciPy. [OpenCV sample consensus reference](https://docs.opencv.org/5.0/main_modules/classcv_1_1SACSegmentation.html)

The PLYs and their [builder](build_example.py) are original
OpenCVDataflowGUI assets under the repository's MIT license.
