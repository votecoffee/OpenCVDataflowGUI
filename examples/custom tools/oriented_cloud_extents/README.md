# Measure a rotated point-cloud cuboid

Open [oriented_cloud_extents.json](oriented_cloud_extents.json) and run it.
Both branches measure the same rotated **82 × 34 × 13 mm** cuboid. The fit
reports the dimensions, the box center at **[128, −52, 480] mm**, and a
rigid pose from the box frame to the named measurement station.

- **Fit principal axes** uses OpenCV PCA. With clearly separated
  variances, it orders axes by decreasing variance and applies a
  right-handed sign convention. A compatible reference pose can select
  signs for those same axes.
- **Measure on known fixture axes** uses the connected reference
  rotation and measures the center and dimensions from the observed
  points.

To see how the tool handles symmetry, change the source block's path to
`assets/rotated_cube.ply` and run the graph. The PCA branch flags the
ambiguous orientation and withholds its pose and dimensions. Its report
and amber preview show the provisional fit. The fixture branch still
measures **36 × 36 × 36 mm** because its axes come from the fixture.

The measured extents enclose the observed samples along the chosen axes.
Sampling density affects PCA, and stray points can expand the box. The
canonical signs are a coordinate convention. Use a reference when the
application depends on a particular signed orientation.
[OpenCV PCA reference](https://docs.opencv.org/5.0/javadoc/org/opencv/core/Core.html#PCACompute2(org.opencv.core.Mat,org.opencv.core.Mat,org.opencv.core.Mat,org.opencv.core.Mat))

The point clouds and [builder](build_example.py) are original
OpenCVDataflowGUI contributions under the repository's MIT license.
