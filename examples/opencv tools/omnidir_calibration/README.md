# Omnidirectional camera calibration

Open [omnidir_calibration.json](omnidir_calibration.json) and click **Run once**.
The graph fits the Mei camera model from ten original rendered checkerboards.
It validates a separate target near the left edge and rectifies that view into
a perspective image.

| Block | Expected result |
| --- | --- |
| **Calibrate omnidirectional camera** | About 0.26 px training RMS, 57% corner-hull coverage and fitted xi of about 0.768. The renderer used xi of 0.75. |
| **Validate lens calibration** | About 0.19 px RMS on 54 held-out corners. The leftmost measured corner is about 38 px from the edge of the 640 px image. |
| **Rectify pinhole / fisheye / omnidir** | Perspective view and a pinhole profile tied to its new coordinate grid. |

The example fixes the four distortion terms because the synthetic camera has
zero radial and tangential distortion. The fitter can estimate all four terms,
both focal lengths, the principal point and xi. It fixes skew at zero, matching
the application's camera model contract. Its initialization comes from OpenCV
and any locked nonzero intrinsic begins at that native estimate.

Supply three to 64 grid images or batch identified ChArUco observations on one
pixel grid. Capture widely separated positions, distances and tilts. OpenCV may
exclude views during initialization. The fit report identifies used and omitted
views in the input order. **Require all** withholds the camera on any exclusion.
**Allow subset** applies the minimum view count and the usual error and corner
coverage limits to the retained views. Both keep the fit report.

For validation, the app estimates a fresh target pose from forward camera rays
and evaluates error back in the original distorted pixels. Use a separate
target view and the same physical spacing units as the fit. The example and
its map comparison qualify this synthetic forward-facing setup.

Original images and project by OpenCVDataflowGUI contributors under
[MIT](../../../LICENSE). The [builder](build_example.py) recreates the graph and
assets in an empty directory. See the [native OpenCV API](https://docs.opencv.org/4.x/d3/ddc/group__ccalib.html)
and the [camera workflow](../../../docs/vision-3d.md#camera-models-and-coordinate-frames).
