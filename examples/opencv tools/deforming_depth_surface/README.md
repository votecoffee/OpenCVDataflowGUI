# Track a deforming depth surface

Open the [example project](deforming_depth_surface.json) and select **Run All**.
The recording shows a textured sheet shearing sideways and bending toward a
fixed calibrated camera. Select the tracker to view the cloud, status and
reference-grid displacement preview.

| Pair index | Expected result |
| --- | --- |
| 0 | Captures the reference and waits for a second usable exposure. |
| 1 | Tracks 685 of 759 reference samples. |
| 2 | Tracks 614 samples and leaves the depth hole empty. |
| 3 | Reports tracking lost and withholds the cloud because depth is missing. |
| 4 | Recovers 609 samples using the original reference. |

The three reconstructed frames have **0.31, 0.37 and 0.55 mm** RMS XYZ error
against the fixture's analytic geometry. The fixture builder contains the
reference geometry and exact image deformation. The automated tests also
evaluate camera-Z error on the reference material grid.

Connect depth to **Image**, the corresponding registered image to **Guide**,
and the calibrated fixed-view pinhole model to **Camera**. Use **Fixed Pixel**
for depth-only changes along fixed camera rays. The cloud uses live camera
coordinates in meters. Its Source Index identifies each original material
pixel within the reference grid. Surface Depth and Displacement retain that
reference grid, with NaN at unsampled or rejected pixels. Send **Reset** to
replace the reference with the next valid input.

The example qualifies visible-surface reconstruction from a single fixed view.
The installed OpenCV wheel rejects the native DynaFu constructor, so this
workflow uses OpenCV optical flow and calibrated depth lifting on the CPU.
The assets are original synthetic fixtures by OpenCVDataflowGUI contributors
under the project's MIT license. Run the builder with an empty output
directory to regenerate them.
[OpenCV DynaFu implementation](https://github.com/opencv/opencv_contrib/blob/5.0.0/modules/rgbd/src/dynafu.cpp)
