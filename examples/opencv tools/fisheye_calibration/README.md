# Fisheye camera calibration

Open [fisheye_calibration.json](fisheye_calibration.json) and click **Run once**.
The fitter reads eight checkerboard images. The downstream blocks validate and
rectify a ninth image that was held out of the fit. All nine images were
rendered from known intrinsics and a fisheye distortion polynomial.

| Block | What to inspect |
| --- | --- |
| **Calibrate fisheye camera** | About 0.17 px fitted RMS and 39% corner-hull coverage. The fit report lists each training image, its error and its target-to-camera pose in millimeters. |
| **Validate lens calibration** | About 0.12 px RMS on the held-out view. Debug displays detected corners and scaled reprojection residuals. |
| **Rectify pinhole / fisheye / omnidir** | Straightened checkerboard and the equivalent pinhole model on the new pixel grid. |

The 640-by-480 rectification in this example maps over 99.9% of destination
pixels to valid source coordinates. Remap coverage and the training corner
hull measure different things. The hull is the fraction of the sensor spanned
by the collected corners, including the area between those corners.
The numerical tests compare the rectification map with the known lens model.
Physical accuracy depends on target dimensions and the range of capture poses.

To use your own images, enter three to 64 checkerboard or circle-grid paths
and the target dimensions and spacing. For identified ChArUco observations,
connect **Detect ChArUco board** through **Batch serial entries** to the
fitter's **Observations** input and leave **Image paths** empty. Use varied
positions, depths and angles on one camera grid. Keep a separate target view
for validation.

A connected initial fisheye camera supplies a starting model. To lock focal
length or principal point, connect that seed and enable the corresponding
checkbox. Fixed distortion terms are set to zero. This example fixes the third
and fourth terms because the known lens has only two nonzero coefficients.
A rejected fit keeps the diagnostic RMS and report and withholds the camera
output. Connect **Save calibration profile** to save an accepted camera for reuse.

Original simulated targets and project by OpenCVDataflowGUI contributors,
[MIT](../../../LICENSE). See [camera models](../../../docs/vision-3d.md#camera-models-and-coordinate-frames)
and [OpenCV fisheye calibration](https://docs.opencv.org/5.0.0-alpha/d2/d14/namespacecv_1_1fisheye.html).
The [fixture generator](build_example.py) recreates the images and project
in an empty directory.
