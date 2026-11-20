# Stereo calibration fit and holdout

Open [stereo_calibration_quality.json](stereo_calibration_quality.json) and run
once. **Calibrate stereo cameras** fits a pinhole rig from ten paired target
views. **Validate stereo calibration** receives that rig and three separate
pairs of ChArUco observations. Select each block to inspect its report.

| Check | Synthetic result |
| --- | ---: |
| Joint calibration RMS | 0.089 px |
| Fit RMS by camera | 0.086 px left, 0.092 px right |
| Worst camera and fit view | Right camera, zero-based view 6 at 0.115 px |
| Held-out cross-camera RMS | 0.071 px across both cameras and all three pairs |
| Rig transform error against the generator | 0.151 mm translation, 0.014° rotation |

The fit report contains the native per-view and per-camera errors, the
point count and target-to-left-camera pose for each view. The optional
per-view RMS limit can identify a bad capture even when its error is diluted
by the other captures. View indices follow the order of the supplied arrays.

The validation block leaves the rig fixed. For each validation pair it first
estimates the target pose from the left image points and predicts the right
image points, then repeats in the opposite direction. The report identifies
both single-camera pose-fit errors and cross-camera prediction errors.
Large cross-camera errors with small pose-fit errors can indicate a
misaligned rig. Supply separate captures with corresponding target points,
matching camera grids and consistent physical units.

The graph embeds the observation arrays in its recipe. The [generator](build_example.py)
created the original synthetic points and six held-out ChArUco PNGs under
the repository [MIT license](../../../LICENSE). Tests detect the held-out
corners again from the PNGs, inject a bad right-camera training observation,
and verify that a perturbed baseline fails cross-camera validation.
[OpenCV stereo calibration reference](https://docs.opencv.org/doc/doxygen/html/d4/d93/group__calib.html)
