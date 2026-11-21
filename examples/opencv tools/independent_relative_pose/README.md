# Independently Calibrated Relative Pose

Open either project and run once. Select **Estimate left-to-right pose** and inspect the left and right correspondence graphics, the report and the triangulated cloud. The match graphics identify accepted and rejected rows on their original images. Both projects use synthetic landmarks and separate camera models for 800 × 540 and 930 × 670 pixel images.

| Project | Camera pairing | Accepted pairs | Scale |
| --- | --- | --- | --- |
| [Relative pose with two pinhole cameras](relative_pose_pinhole.json) | Pinhole and pinhole, each with its own distortion and intrinsics | 84 of 96 | Unit baseline |
| [Relative pose with a fisheye camera](relative_pose_fisheye.json) | Pinhole and fisheye | 83 of 96 | Measured synthetic baseline of 0.249395 m |

Each project injects 12 incorrect matches. The fisheye project also supplies one observation outside the supported forward-ray domain. Both reject the labeled bad rows and recover the baseline direction to within 0.011° of the known direction in the fixture. Reprojection RMS stays below 0.10 px in both original images in these runs.

## Use your own camera pair

Supply equal-length point arrays with one corresponding pair per row, both camera models with distinct frame names, and optionally both source images. A calibration must match its own grid and point coordinates. Use images of the same scene with valid correspondences. Point arrays from a moving scene require aligned captures. The block accepts pinhole, fisheye and Mei omnidirectional calibration within the forward camera chart.

**Normalized threshold** is expressed in normalized ray coordinates. **Maximum reprojection** is expressed in pixels and applies independently to both source images. Points also need positive depth in both views and sufficient parallax. Accepted row numbers reference the original point arrays and appear on the cloud as the `source_index` attribute.

The pose maps left-camera coordinates to right-camera coordinates with `X_right = R @ X_left + t`. The report includes `t`'s direction in the right frame and the right camera center direction `-R.T @ t` in the left frame. Supply a positive **Baseline length** in metres or millimetres to scale the translation and triangulated cloud. With zero baseline length configured, they are expressed in units of one baseline.

## Reproduce the fixtures

The generator uses independent NumPy lens equations and saves the known motion, match labels and withheld 3D points in [truth.json](assets/truth.json). Run `python build_example.py --output <folder>` to generate the files in another directory. See the [OpenCV calibration and 3D reconstruction API](https://docs.opencv.org/5.0/main_modules/calib.html) for the native estimators.
