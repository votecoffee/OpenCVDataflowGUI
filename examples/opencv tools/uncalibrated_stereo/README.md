# Uncalibrated stereo rectification

Open [uncalibrated_stereo.json](uncalibrated_stereo.json) and run once.
Recipe blocks supply corresponding pixels from two different-sized images.
**Fundamental matrix** estimates the epipolar geometry, and **Rectify
uncalibrated stereo images** uses its inlier mask to fit two homographies
onto one 640 × 480 pixel canvas. Select the block to see both rectified
images, their valid support masks and the fit report.

| Check | Result |
| --- | --- |
| Fitted pairs | 52 of 56 |
| Injected mismatches rejected | All four |
| Fitted vertical disparity | 23.90 px before, 0.177 px RMS after |
| Vertical disparity on 20 held-out matches | 22.23 px before, 0.261 px RMS after |
| Shared raster support | 87.55% of the output canvas |

The orange points are held-out observations. They were projected independently
into both images and excluded from matrix estimation and rectification. The
tests apply the emitted homographies to the held-out points and confirm their
positions against the warped images. The [fixture generator](build_example.py),
images and point sets are original controlled examples by OpenCVDataflowGUI
contributors under the repository [MIT license](../../../LICENSE).

For other projects, connect both images, their ordered matched points, and
a left-to-right fundamental matrix for those exact pixel grids. If lens
distortion is significant, undistort the images before collecting matches
and estimating the matrix. Each emitted homography maps source pixels into
the common canvas. The left and right outputs retain their own image
provenance. **Common valid** is the intersection of both support masks,
expressed in the left output grid. A valid pixel has the full sampling
footprint for the selected interpolation, independent of pixel intensity.
The point outputs list the fit correspondences on the rectified grids,
including any that map beyond the output crop.

This project demonstrates pixel alignment. Metric depth requires a
calibrated stereo rig and its corresponding depth reprojection geometry.
[OpenCV stereo geometry reference](https://docs.opencv.org/5.0/extra_modules/modules_2stereo_2include_2opencv2_2stereo_8hpp.html)
