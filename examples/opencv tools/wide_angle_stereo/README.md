# Wide-angle stereo calibration

Open [fisheye.json](fisheye.json) or [omnidir.json](omnidir.json) and click
**Run once**. Each graph fits a tilted stereo rig with a baseline of about
120.08 mm from fourteen projected board poses. It then rectifies a separate
pair of images of a textured plane at 3 m, computes disparity and reconstructs
a point cloud.

Both fixtures use original numerical observations. The plane images were
rendered independently of the calibration solver. The geometry checks compare
held-out correspondences over the perspective grid and verify vertical
disparity below 0.005 px and Q reprojection within 3 mm per axis. Dense SGBM
yields a median depth of about 3.07 m on both rendered scenes.

## Prepare your own inputs

Use **Calibrate wide-angle stereo** with two camera records of the same lens
family, resolution and units. Connect the matched **Object views**,
**Left views**, and **Right views** lists, or select a JSON file with
`object_views`, `left_views` and `right_views` keys. Each list contains the
views in the same order. Each view has corresponding XYZ or distorted XY rows
in common target-point order. Use 3 to 32 varied stereo pairs and 6 to 256
points per pair. Both native stereo optimizers require the same count in
every view. Omnidirectional targets lie on the board's XY plane.

For fisheye, supply calibrated intrinsics for fixed mode or starting values
for refinement. The omnidirectional solver initializes both eyes internally.
It uses the input camera records for frame names, units and dimensions.
The fitter reports used and omitted view indices, a baseline, and errors per
eye and view. **Require all** withholds the rig on exclusions. **Allow subset**
accepts a retained fit that passes your minimum count and error limits.

**Rectify stereo images** uses native fisheye balance and FOV controls for
fisheye pairs. For omnidirectional pairs, it uses OpenCV's stereo rotations
with a shared perspective focal length and a metric Q derived from the
rotated baseline. Tune the focal length to trade perspective coverage against
sampling density. The qualified omnidirectional output is a perspective
grid for ordinary horizontal disparity.

The [fixture builder](build_example.py) recreates both graphs and assets.
Original fixtures by OpenCVDataflowGUI contributors under
[MIT](../../../LICENSE). [OpenCV fisheye stereo API](https://docs.opencv.org/5.0/javadoc/org/opencv/stereo/Stereo.html),
[OpenCV omnidirectional calibration](https://docs.opencv.org/5.0/tutorials_contrib/ccalib/omnidir_tutorial.html).
