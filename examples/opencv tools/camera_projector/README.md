# Camera and projector calibration and reconstruction

Open [camera_projector.json](camera_projector.json) and click **Run once**.
The graph fits a camera–projector pair from fourteen projected target views,
then triangulates an independent two-depth scene. The camera grid is 152 × 116
and the projector grid is 256 × 192. Both have lens distortion.

The fit reports **0.011 px RMS** and a **138.6 mm** baseline. The stepped
scene recovers **16,789** valid points on the 1.10 m and 1.33 m planes,
with maximum camera-Z error below **0.19 mm**. It rejects all 72 injected
bad matches and preserves the shadow and unsupported pixels as invalid depth.

## Fit a rig

Supply two pinhole intrinsic models with distinct frame names and the same
metric units. The projector model uses its own resolution. Keep calibrated
intrinsics fixed or refine the projector intrinsics from the supplied
projector observations before estimating the camera-to-projector transform.
The camera intrinsics stay fixed in either mode.

Use 3 to 32 varied target views with 6 to 256 corresponding rows each.
Connect or load a JSON object with `object_views`, `camera_views`, and
`projector_views`. Supply target XYZ in the declared metric units and
XY pixel observations in the same point order for each view. Only keep
jointly observed target points. The report shows per-device and overall
reprojection RMS. A rejected fit withholds the rig. Use **Save calibration
profile** and **Load calibration profile** in single-file mode to reuse a rig.

## Reconstruct

Connect the decoded phase X and Y images, or the three sparse
outputs from **Decode Gray-code correspondences**. The dense maps use
float projector pixels on the camera grid. For recorded dense files
you can mark unavailable values with NaN or use an out-of-domain value
such as -1. The two phase sequences must show the same stationary scene.
The example TIFFs are predecoded synthetic correspondences.

The block undistorts camera and projector observations and triangulates in
the camera frame. It rejects weak ray angles, depth outside the configured
range, and excessive reprojection error at either device. A mask or region
limits samples on the camera grid. Depth and cloud units are meters.
Invalid depth is NaN. The cloud retains the source camera pixels.

The [fixture builder](build_example.py) recreates the original synthetic
assets under [MIT](../../../LICENSE).
[OpenCV stereo calibration and triangulation](https://docs.opencv.org/5.0/javadoc/org/opencv/stereo/Stereo.html).
