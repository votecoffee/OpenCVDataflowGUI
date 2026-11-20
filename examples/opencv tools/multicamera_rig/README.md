# Multi-camera rig calibration

Open [multicamera_rig.json](multicamera_rig.json) and run once. Camera B
bridges two sets of pinhole observations. Camera A sees the first 10
training captures and camera C sees the last 10. B sees all 20. The
reported initialization links run A to B and B to C. The fitted camera
poses use A as the origin and meters for translation.

The fit reports about **0.028 px RMS**. The validation block uses six
separate target poses and reports about **0.035 px RMS** over 12
cross-camera predictions. For each prediction it estimates the target pose
from only one camera and scores it in a different camera using the fitted
rig. Each debug heatmap uses cameras as rows and captures as columns.
Black cells have no scored view. Both profiles retain the numeric reports.

## Use another set of observations

Load a JSON record through **Load rig observations** or wire an equivalent
data record to **Calibrate multi-camera rig**. The asset files show the
`multicamera_observations_v1` format.

- `units` is `m` or `mm`. `object_points` is the shared ordered list of
  target XYZ points. Use a target on the XY plane when fitting intrinsics.
- `camera_frames` lists the camera names, and `image_sizes` lists their
  corresponding `[width, height]` values. The first camera defines the
  rig frame.
- `capture_ids` identifies synchronized observations of each target pose.
  `visibility` and `image_points` are camera-by-capture tables. Use 1 and
  the full ordered target point list for a detected view. Use 0 and
  `null` for a missing view.
- For fixed intrinsics, supply an ordered list of existing camera model
  records with matching frames, image sizes, lens families and units.
  Fit and validation captures need distinct IDs.

The saved project qualifies a synthetic pinhole chain. Focused tests also
cover fisheye, fixed intrinsics, millimeter inputs, and a shifted camera
pose that fails the holdout. This workflow starts from detected,
corresponding target points and masks whole views.

The assets are original synthetic measurements by the OpenCVDataflowGUI
contributors (MIT). Rebuild in a fresh directory with
`python build_example.py --output <directory>`.

[OpenCV 5 multi-view calibration API](https://docs.opencv.org/5.0/extra_modules/calib_8hpp.html)
