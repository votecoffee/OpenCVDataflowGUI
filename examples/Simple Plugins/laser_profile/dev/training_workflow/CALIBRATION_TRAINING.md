# Laser calibration setup and review

Dev4 keeps the reference image, ROI, background, frozen stripe settings and known
camera/laser definition. **Review retained calibration…** shows declared units,
camera frame/dimensions/intrinsics/distortion, normalized plane coefficients and
the complete retained JSON. Review changes no training. The pixel-ray probe
checks the declared geometry at an explicit original pixel coordinate; it does
not detect a stripe, fit a plane or validate physical accuracy. Missing, parallel,
behind-camera, out-of-image and out-of-range samples remain unavailable.

All physical coordinates and offsets use **meters**, in the declared camera
frame: X right, Y down, Z forward. A plane is `normal dot XYZ + offset_m = 0`.
Default runtime checks use axial camera Z from 0.001 to 100 m and a ray/plane
incidence floor of 1e-6. Original pixel coordinates must match the declared camera
size and distortion model. Relabeling units or importing intrinsics does not
calibrate a camera, register a cropped image or establish a robot/workcell frame.
Without camera and laser plane, the tool explicitly remains pixel only.

## Fit supplied target observations

**Fit plane from observations…** imports a bounded JSON definition with a known
camera and explicitly supplied stripe pixels on known target planes. It runs
the existing all-point least-squares `calibrate_plane` implementation in one
background worker. It does not detect a calibration board, estimate camera
intrinsics/target poses, remove outliers or select favorable settings.

The input schema is `simple_laser_observations/v1`, with `units: "m"`, a camera
object and observations. Every observation needs a unique ID, explicit role
(`training` or `validation`), at least three original pixel pairs, and exactly
one of `target_depth_m` or `target_plane`. The depth shorthand describes a plane
perpendicular to camera Z; a general plane supplies normal, offset_m and camera
frame. Target planes must already be known in that frame.

```json
{
  "schema": "simple_laser_observations/v1",
  "units": "m",
  "source": "Analytic example, not an acquired calibration",
  "camera": {
    "matrix": [[800, 0, 320], [0, 800, 240], [0, 0, 1]],
    "image_size": [640, 480], "frame": "camera"
  },
  "observations": [
    {"identifier": "near", "role": "training", "target_depth_m": 0.4,
     "pixels": [[280, 100], [280, 240], [280, 380]]},
    {"identifier": "far", "role": "training", "target_depth_m": 0.6,
     "pixels": [[346.6666666666667, 100], [346.6666666666667, 240], [346.6666666666667, 380]]},
    {"identifier": "check", "role": "validation", "target_depth_m": 0.5,
     "pixels": [[320, 100], [320, 240], [320, 380]]}
  ]
}
```

At least two training observations must span a stable plane at distinct target
depths/poses. The fit uses every supplied training point; invalid training hits,
duplicate IDs, identical target planes and collinear geometry are rejected.
Limits are 256 total observations, 100,000 total points and 4 MiB for the input
and resulting retained metadata. Cancellation is checked between validation and
fit phases; an active bounded OpenCV/NumPy call finishes before cancellation.

Training residuals measure perpendicular distance of target-intersected points
to the fitted laser plane. The in-plane rank ratio describes geometric spread;
it is not a confidence score. Validation observations never enter fitting.
Their metric is the 3D distance between fitted-laser and declared-target
intersections along each supplied camera ray. Available/unavailable counts and
both ray-reason ledgers are retained; no valid intersections means unavailable
statistics, not zero error. Validation may be omitted, explicitly leaving no
validation evidence. IDs/roles alone do not prove independent acquisitions.

Review the fit/validation table and raw record, then choose **Use calibration
in training draft** or **Close / keep draft**. Use changes only the private
calibration draft and invalidates old learning. Learn the retained reference
again, preview an independent image with frozen settings, then Stage and Apply
in the inspector. Closing or a stale worker never activates a partial fit.
New recipes retain the observation pixels, target definitions, roles, fitted
coefficients and summary statistics. They do not retain original target images.

There is no automatic calibration acceptance threshold. Good residuals can
coexist with incorrect intrinsics/target poses; both validation intersections
use the same declared camera. Independently measured gauges and sensor bias,
repeatability and acquisition checks remain necessary for metrology acceptance.

## Reference and preview state

Persistent guidance distinguishes missing references, edited drafts, running
jobs, unavailable reference previews and ready teaching. Taught-reference
health stays separate from the last independent image preview. Editing closes
stale image/cloud/calibration snapshots and requires learning again. The window
uses available screen space, scrolls setup/native content and keeps Learn,
independent previews, Stage and Cancel accessible in a fixed footer.

Future work includes image-based target/stripe collection and association,
graphical observation editing, draft Undo/history, background file restoration,
finer cancellation and hands-on operator/physical gauge qualification.
