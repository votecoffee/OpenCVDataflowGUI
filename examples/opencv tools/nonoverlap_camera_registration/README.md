# Register cameras with disjoint views

Open `nonoverlap_camera_registration.json` to compare two pinhole cameras
and a pinhole–fisheye pair. The two cameras see separate panels of the same
rigid target. Each synchronized capture can contain different point IDs
and different numbers of points for the two cameras. The target spacing
and the known separation of the panels set the metric scale.

The two nominal branches fit eight synchronized target poses and check four
independent holdout captures. On holdout, the block solves the target pose
in one camera, transfers it through the fitted camera transform, and
checks the other camera's own points. It repeats the check in both
directions and reports the difference between the two pose estimates.
Both nominal pairs have about 0.045 px held-out RMS. The drift control
holds the fit constant and shifts one camera by 9 mm during holdout.
It intentionally fails at 8.72 px RMS, so the headless project returns a
failed inspection.

The observations use the existing `multicamera_observations_v1` format
with exactly two ordered camera frames. Use one shared ordered XYZ table
and `null` slots for unseen points. Fit and holdout capture IDs must be
distinct. A camera with missing data for an entire capture is excluded
from that pair and reported. The fixed camera records must match the
image sizes, frames and target units in the observations. Calibration of
the lens profiles and measurement of the target are separate steps.

The **Pose** output maps the first camera into the second. The essential
matrix applies to normalized undistorted rays. The fundamental matrix
is available for ideal undistorted pinhole pixels when both cameras
are pinhole. Use the block judgment to qualify a candidate pose when
a fit or holdout limit fails.
[OpenCV registration API](https://github.com/opencv/opencv/blob/5.x/modules/calib/include/opencv2/calib.hpp)

The observations are independently projected synthetic points by
OpenCVDataflowGUI contributors under the repository's MIT license.
Run `build_example.py --output <empty directory>` to regenerate them.
