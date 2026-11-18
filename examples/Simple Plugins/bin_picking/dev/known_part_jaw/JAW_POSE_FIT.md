# Review current pose depth agreement

SimpleBinPicking Jaw `0.1.0.dev8` exposes a native colored **Cloud / pose_fit**
result beside the usual grasp graphics. The training popup has a **Pose fit**
layer and counts for the selected candidate's instance. These use cached evidence
from the completed pose check; changing selection, orbiting and hiding layers
does not run matching or grasp planning again.

| Color | State | Meaning at the tested model sample |
| --- | --- | --- |
| Green | Supported | All four observed neighboring depths are within the configured tolerance, with matching instance labels when supplied. |
| Magenta | Contradiction | Predicted model lies in front of every observed neighbor, beyond tolerance. |
| Blue | Occluded | Predicted model lies behind every observed neighbor, beyond tolerance. This does not measure the hidden target surface. |
| Orange | Unknown | One or more required depth rays are missing or unavailable. |
| Yellow | Ambiguous | Known neighbors do not support the sample, but do not all establish contradiction or occlusion; this includes depth discontinuities. |
| Purple | Label mismatch | Known depths have nonmatching instance labels and have not already been classified as contradiction or occlusion. |

The separate `pose_fit` output isolates the colored queries for clear review.
The combined `graphics` output and trainer's Scene layer supply measured context.
Gray points there are the measured scene. Colored pose-fit points are **derived CAD
queries**, transformed into the current bin frame. They are not additional
measurements, observed free space or physical grasp evidence. Upstream rejected
poses are not evaluated or drawn as pose fit. Valid empty instance input has no
pose-fit output.

The review retains at most 256 deterministic rows per state per instance, so
rare failures remain visible in a mostly supported surface. The displayed sample
densities are not percentages. `PointCloud.metadata.pose_fit` records counts for
**all** tested rays, display counts and the configured tolerance. Components
identify measured versus derived row ranges; `pose_fit_legend` records RGB values.

The metadata also includes signed nearest/farthest-neighbor depth residual
quantiles (5th, 50th and 95th percentiles), in meters. Residual means predicted
camera Z minus observed Z. Statistics include all known rays, including possible
occluders and discontinuities. They cannot distinguish pose error, calibration,
sensor bias or CAD differences by themselves. Missing rays never supply residuals.
This is the existing sampled first-return check, not an exact triangle renderer.

The host disables review retention when graphics are off. The report-only CLI
also skips it. Array-API callers can use `plan(..., retain_pose_fit=False)` for
the same numerical report without retaining display rows or residual quantiles.

The original report, scores, acceptance thresholds, overlap and gripper clearance
decisions are unchanged. Host graphics-off and `render_graphics=False` skip
retaining pose-fit arrays, residual summaries and both clouds. Direct API users
can likewise pass `retain_pose_fit=False` to `JawPlanner.plan`.

Use **Cloud / pose_fit** to isolate the issue at runtime. In training, hide the
cyan Model and Jaws/Sweep layers to inspect pose-fit colors, then select another
candidate or use Overview. Export cached review to preserve the chosen colored
cloud and its full-count metadata alongside the frozen recipe and report.
