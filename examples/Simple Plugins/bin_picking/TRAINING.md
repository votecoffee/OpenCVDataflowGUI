# Train and review SimpleBinPicking

Requires the optional `opencv-simple-bin-picking` package and a host with
`PluginDialogContext.snapshot_connected_inputs`. The initial tool proposes
suction contacts from calibrated first-return depth. It does not require a
part matcher and does not command a robot.

1. Save the project. Connect a rectified axial-Z depth image and a
   `camera_model_v1` calibration, then run the sources. The new picker itself
   can remain untrained. Depth is uint16, float32 or float64, at the retained
   camera dimensions. Zero/negative/nonfinite samples stay unknown.
2. Open **Train bin picking…** and set **Depth units per sample** before
   **Capture connected reference**, unless the camera JSON supplies
   `depth_scale`. Calibration declares mm or m. Optional image and integer
   instance-map connections are retained together with the reference. An
   imported `.sbp` recipe restores all of these directly. Raw PNG/TIF/NPY depth
   plus calibration JSON can also initialize training.
3. In **Reference / bin / floor**, orbit, zoom and pan the native cloud. Enable
   **Collect floor picks**, pick at least three spread-out observed floor
   points and choose **Fit floor / align +Z**. Check the fit and direction.
   Draw/resize the axis-aligned bin interior, including its floor and walls;
   the top is open. An automatic box is only an initial proposal. A connected
   `bin_pose` explicitly owns the bin frame; edit that upstream calibration
   rather than silently redefining it in the picker.
4. Enter the actual cup radius, body envelope, body length, TCP offset,
   pregrasp travel and clearance. Lengths use the declared physical units.
   Surface tolerance, tilt, minimum height and contact fraction are inspection
   requirements. Candidate/pick limits and 1–8 workers bound execution.
5. **Preview reference** runs the current draft in the background. Inspect
   the contact/clearance table and **Cached candidate review**. Green marks
   eligible alternatives; red is rejected, orange is unknown, yellow is the
   selected candidate. RGB axes show the TCP; +Z retracts from the surface,
   approach motion is -Z. Blue swept cylinders show configured tool geometry;
   magenta/orange witnesses explain observed collisions/unknown space.
   The overview limits graphic clutter; every candidate remains in the table
   and raw report. Select a row to review its own cup and envelope.
6. Run another source image, then **Preview completed run** to freeze the
   completed picker inputs for comparison with the current trained recipe.
   Selection and layer changes reuse frozen results. **Export review…** writes
   RGB PLY, raw depth, bin transform, recipe, report, selection/layers and
   available capture identity to a new directory.
7. **Stage recipe** creates a new sealed project asset and updates the
   inspector draft. Inspector **Apply** activates it. Cancel preserves active
   parameters. Edits invalidate Stage until a current reference preview
   finishes. Undo restores committed draft edits; stale or late background
   results cannot replace newer training.

Capture reads one private host publication without executing the target.
Publication/execution identities are descriptive, not evidence that separate
cameras exposed simultaneously. Complete upstream acquisition/calibration
qualification separately. The archive retains reference arrays, intrinsics,
transform and settings; arbitrary acquisition metadata is not embedded in the
version-1 archive. Available capture identities accompany exported reviews.

The depth image determines surface geometry. Reference RGB supplies retained
appearance, not a trained texture model. A reference instance map applies to
reference preview only; each execution needs its own aligned map if instance
constraints are desired. Instance 0 is excluded and positive IDs constrain
contact to one instance. Ranked alternatives can lie on the same object:
reobserve after every physical pick.

No eligible pick is a valid result. Do not turn unknown clearance into free
space or relax physical tolerances merely to get a candidate. The initial
real T-LESS examples retain zero eligible picks. Independent suction labels,
actual bin scenes, seal tests, full robot geometry/IK and observed pick
outcomes remain qualification work.
