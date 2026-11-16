# Saved image-motion trajectories

Four tools separate motion fitting, persistence and replay:

| Tool | Role |
| --- | --- |
| `sequence.capture_trajectory` | Bind original frames to matching stabilization transforms and explicit validity. |
| `output.motion_trajectory` | Save a validated trajectory as strict JSON. |
| `sequence.load_trajectory` | Load geometry, optionally pinned by exact file SHA256. |
| `sequence.replay_trajectory` | Apply saved corrections to the same decoded clip without feature fitting. |

Connect the original Frames group to both **Stabilize recorded sequence** and
**Capture motion trajectory**; connect the stabilizer's Transforms group to
Capture. Set Capture's interpolation to the original sampling policy. Save its
Trajectory output. A later program uses Load → Replay, with the original Frames
group on Replay's other input. Corrected/Coverage/Transforms are matching groups;
Unbatch enables ordinary per-ticket downstream inspections. Image previews one
original ordinal without discarding other frames.

## Capture contract

Each record contains 1–64 same-size/depth/encoding gray/RGB/BGR(A) frames, using
uint8, uint16, float32 or float64. Transforms are same-size
`image_stabilization_v1` records with a finite invertible affine Source to target,
its consistent inverse, source spatial/pyramid metadata and target spatial frame.
Source geometry must match; when both packets have tickets their acquisition
identities must match. Transform order is the caller's assertion when tickets
are absent. A transform does not prove its fitted accuracy merely by being valid.

Capture does not fit motion. Custom tools can provide the same transform contract
or an explicit `{"valid": false, "reason": "reference occluded"}` placeholder.
Invalid records retain a reason and their source association; they never receive
an invented identity/last-good correction. Target-root changes start new segments.
Target root sizes must remain consistent within a segment. Saved target poses
retain their image-to-root matrices; replay assigns stable trajectory-owned root
IDs rather than reviving temporary fitting-session names.

Media timing requires per-member `video_frame.frame_index` and `timestamp_ms`.
Both must strictly increase; stride and nonzero start indices are allowed.
Replay checks indices exactly and timestamps within 1e-6 ms. Indexed timing uses
the configured nominal Frame period ms, starting at zero. That mode asserts frame
order/timing; it does not measure exposure times or require current acquisition
wall-clock timestamps to equal old ones. Current ticket/timestamp metadata is
retained independently of the saved clip clock.

The versioned `image_motion_trajectory_v1` record contains:

- Direction `source_to_target`, saved interpolation and a declared clock.
- Ordered frame indices, source pixel/type/encoding SHA256 fingerprints, source
  spatial/pyramid descriptors, timestamps, segment IDs and explicit validity/reasons.
- Valid correction matrices and target grids; invalid entries have no correction.
- `clip_id`, hashing ordered pixel signatures and declared times, and
  `trajectory_id`, hashing the complete finite canonical record except that ID.

The public `capture_trajectory`, `seal_trajectory`, `validate_trajectory` and
`trajectory_bytes` helpers support Python plugins and assistant-authored workflows.
Reseal an intentional edited revision; stale IDs are rejected. These digests check
consistency, not authenticity or physical camera identity. Capture tickets and
native tracking objects are not stored.

## Replay and omissions

The complete current group must match the record before any native warping:
count/order, exact decoded pixels, depth, encoding, source coordinate/pyramid
geometry, and media timing when selected. Acquisition root labels may change;
source root sizes and index-to-root matrices must still match. A different path or
lossless container is acceptable when decoded pixels/timing match. Different
compression artifacts, preprocessing, changed frames or reordered entries reject
all output events. No registration is refitted to make a mismatch pass.

Saved interpolation is the default; an explicit nearest/linear override changes
resampling and coverage. Minimum coverage fraction gates each frame; zero coverage
always fails. Invalid borders are zeroed and Coverage identifies actual support.

| Invalid frames | Behavior for recorded invalidity or insufficient coverage |
| --- | --- |
| `reject` (default) | Failed Report and no primary frame/image events. |
| `fail_partial` | Emit valid frames, with a failed aggregate judgment and `trajectory_invalid_frames`. |
| `skip` | Pass an explicitly partial result containing only valid frames. |

All-invalid clips fail under every policy. Report lists omitted original indices,
reasons, current tickets, saved times and coverage. Skipping is not a final ticket
judgment. Preview frame names an original ordinal; an omitted preview emits no
Image instead of showing another frame. Cancellation withholds the whole result.

Outputs retain current custom data, tickets and video metadata. Their common
segment root is `trajectory:<trajectory_id>:<segment>`, with saved target geometry.
The old top-level pyramid descriptor is removed after warping; each Transform
retains the original source frame/pyramid and both mapping directions. Object
coordinates can therefore be mapped between source and corrected images.
Replay keeps no cursor, learned state or native motion cache; repeated runs,
reset, close/reopen and independent instances start from the supplied complete
clip. Lean/graphics-off preserves the same primary pixels and judgments.

## Files and bounds

Load reads at most 1 MiB of UTF-8 JSON, rejecting BOMs, duplicate keys, nonfinite
numbers and invalid schema/digests. Expected SHA256 optionally pins exact file
bytes. Relative paths use the saved project folder; Save As copies this declared
sealed asset. Save defaults to exclusive `new` publication; `replace` uses a
temporary file beside the target and atomic replacement. Existing directories are
required; live previews cannot execute the output writer. Failed replacement
cleans only its own temporary and preserves the previous file.

Records are bounded to 1–64 frames. Frame sides are 8–32,766, each image at most
2.1M pixels; corrected images plus coverage have a 128 MiB array preflight. JSON
is bounded independently. Graph snapshots, preview/history copies and native
temporaries consume additional memory. Affine coefficients are bounded to 1e6
and linear singular values to 1e-6–1e6. This is finite-group replay, without an
append-mode/unbounded stream recorder; longer sessions need explicitly bounded
clip segments. Projective or 3D motion is not part of this format.

The installed OpenCV 5.0.0/contrib 5.0.0.93 wheel has `videostab` but does not
expose the native [FromFileMotionReader](https://docs.opencv.org/5.0/extra_modules/classcv_1_1videostab_1_1FromFileMotionReader.html)
or [ToFileMotionWriter](https://docs.opencv.org/5.0/extra_modules/classcv_1_1videostab_1_1ToFileMotionWriter.html).
This validated JSON format is an application alternative, not their native file
format. Replay uses the existing OpenCV affine warp/support implementation.
No general speedup, stabilization accuracy or physical calibration is asserted.

The [ten original examples](<../examples/opencv tools/motion_trajectory/README.md>)
use independently known integer corrections, exact PNG/FFV1 data, uint16, media
timestamps, explicit invalidity and deliberate mismatch cases. A separate native
fit/capture graph verifies replay pixel equality against current stabilization.
