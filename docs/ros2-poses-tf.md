# ROS poses, exact-time TF and run identity

These optional tools live under **Industrial integration → ROS 2**. Pose
conversion and run tags work offline; TF uses the configured Lyrical worker.
See [ROS setup](ros2.md) and the [examples](<../examples/industrial integration/ros2_fastdds/README.md>).

## Pose and transform conversion

**Decode ROS 2 pose / transform** (`industrial.ros2_pose_decode`) accepts complete
TransformStamped or PoseStamped fields from the existing data reader. The app's
`pose_3d_v1.source_to_target` maps coordinates in **child/source** into
**parent/target**. ROS header.frame_id is the parent. TransformStamped names its
child; PoseStamped does not, so configure Child frame explicitly. Optional
expected frames are checked before inversion. Invert swaps frames and inverts
the matrix. Physical frame names remain intact when correlating acquisitions.

ROS translation is meters and quaternion order is **x/y/z/w**. The app pose can
use meters or millimeters. Unitless poses, nonfinite values, zero quaternions,
malformed fields and inconsistent frame names are rejected. Require unit allows
only the configured small norm tolerance; Normalize explicitly accepts finite
nonzero nonunit quaternions. Details records the original norm. Encode constructs
canonical unit quaternion fields from an existing rigid pose, optionally inverted.
Quaternion sign can change without changing the rotation. Decode's Message output
preserves the original fields for exact data relay. Zero stamps are allowed by
offline converters, but not as TF acquisition queries.

## Acquisition-time TF

**Lookup ROS 2 transform at acquisition** (`industrial.ros2_tf`) starts disabled
and cannot run in live preview. It returns the original Value, a typed Pose,
TransformStamped Message and Info. Target is the destination/parent frame; Source
defaults to the input ROS header frame or can be overridden explicitly, such as
`tool0` when asking for the robot pose at a camera timestamp.

The worker keeps a TF2 listener/cache across graph cycles. It queries the
**positive, nonzero integer acquisition stamp** and never substitutes latest TF.
TF2 interpolation at that time is allowed; extrapolation beyond available data
is unavailable. Timeout is a bounded wall-clock wait and cancellation terminates
the worker. Unavailable defaults to a validation error; Wait explicitly emits no
output event. Start the listener before collecting dynamic acquisitions: a new
listener cannot recover transforms published before it subscribed. Static TF
uses the library's transient-local listener. TF, static-TF and clock topics are
configurable for private namespaces/tests.

Max age defaults to one second against the node clock. Use sim time and the
matching Clock topic for simulated headers. Zero max age explicitly disables age
checks for historical inspection. Future tolerance is separate when age checks
are enabled. This checks acquisition age, not the publication frequency or age of
TF2's underlying interpolation samples. The result stamp is the requested query
time, including static chains; it is **not a new measurement timestamp**. Info
retains the TF2-returned header and reports that source sample times are unavailable.
See the [TF2 Buffer API](https://github.com/ros2/geometry2/blob/lyrical/tf2_ros_py/tf2_ros/buffer.py).

Observed backward node-clock jumps or clock-source changes clear the cache and
advance the worker epoch. Subsequent queries are held until flow reset or a new
validated current run context creates a new worker. Reset recreates subscriptions
and obtains static TF again. Worker ID and epoch remain diagnostic metadata; they
are not saved in projects. A run-context change also refreshes the cache. Static
frame/calibration revisions must advance the application's context explicitly;
TF2 does not supply historical versions of static calibration.

## Local run/reset context

**ROS 2 run identity** (`industrial.ros2_run`) emits:

```json
{
  "model": "ros_run_context_v1",
  "dataset_id": "inspection",
  "run_id": "operator-label-or-generated-UUID",
  "session_id": "runtime-UUID",
  "generation": 0
}
```

The Run token text output is the same opaque context scope digest used by the
scoped tickets. It can be echoed as `run_id` in external request/result JSON.
Keep the human-readable Run ID separately when needed.

Share one context authority across related branches. A new execution graph,
project reopen or flow reset creates a fresh session nonce, even if Run ID is a
fixed label. Continuous cycles retain that session. A rising true Reset input
advances generation once; holding true does not repeat resets. Display snapshots
mirror the worker context without creating another run. These live fields are
excluded from project/XML persistence.

**Attach ROS 2 acquisition identity** (`industrial.ros2_acquisition`) takes Value
and Context. Attach it before queues, parallel work or external tool adapters.
Capture mode can use the original ticket, a positive header stamp **without
frame_id**, or a dotted string/number field. This allows an image and an
exact-time robot pose to share a capture key without renaming their physical
frames. It preserves the original ticket in the acquisition record and emits a
ticket whose session is `ros-acquisition:` plus SHA-256 of the sorted, compact,
ASCII JSON context. The ticket's kind/value retain the capture identity.

The Value payload gains:

```json
{
  "acquisition": {
    "model": "ros_acquisition_v1",
    "context": {"model": "ros_run_context_v1", "dataset_id": "inspection", "run_id": "...", "session_id": "...", "generation": 0},
    "capture": {"kind": "string", "value": "17:123456789"}
  }
}
```

**Validate ROS 2 run identity** (`industrial.ros2_run_guard`) compares queued
results with the current Context. Missing, malformed or stale identities emit no
output and report a validation error. Every member of a FlowGroup is checked;
mixed generations cannot be accepted by merely tagging the outer group. Capture
keys and scoped tickets are checked for consistency. Valid child/parallel tickets
can retain the original acquisition context. TF performs the same current-context
check when its optional Context input is connected.

For an external JSON envelope, explicitly choose Guard's Envelope identity source
and the dotted Run field (default `run_id`). It compares the echoed token with
the current scope and reports `capture_checked: false`; it does not invent an
acquisition payload or correlate request IDs. The optional
[planning bridge package](../integrations/ros2/README.md) accepts this opaque token
as its run ID. Server, robot and scene revisions and request/capture correlation
remain additional checks in the adapter/application.

## Boundaries and future robot planning

Tags are local payload data. Standard ROS Image, PoseStamped and TransformStamped
messages do not carry this run context. External request/result envelopes must
echo it explicitly through JSON/custom interfaces; do not reattach the current
context to an old external result. Attach rejects an already tagged input from a
different context instead of relabeling it as current. Local tagging cannot identify
an old untagged message newly received after a restart. Header time alone does not prove
clock synchronization, calibration correctness or unique capture identity.

These checks do not authorize motion. GUI plan-only service/action integration still
needs server/robot/scene/frame revisions, reviewed results and explicit execution
contracts. The shared ROS/tool work owns those packages and adapters; this
increment neither launches a simulator nor sends robot commands.
