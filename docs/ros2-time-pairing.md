# Ordered image timestamp pairing

**Pair ROS 2 images by timestamp** (`industrial.ros2_image_pair`) buffers two
ticketed image streams and associates nearby acquisition times. It runs offline
under **Industrial integration → ROS 2** and starts no ROS process. Strict shared
workpiece tickets remain preferable when a PLC or hardware trigger supplies them.

ROS [Image headers](https://github.com/ros2/common_interfaces/blob/lyrical/sensor_msgs/msg/Image.msg)
describe acquisition time and optical frame. A common clock label is an
application assertion: this block neither synchronizes clocks nor measures
their offset, drift or exposure accuracy. It never substitutes receipt time,
averages acquisition stamps or changes image coordinates.

## Matching rule

Connect the streams to Input 1 and Input 2. Each stream must have strictly
increasing positive integer header stamps. The matcher compares FIFO heads:

1. If their absolute difference is at most **Tolerance ms**, emit that pair and
   consume both images. Zero tolerance requires identical nanoseconds.
2. Otherwise discard the older head: with ordered streams, a later arrival on
   the other stream cannot make that old head a valid match.
3. Continue until either queue is empty. No image is reused for another pair.

This is the earliest feasible ordered pair. It does not wait for a potentially
closer future image, solve a global assignment problem or implement the ROS
`message_filters` matching algorithm. Per-stream order is required; cross-stream
arrival order can differ. Equal stamps on one stream are ignored as duplicates.
Backward stamps clear queued images and latch a failed **Reset required** state,
because reordered delivery and a clock rollback cannot be distinguished here.
Nothing is emitted while held. Use Reset flow state, a rising Reset input or a
new current run context to resume. General out-of-order/watermark matching is
future work.

**Require same frame** defaults on. Turning it off permits stereo or other
distinct optical frames but performs no registration or pose transformation.
The default tolerance is 5 ms; select it for the actual speed, motion and timing
requirements rather than treating the default as an accuracy guarantee.

## Queues, deadlines and diagnostics

**Max wait ms** is a monotonic receipt deadline per image. Images expire on
subsequent input events or idle graph polls, including inside ToolBlocks. Zero
disables expiry. Time stopped counts toward the deadline. A single Run once
returns waiting without maintaining a background timer; no hard real-time
deadline is implied while an upstream source blocks the graph thread.
Run All can also finish when finite sources are exhausted; unmatched queues do
not keep it alive. Resume graph polling or reset flow state to handle that remainder.

**Max pending** bounds the total number of queued images across both streams.
**Max bytes MB** bounds accounted packet bytes. Exceeding either reports a
validation error and clears buffers; these counters are not a process-heap cap.
Dropped, expired, duplicate and backward-time images produce diagnostic/activity
markers and intermediate ticket events, with no downstream image event. Ignored
duplicates do not assign a new pass/fail judgment. Normal input failure gates
apply before buffering; accepted failed images retain their failure status.

State survives GUI Stop/resume and layout moves. Flow reset, configuration changes,
rewiring and project reopen clear it. Runtime copies have private image buffers.
Live preview is disabled, and the block executes serially. Queues and live clock
values are excluded from JSON/XML persistence.

## Outputs and identities

**Group**, **Input 1**, **Input 2** and **Info** emit together for each match.
Group labels are `input_1` and `input_2`. A new common processing ticket identifies
the temporal pair; the two original tickets and headers stay in the
`ros_time_pair_v1` report and output `ros_time_pair` payload. Pixels, format,
coordinates and each original acquisition stamp remain unchanged. A final
judgment downstream belongs to this new pair ticket; original source tickets
remain separate histories.

Info records the ordered policy, exact `spread_ns`, configured `tolerance_ns`,
declared clock domain and each member's original header/ticket/acquisition and
receipt age. `clock_domain_verified` and `physical_sync_verified` remain false.
Close times alone do not prove that images show the same physical part.
Managed recipe revisions must agree. Group processing time retains the earliest
available input start; each image keeps its own timing provenance.

For run/reset protection, prime **Context** from one Run identity authority and
tag both streams with Attach acquisition identity before queuing. **Require
context** enforces this. Tagged images are also rejected without a connected,
primed authority: the block waits and discards them, allowing a later Context
event to prime it without replaying early images. A changed context flushes
queues and checks incoming tags;
paired outputs carry that context and a new pair capture key. Original capture
keys remain in Info. A local Reset alone does not advance the external run
generation; advance Run identity when invalidating old external work.

## Registered RGB-D with static calibration

Pair color and depth **before** Cache ROS 2 CameraInfo. Feed the paired depth to
the cache, then coordinate paired color, cached depth and bound camera with
**ticket_all**. The cache uses the new pair ticket and keeps the depth's original
stamp in its binding. Select **Bound static** on the RGB-D validator and set
**Max skew ms** at least as wide as the intended pairing tolerance. The validator
still checks ticket, frame, size, calibration, units and pixel-grid compatibility.
Pairing an already statically bound image is rejected because reticketing would
invalidate its earlier binding.

Open the [disabled RGB-D template](<../examples/industrial integration/ros2_fastdds/approximate_rgbd_subscriber.json>)
or [offline control fixture](<../examples/industrial integration/ros2_fastdds/offline_timestamp_pair.json>).
The offline fixture uses two copies of one real VisA PCB photograph with explicitly
artificial timestamps. It demonstrates software policy, not real capture timing;
reset flow state before replaying its fixed stamps. The native test sends real
local DDS messages, drops a pair outside tolerance, then reconstructs a calibrated
colored cloud from a later valid pair and one static calibration:

```powershell
python -m pytest tests/test_ros2_pairing.py -q
$env:OPENCV_DATAFLOW_TEST_ROS2 = "1"
python -m pytest tests/test_ros2_pairing_native.py -q
```

No simulator or robot is launched. General multi-stream assignment, clock
alignment, late-event watermarks and measured camera exposure synchronization
remain separate milestones.
