# ROS 2 Lyrical and Fast DDS

Twenty-two blocks live in **Industrial integration → ROS 2**. The converters work
offline. Read/publish blocks are optional and start disabled; opening, validating
or previewing a graph never starts ROS. There is no `pip install rclpy` dependency
in the GUI environment.

| Block | Interface |
| --- | --- |
| Read ROS 2 image (`input.ros2_image`) | `sensor_msgs/msg/Image` → image, message details, trigger and acquisition ticket |
| Publish ROS 2 image (`industrial.ros2_image_publish`) | Image plus true Enable → publication and delivery details |
| Read ROS 2 data (`input.ros2_data`) | Installed ROS message → structured data, message details, trigger and ticket |
| Publish ROS 2 data (`industrial.ros2_data_publish`) | Structured fields or String JSON plus true Enable → delivery details |
| Encode ROS 2 image (`industrial.ros2_image_encode`) | Image → portable in-memory ROSImage packet and bounded details |
| Decode ROS 2 image (`industrial.ros2_image_decode`) | Portable ROSImage packet → image, preserving local payload metadata |
| Read ROS 2 point cloud (`input.ros2_cloud`) | PointCloud2 → typed cloud, original raw message, details, trigger and ticket |
| Publish ROS 2 point cloud (`industrial.ros2_cloud_publish`) | Typed cloud or raw portable message plus true Enable → PointCloud2 publication |
| Encode ROS 2 point cloud (`industrial.ros2_cloud_encode`) | Typed cloud → canonical portable PointCloud2 packet, offline |
| Decode ROS 2 point cloud (`industrial.ros2_cloud_decode`) | Portable packet → finite XYZ/RGB/normals cloud with explicit units and pixel organization |
| Decode ROS 2 CameraInfo (`industrial.ros2_camera_decode`) | CameraInfo fields → existing camera model, original message and geometry details |
| Encode ROS 2 CameraInfo (`industrial.ros2_camera_encode`) | Camera model → canonical current-grid CameraInfo fields for the existing data publisher |
| Validate registered ROS RGB-D (`industrial.ros2_rgbd`) | Completed ticketed FlowGroup → checked color/depth/camera, physical meter depth and association report |
| Cache ROS 2 CameraInfo (`industrial.ros2_camera_cache`) | Reuse decoded static calibration with explicit per-image ticket/grid/revision bindings and optional run/reset guards |
| Pair ROS 2 images (`industrial.ros2_image_pair`) | Bounded ordered FIFO timestamp pairing with original-header provenance, common tickets and optional run/reset guards |
| Rectify ROS 2 intensity (`industrial.ros2_rectify_image`) | Pinhole/fisheye remapping with nonidentity R, raw ROI/binning, explicit rectified output grids, new camera/header/rotation, valid mask and reusable maps |
| Decode / Encode ROS 2 pose (`industrial.ros2_pose_decode`, `industrial.ros2_pose_encode`) | Strict TransformStamped/PoseStamped ↔ typed m/mm parent-from-child poses |
| Lookup ROS 2 transform (`industrial.ros2_tf`) | Persistent exact-acquisition-time TF2, bounded freshness/wait/cancellation, optional current-context guard |
| ROS 2 run identity (`industrial.ros2_run`) | Unsaved runtime session and rising-edge reset generation |
| Attach acquisition identity (`industrial.ros2_acquisition`) | Scope a ticketed value by dataset/run/session/generation and capture key |
| Validate run identity (`industrial.ros2_run_guard`) | Reject stale or untagged values/groups before consuming results |

See [pose/TF and run-context contracts](ros2-poses-tf.md) for frame direction,
clock reset behavior, context schemas and external-envelope boundaries.

## Runtime setup

[Self-starting image/data demos](<../examples/industrial integration/ros2_program_lifecycle/README.md>)
use the program's first-run managed service to launch a local publisher through
this runtime profile. Repeated Run once calls reuse it; program replacement and
application close destroy its ROS nodes and process tree. Configure **Program →
Settings and scripts…** for hooks, readiness, shutdown and status. See
[program lifecycle](program-lifecycle.md). The existing disabled endpoint templates
remain available for independently managed peers.

The GUI uses Python 3.14/OpenCV 5. ROS runs in a **separate interpreter** configured
on this computer. Its extensions must match that interpreter. The locally verified
Lyrical runtime uses Python 3.12.3, rclpy 10.0.10 and `rmw_fastrtps_cpp`; the GUI
does not import those extension modules. No `cv_bridge` is needed.

For the Windows Lyrical workspace under `G:\dev`, run from the repository
root. Set the prefix to your Lyrical installation directory.

```powershell
$lyrical = Join-Path 'G:\dev' 'lyrical'
python scripts/configure_ros2_runtime.py --prefix $lyrical --python (Join-Path $lyrical '.pixi/envs/default/python.exe')
```

The script creates the ignored `.local/ros2-runtime.json`. The source checkout
and a virtual environment directly inside it can use this local profile.
When moving a runtime, use `--output` to create a profile for the new prefix.
Set `OPENCV_DATAFLOW_ROS2_PROFILE` to its absolute path before launching the
app, or back up the active profile and use the newly generated one in its place.
Profiles are trusted local configuration; saved projects cannot select a worker
script or executable. They hold `python`, `ros_distro`, and explicit
`path_prepend`, `python_paths`, `ament_prefixes` directories. For other layouts,
provide these paths for the installed runtime and any trusted interface overlays.
Alternatively, launch from a sourced Lyrical environment and set
`OPENCV_DATAFLOW_ROS2_PYTHON` to its absolute Python executable. That override
changes the executable only; choose a matching profile/environment as well.

The packaged worker has a fixed entry point and length-prefixed local pipes.
It verifies Lyrical and the actual middleware identifier. The host sets
`RMW_IMPLEMENTATION=rmw_fastrtps_cpp` and the selected publication mode, domain and
discovery range. Fast DDS is provided by the ROS installation; see its
[RMW configuration](https://github.com/ros2/rmw_fastrtps/blob/lyrical/README.md).
Nothing installs into or upgrades the ROS environment.

Each active I/O block owns one private ROS node/worker. Subscriptions keep
receiving between **Continuous** graph cycles. Reset, graph close, disabled reads,
configuration changes and transport failures close the worker. Startup and RPC
waits are bounded and cancellable; a hung worker is terminated. A new manual run
can create fresh subscriptions, so use Continuous for a sustained stream.

## Topics, queues and QoS

Start with an isolated test topic, **domain 71**, **localhost** discovery.
Use the same domain, resolved topic and installed interface at both endpoints.
Relative topics resolve under Namespace. The initial name validator supports
ordinary slash-separated identifiers; remapping/substitution syntax is deferred.
Inherited Fast DDS XML/security configuration can also influence transports;
localhost discovery is not a network security boundary.

Image/cloud blocks default to **best effort / volatile / latest** receiving. Data uses
**reliable / volatile / FIFO**. Match the peer's offered/requested QoS. A reliable
subscriber cannot receive from a best-effort publisher. Depth, durability,
deadline and lifespan are exposed. Zero deadline/lifespan uses the middleware's
infinite default; this release does not export individual QoS event callbacks.
Transient local can replay cached samples while their publisher stays alive.

The subscription queue is bounded by Depth and Queue bytes. Overflow drops the
oldest entries; Latest also discards superseded entries. Message details report
receive sequence, received/dropped counts, queued bytes, local age and matched
publishers. Max age is measured from local receipt, independently of ROS clocks.
An empty or stale queue emits no output, trigger or ticket.

Timeout seconds bounds each poll or publication discovery/acknowledgment wait.
One graph activation consumes at most one sample. The executor does not schedule
a graph cycle on every DDS callback; choose Continuous cadence and queue settings
for your acquisition rate. Each endpoint uses a worker, so profile process and
serialization overhead before deploying a large number of endpoints.

## Images and payloads

Supported encodings: mono8/mono16, RGB/BGR/RGBA/BGRA at 8 or 16 bits, and scalar
8UC1/16UC1/16SC1/32SC1/32FC1/64FC1. Conversion checks dimensions, channel count,
depth, stride, endian and exact buffer size, including padded rows. Auto preserves
sample type and tagged color order. Explicit RGB↔BGR conversion swaps color
channels; depth scaling and HSV/Lab/Bayer/YUV conversion require explicit blocks.
Images are limited to **16 MiB** of pixel storage; data messages to **1 MiB** of
strict finite JSON, with bounded local protocol frames and buffers.

ROS Image header, encoding and endpoint details travel in the image's `ros2`
payload. A received image gets its own Dataflow pixel frame. A ROS `frame_id`
does not establish a calibrated pixel-to-world transform. The
[Image definition](https://github.com/ros2/common_interfaces/blob/lyrical/sensor_msgs/msg/Image.msg)
defines acquisition header, encoding and row stride. A 16UC1 or 32FC1 image has
**no implicit physical depth unit** in these blocks.

Offline encode/decode retains arbitrary local JSON payloads, spatial/pyramid
provenance and tickets. Across DDS, only standard ROS Image fields are sent;
Dataflow tickets, judgments, graphics and calibration metadata are not embedded
in pixel bytes. Use a correlated data topic or a custom installed message for
those fields. CameraInfo association is described below; TF lookup and geometric
depth registration between different camera frames are separate operations.

Image publishing can preserve an incoming ROS header (otherwise zero), clear its
stamp, override frame ID, or stamp from the ROS node clock. With Use sim time and
no `/clock`, node time remains zero. Incoming headers remain unchanged regardless
of that setting. Header-stamp tickets are strings of frame ID + integer
seconds/nanoseconds, not UTC timestamp tickets. Reused stamps can collide across
sources or simulation resets; use an application acquisition ID when appropriate.

## PointCloud2 conversion

The cloud blocks use the same bounded worker and enable/ticket/QoS contracts.
Native messages are kept as portable `ROSPointCloud` packets, retaining the exact
header, fields, record and row strides, endian, density flag and binary data.
This supports organized sensors and unordered clouds. Raw packets are limited to
16 MiB/2 million points, 64 fields and 1024 components per field. Invalid offsets,
duplicate names, unknown datatypes, truncated buffers and out-of-record fields are
rejected before publication. Overlapping field aliases are allowed.
The schema follows the official
[PointCloud2](https://github.com/ros2/common_interfaces/blob/lyrical/sensor_msgs/msg/PointCloud2.msg)
and [Lyrical PointField](https://github.com/ros2/common_interfaces/blob/lyrical/sensor_msgs/msg/PointField.msg)
definitions, including int64/uint64/bool and vector fields.

The **Cloud** output projects scalar float32/float64 XYZ into the app's existing
finite **float32 PointCloud**. Choose the incoming units explicitly (m or mm);
decoding labels the units without scaling them. Header frame_id names the cloud
frame; Empty frame is used only for an empty header. Neither establishes a TF
transform or calibration. Float64 XYZ can lose precision in this projection;
overflow is invalid. Drop or Reject controls invalid XYZ/normals. A message
claiming Is dense while containing invalid XYZ is rejected.

Auto colors supports packed uint32/float32 RGB/RGBA or separate uint8 r/g/b.
Packed float bits are decoded as bits even when they represent IEEE NaN. Alpha
remains in the raw packet. Complete normal_x/y/z or nx/ny/nz triples can be
normalized; zero/nonfinite normals follow the invalid-point policy. Ignore colors
or normals when those fields are not needed. Details reports discarded points,
ignored fields and float32 storage. For organized input, retained points keep
their source pixel coordinates and source image size, including after dropping
invalid entries. Image masks still need the corresponding pixel frame.

The **Message** output on Read retains extras such as intensity, IDs and
descriptors, even though the typed cloud does not contain them. Custom tools can
use `ros2_clouds.cloud_records(packet)` to inspect a structured NumPy view with
the original field types and component counts. Receipts describe these packets
without serializing point data. Arbitrary local JSON payloads survive offline
converters; standard ROS messages carry only their defined fields.

Encoding/publishing constructs canonical XYZ float32, optional unit normals and
packed uint32 RGB. It converts mm to **meters**, rejects unitless clouds and uses
`cloud.frame` for header frame_id. Preserve keeps an incoming acquisition stamp
(otherwise zero); Zero clears it; publishers can instead stamp the node clock.
Processing followed by encoding rebuilds these supported fields: it does not
remerge unknown fields, alpha or original padding automatically. Publisher
**Input mode → Message** sends the raw packet, retaining every field, byte and
stride; the selected stamp mode still applies. Only wire the selected Cloud or
Message input. Raw mode does not rescale units or reorganize storage, so the
caller must know its units. Use it to relay unchanged sensor data without losing
extras. This is distinct from rebuilding an inspected typed cloud.

Unorganized output packs the retained points into one row and drops pixel
correspondence. Source pixels reconstructs the original image grid, filling
missing positions with NaN and marking Is dense false. It requires valid unique
pixel correspondence; duplicate pixels and grids exceeding the buffer limit are
rejected. Little/big endian output is selectable. Both forms reuse the existing
3D thumbnail, popout, filtering, measurement and rendering tools.

## CameraInfo and registered RGB-D

Use the existing data reader/publisher with `sensor_msgs/msg/CameraInfo`; no
additional ROS process type or dependency is needed. Decode CameraInfo selects
**Raw** K/D or **Rectified** P intrinsics for images already in that space. It
does not warp images. Supported models are plumb_bob, rational_polynomial and
equidistant, plus an explicitly ideal empty model. Empty D on a recognized model
is treated as zero distortion; uncalibrated K is rejected. The original complete
message remains on Message and in the camera's `ros2.camera_info` record.
See the [ROS CameraInfo definition](https://github.com/ros2/common_interfaces/blob/lyrical/sensor_msgs/msg/CameraInfo.msg).

Raw ROI offsets and binning adjust focal lengths, principal point and output
dimensions. Zero binning means one; all-zero ROI means the full sensor. Details
includes the output-to-full pixel mapping and discarded partial edge bins.
Rectified selection currently requires identity R, zero P translation and no
requested ROI recomputation. These checks prevent silently losing stereo
baseline or changing the optical frame. Use explicit stereo/TF/rectification
workflows for those cases. Encode creates current-grid K/D, identity R, P=K|0,
and zero ROI/binning; use Decode's original Message for an exact field relay.

For raw intensity pixels, [explicit CameraInfo rectification](ros2-rectification.md)
supports nonidentity R and raw ROI/binning on monocular grids. An output-grid
dropdown selects the CameraInfo ROI, full sensor canvas or custom rectified ROI.
When `do_rectify` is true, explicitly choose Full sensor or Custom ROI; automatic
tight ROI estimation remains unavailable. It warps pixels and
returns a named virtual optical frame, raw-to-rectified rotation and canonical
current-grid CameraInfo. It rejects physical depth and
stereo P translation. The converter's conservative Rectified mode stays unchanged.

For live registered RGB-D, connect color Image, depth Image and decoded Camera
to inputs 1/2/3 of **Coordinate inputs**, using **ticket_all**. Configure all three
ROS readers for Header stamp tickets and a common optical frame/acquisition
stamp. Set a deadline for incomplete acquisitions. The existing coordinator owns
queues, limits, timeouts and late suppression; the RGB-D validator owns no queue.
Per-acquisition calibration tickets remain the default. For startup-only or
retained calibration, use the [static CameraInfo cache](ros2-static-camera.md)
and explicitly select Bound static in the validator. The original calibration
header stays unchanged. For ordered streams with nearby stamps, use the explicit
[timestamp pairer](ros2-time-pairing.md) before static calibration binding. It
requires a declared common clock and does not establish physical synchronization.

The validator requires all member tickets to match, named matching optical
frames, matching original ROS image sizes and camera dimensions, unchanged pixel
geometry, and undistorted pinhole intrinsics. Header spread is compared using
integer nanoseconds; Max skew permits an explicit tolerance after the caller has
already correlated the tickets. It does not make different header-stamp tickets
match in the coordinator. Different camera frames need real registration/TF
before this block. Matching fields do not prove physical pixel registration.

Uint16 depth requires explicit meters per sample; floating depth declares meters
or millimeters. Output is a typed physical-meter DepthArray. Zero, negative,
nonfinite and overflowing values become NaN. An all-invalid depth image is a
valid failed inspection with a reason; geometry/frame/ticket/time mismatches are
validation errors with no output event. Color and depth outputs share the
declared registered pixel frame and retain member metadata. Feed Color, Depth
and Camera to **Depth to point cloud** using its existing typed inputs.

Zero stamps are rejected by default. Allow zero stamps is an explicit mode for
known recorded/offline pairs; Info reports `timestamps_nonzero: false`.
`clock_domain_verified` remains false because Header does not identify a clock
domain or prove clock synchronization. Equal stamps, `use_sim_time`, and matching
tickets alone cannot establish exposure synchronization. The examples use actual
dataset pairing and disclose missing ROS timestamps instead of inventing them.

## Structured data and publishing

Fields mode maps JSON objects to the chosen ROS interface. Nested objects/arrays
are supported and unspecified ROS fields retain interface defaults. Conversion
checks reject silent integer truncation, type changes or out-of-range values;
ordinary float32 rounding is allowed. NumPy arrays/scalars on the ROS side are
normalized before crossing the pipe. JSON string mode is restricted to
`std_msgs/msg/String` and serializes/parses its `data` field as strict JSON.

Dropdowns include scalar messages, Float64MultiArray, PoseStamped, PoseArray,
Twist, TransformStamped, JointState, CameraInfo, Odometry and Clock. Custom
`package/msg/Type` interfaces must already be installed in the trusted ROS
environment. These are data exchanges: no automatic robot-unit, frame, TF,
quaternion or controller-command conversion is implied.

Readers issue tickets after successful decoding. Choose normal source ticket
settings, header stamp, or an external string/number in a dotted data field,
such as `part.id`. Wire Ticket into another source in External ticket mode to
coordinate image acquisition. Header stamps alone do not prove synchronization.

Writers require **Write enabled** and a true **Enable** input for every event.
Require subscriber defaults on; no compatible subscriber returns waiting without
an output event or attempted publication. Reliable QoS can wait for DDS
acknowledgments. Best effort reports local publication without acknowledgment.
**DDS acknowledgment is not application acceptance or robot completion.** Add an
explicit correlated response topic for that workflow.

A publish timeout, disconnect or cancellation after dispatch can have an unknown
outcome. The block holds further publications until flow state is reset; inspect
the consumer before reset. It never automatically retries an ambiguous write.
Pre-publication field validation failures remain known rejections.

## Examples, verification and next milestones

Open the [ROS examples](<../examples/industrial integration/ros2_fastdds/README.md>)
for real openly licensed PCB/RGB-D graphs, offline pose/run-context workflows
and ten disabled live templates.
The optional native tests use localhost, private UUID topics and domain 71:

```powershell
python -m pytest tests/test_ros2.py -q
python -m pytest tests/test_ros2_clouds.py -q
python -m pytest tests/test_ros2_calibration.py -q
python -m pytest tests/test_ros2_camera_cache.py -q
python -m pytest tests/test_ros2_pairing.py -q
python -m pytest tests/test_ros2_rectification.py tests/test_ros2_rectification_roi.py -q
python -m pytest tests/test_ros2_poses_context.py tests/test_ros2_tf.py -q
$env:OPENCV_DATAFLOW_TEST_ROS2 = "1"
python -m pytest tests/test_ros2_native.py -q
python -m pytest tests/test_ros2_cloud_native.py -q
python -m pytest tests/test_ros2_calibration_native.py -q
python -m pytest tests/test_ros2_camera_cache_native.py -q
python -m pytest tests/test_ros2_pairing_native.py -q
python -m pytest tests/test_ros2_tf_native.py -q
```

These tests publish only local fixture images/state. They do not launch a
simulator or connect a robot. The [roadmap issue](https://github.com/votecoffee/OpenCVDataflowGUI/issues/20)
covers general out-of-order/multi-stream timing, extended rectification and
cloud field/remerge work, clock/run-envelope integration and rosbag2, GUI
services/actions, and coordinated Gazebo/ros_gz/MoveIt examples. Pose/TF and local
run/reset context are implemented; standard messages still need explicit external
identity envelopes. Shared simulation
environments must be coordinated with their owners before dependencies or jobs
are changed. Native DDS loopback does not qualify simulation or robot motion.
