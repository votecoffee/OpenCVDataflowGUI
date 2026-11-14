# ROS 2 Lyrical and Fast DDS

Six blocks live in **Industrial integration → ROS 2**. The image converters work
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

## Runtime setup

The GUI uses Python 3.14/OpenCV 5. ROS runs in a **separate interpreter** configured
on this computer. Its extensions must match that interpreter. The locally verified
Lyrical runtime uses Python 3.12.3, rclpy 10.0.10 and `rmw_fastrtps_cpp`; the GUI
does not import those extension modules. No `cv_bridge` is needed.

For a Windows Lyrical prefix with its Python environment beneath `.pixi`, run
from the repository root, substituting your actual installation paths:

```powershell
python scripts/configure_ros2_runtime.py --prefix C:/path/to/lyrical --python C:/path/to/lyrical/.pixi/envs/default/python.exe
```

The script creates the ignored `.local/ros2-runtime.json`. It preserves an existing
different profile. Use `--output` to create another profile and set
`OPENCV_DATAFLOW_ROS2_PROFILE` to its absolute path before launching the app.
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

Image blocks default to **best effort / volatile / latest** receiving. Data uses
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
those fields. CameraInfo association and TF/depth calibration are future work.

Image publishing can preserve an incoming ROS header (otherwise zero), clear its
stamp, override frame ID, or stamp from the ROS node clock. With Use sim time and
no `/clock`, node time remains zero. Incoming headers remain unchanged regardless
of that setting. Header-stamp tickets are strings of frame ID + integer
seconds/nanoseconds, not UTC timestamp tickets. Reused stamps can collide across
sources or simulation resets; use an application acquisition ID when appropriate.

## Data, tickets and publishing

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
for a real openly licensed PCB converter graph and three disabled live templates.
The optional native tests use localhost, private UUID topics and domain 71:

```powershell
python -m pytest tests/test_ros2.py -q
$env:OPENCV_DATAFLOW_TEST_ROS2 = "1"
python -m pytest tests/test_ros2_native.py -q
```

These tests publish only local fixture images/state. They do not launch a
simulator or connect a robot. The [roadmap issue](https://github.com/votecoffee/OpenCVDataflowGUI/issues/20)
covers CameraInfo/depth/PointCloud2, TF2 transforms, clocks/reset and rosbag2,
services/actions, and coordinated Gazebo/ros_gz/MoveIt examples. Shared simulation
environments must be coordinated with their owners before dependencies or jobs
are changed. Native DDS loopback does not qualify simulation or robot motion.
