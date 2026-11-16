# ROS 2 Lyrical / Fast DDS

For demos that start their own ROS publishers on the first run and stop them when
the program/application closes, use [the self-starting image/data programs](../ros2_program_lifecycle/README.md).

Open these projects with File → Open. Tools are in
**Industrial integration → ROS 2**. Read the [setup and contracts](../../../docs/ros2.md).

| Project | Purpose |
| --- | --- |
| [Offline image round trip](offline_image_roundtrip.json) | Real PCB photo → explicit BGR-to-RGB ROS Image encoding → decode → grayscale → edges; no ROS runtime required. |
| [Image subscriber](image_subscriber.json) | Disabled camera subscription feeding grayscale/edge tools. |
| [Triggered image publisher](triggered_image_publisher.json) | Disabled String JSON trigger supplies a part ticket, loads the real PCB photo, then publishes an image. |
| [JointState relay](joint_state_relay.json) | Disabled structured state subscription and gated publication to another demo topic, for future simulation adapters. |
| [Offline cloud round trip](offline_cloud_roundtrip.json) | Real T-LESS RGB/depth and intrinsics → organized PointCloud2 conversion → typed RGB cloud and measurements; no ROS runtime needed. |
| [Cloud subscriber and raw relay](cloud_subscriber.json) | Disabled PointCloud2 source feeding existing measurements/3D rendering and an optional gated raw relay preserving all fields and bytes. |
| [Triggered cloud publisher](triggered_cloud_publisher.json) | Disabled JSON trigger tickets the paired T-LESS RGB/depth capture and publishes a meter-unit organized RGB cloud. |
| [Recorded calibrated RGB-D](offline_calibrated_rgbd.json) | Real T-LESS capture → ROS image/CameraInfo conversions → ticket coordinator → strict RGB-D validation → calibrated meter cloud. |
| [Registered RGB-D subscriber](registered_rgbd_subscriber.json) | Disabled three-topic acquisition using matching header-stamp tickets, per-acquisition CameraInfo and a bounded coordinator. |
| [Offline pose round trip](offline_pose_roundtrip.json) | Synthetic numeric fixture: m/mm conversion, parent-from-child direction, quaternion encoding and inversion without ROS. |
| [Offline run identity](offline_run_identity.json) | Real PCB acquisition with a run/reset scope, scoped ticket and current-context guard. |
| [Exact-time TF subscriber](exact_time_tf_subscriber.json) | Disabled image acquisition and persistent TF lookup at its nonzero header stamp, guarded by current run context. |
| [Offline static CameraInfo](offline_static_camera.json) | Real T-LESS depth and intrinsics → explicit static calibration binding → ticket coordination → calibrated RGB cloud, with missing ROS timestamps disclosed. |
| [Static CameraInfo subscriber](static_camera_subscriber.json) | Disabled depth subscriber with a retained-calibration reader and explicit static RGB-D validation. |
| [Offline timestamp pairing](offline_timestamp_pair.json) | Two copies of one real PCB photo with explicitly artificial timestamps demonstrate ordered FIFO pairing; reset flow state before replaying the fixed stamps. |
| [Approximate RGB-D subscriber](approximate_rgbd_subscriber.json) | Disabled ordered color/depth timestamp pairing, then static calibration binding, ticket coordination and calibrated colored cloud. |
| [Offline intensity rectification](offline_intensity_rectification.json) | Real PCB photo with explicitly synthetic camera parameters → nonidentity-R pixel remap, camera/frame outputs, valid mask and masked edges. |
| [Intensity rectification subscriber](intensity_rectification_subscriber.json) | Disabled raw intensity plus retained CameraInfo → static binding → explicit pixel/optical rectification. |
| [Offline ROI/binning rectification](offline_roi_binning_rectification.json) | Real PCB photo with synthetic crop/decimation and calibration → independently selected full rectified canvas, coverage mask and masked edges. |

Run the offline example from the repository root:

```powershell
python -m opencv_dataflow.runner "examples/industrial integration/ros2_fastdds/offline_image_roundtrip.json"
```

Live templates use domain **71** and **localhost** discovery. They remain waiting
or inactive until deliberately enabled and matched to a compatible peer. Start
the subscriber before publishing; use Continuous to keep it alive between graph
cycles. The trigger expects `std_msgs/msg/String` with `data` containing JSON such
as `{"part":"trial-1"}` on `/opencv_dataflow/demo/trigger`. Image topic is
`/opencv_dataflow/camera/image`, best effort / volatile. The JointState relay uses
reliable / volatile `/opencv_dataflow/demo/joint_states` →
`/opencv_dataflow/demo/observed_joint_states`. These are private demonstration
topics, not robot command endpoints. ROS meters/radians conventions and header
frames must be interpreted explicitly by the application.

Publishing requires Write enabled and a true Enable event. DDS acknowledgments
do not acknowledge robot work. Tickets and other Dataflow payload fields remain
local unless explicitly encoded in a data topic/custom message. A ROS header is
not a pixel-to-world calibration.

The real VisA photograph credits Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing
Zhang and Onkar Dabeer, *SPot-the-Difference*, ECCV 2022. It is licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as recorded by the
[publisher](https://registry.opendata.aws/visa/). This reuses the resized and
letterboxed PCB2 0000 image; no defects were added. See the existing
[source/hash/change chain](../../assets/showcase/SOURCES.md) and
[machine-readable hashes](../../assets/showcase/sources.json). Code is MIT; image
rights remain CC BY 4.0. Both photo graphs contain on-canvas attribution.

The cloud examples reuse **T-LESS**, Tomáš Hodaň, Pavel Haluza, Štěpán Obdržálek,
Jiří Matas, Manolis Lourakis and Xenophon Zabulis, WACV 2017, under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The [publisher](https://cmp.felk.cvut.cz/t-less/) and
[pinned source/hash/license record](<../../Simple Plugins/3d_locate/real_tless/source/ATTRIBUTION.md>)
cover the original Primesense scene 1/frame 220 RGB, depth and intrinsics.
Depth scale is 0.1 mm/sample. These graphs sample valid pixels every four rows
and columns, reconstruct the organized grid with NaN holes and retain source
pixel coordinates. They do not establish robot/world calibration or accuracy.
On-canvas comments preserve attribution. No extra dataset files are copied.

Cloud topics use `/opencv_dataflow/camera/points`, best effort / volatile.
Choose incoming units explicitly. The typed cloud is finite float32 XYZ with
optional RGB/unit normals; Message preserves arbitrary fields, exact bytes,
padding and header. Canonical encode/publish emits meters and rebuilds the
supported fields. Raw Message publish mode preserves the original packet instead;
the subscriber example relays to `/opencv_dataflow/demo/observed_points` with
Write enabled off. See the [conversion limits](../../../docs/ros2.md#pointcloud2-conversion).

The calibrated example uses known dataset frame identity `TLESS:scene1:220` for
its single recorded capture. No ROS exposure stamp is available: headers remain
zero and Allow zero stamps is explicitly on. Its report does not claim clock
synchronization. Input depth is already physical meters and travels as 32FC1.
Original RGB/depth/intrinsics, source scale and on-canvas T-LESS attribution are
preserved. This is an offline application example, not a sensor timing test.

The disabled registered subscriber expects `/opencv_dataflow/camera/color_rect`,
`/opencv_dataflow/camera/depth_registered` and `/opencv_dataflow/camera/camera_info`.
All use best effort / volatile and require the same nonzero stamp and optical
frame. CameraInfo selects rectified P with identity R and no stereo translation.
The coordinator discards incomplete tickets after two seconds. Declare the actual
depth scale before enabling. This template assumes already registered pixels;
it does not register different cameras or perform TF lookup.

The run-identity example creates a fresh unsaved session per execution graph and
scopes the image filename ticket. A rising Reset input advances generation; flow
reset/reopen creates a new session. Use Run token as the echoed `run_id` in an
external JSON envelope, then select Envelope mode on the guard. That checks the
run token without inventing acquisition tags or validating request/scene revisions.
Local tagging and guard examples perform no network I/O.

The disabled TF template queries `base_link_from_camera` at the actual incoming
stamp; Source can be overridden to `tool0` for a robot-pose lookup at image time.
Zero/latest is rejected. Default age limit is one second against the selected
node clock. Configure Use sim time/Clock topic for simulated headers and start
the persistent listener before dynamic captures. Static lookup timestamps denote
query time, not a new measurement. See [pose/TF/context contracts](../../../docs/ros2-poses-tf.md).

The static-calibration examples preserve the original CameraInfo header and bind
each image ticket separately. Calibration updates alone emit nothing; early images
are dropped until the cache is primed. Zero receipt-age limit explicitly allows
unlimited static reuse. Use Context/Require context and tagged images for run/reset
invalidation. See [static calibration contracts](../../../docs/ros2-static-camera.md).

The timestamp pairer keeps original headers and issues a common processing ticket
with original member provenance. Nearby stamps are not proof of physical capture
synchronization. Streams must increase in time; duplicates are ignored and a
backward stamp holds until reset/new run context. Pair before static calibration
binding. See [timestamp pairing contracts](../../../docs/ros2-time-pairing.md).

The intensity rectifier actually remaps pixels and returns a new camera/pixel
frame plus raw-to-rectified rotation. It does not relabel an unwarped image. The
PCB example's calibration is synthetic and its acquisition time is unknown/zero;
it makes no physical correction/accuracy claim. Typed depth, ambiguous scalar
samples without an Intensity declaration and stereo baseline projection are
rejected. Source ROI/binning and an independent rectified output grid are supported.
The ROI fixture crops/decimates the 480×320 photo to 160×120 and rectifies it to a
240×160 full-sensor canvas. This is synthetic software geometry, not a model of
hardware binning. Its CameraInfo sets `do_rectify=true` but explicitly selects
Full sensor; no tight rectified ROI is estimated. Select Custom ROI and enter
full-resolution rectified coordinates to choose another output region. See
[rectification contracts](../../../docs/ros2-rectification.md).

Rebuild all nineteen graphs without downloads or ROS activity:

```powershell
python examples/assets/build_ros2_examples.py
```

Native localhost checks cover actual Fast DDS image/cloud/data exchange and QoS/queue
behavior. The templates do not establish robot, Gazebo, TF or hand–eye acceptance.
