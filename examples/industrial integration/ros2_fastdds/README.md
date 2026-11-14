# ROS 2 Lyrical / Fast DDS

Open these projects with File → Open. Tools are in
**Industrial integration → ROS 2**. Read the [setup and contracts](../../../docs/ros2.md).

| Project | Purpose |
| --- | --- |
| [Offline image round trip](offline_image_roundtrip.json) | Real PCB photo → explicit BGR-to-RGB ROS Image encoding → decode → grayscale → edges; no ROS runtime required. |
| [Image subscriber](image_subscriber.json) | Disabled camera subscription feeding grayscale/edge tools. |
| [Triggered image publisher](triggered_image_publisher.json) | Disabled String JSON trigger supplies a part ticket, loads the real PCB photo, then publishes an image. |
| [JointState relay](joint_state_relay.json) | Disabled structured state subscription and gated publication to another demo topic, for future simulation adapters. |

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

Rebuild all four graphs without downloads or ROS activity:

```powershell
python examples/assets/build_ros2_examples.py
```

Native localhost checks cover actual Fast DDS image/data exchange and QoS/queue
behavior. The templates do not establish robot, Gazebo, TF or hand–eye acceptance.
