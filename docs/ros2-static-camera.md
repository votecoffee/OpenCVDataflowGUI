# Static CameraInfo reuse

**Cache ROS 2 CameraInfo** (`industrial.ros2_camera_cache`) applies one decoded
calibration to successive ticketed images. It is an offline-capable event block
under **Industrial integration → ROS 2**. It starts no ROS process and never
rewrites the original CameraInfo acquisition header.

The [ROS CameraInfo definition](https://github.com/ros2/common_interfaces/blob/lyrical/sensor_msgs/msg/CameraInfo.msg)
separates acquisition metadata from calibration parameters, which may remain
unchanged until recalibration. Selecting this cache is an explicit application
decision that the calibration remains applicable; arrival of an old retained
message does not establish its correctness for a new camera setup.

## Build the graph

1. Connect the data reader's CameraInfo output through **Decode ROS 2 CameraInfo**
   to the cache's **Camera** input. Match raw/rectified space to the image grid.
2. Connect decoded ROS images to **Image**. Prime calibration before sending
   images. Camera updates alone emit nothing. Images that arrive before a valid
   calibration are dropped, with a waiting message; they are not queued or replayed.
3. Use the **Image** and **Camera** outputs together. Every image event produces
   one binding, with the image's ticket on both outputs. **Info** describes it.
4. For registered RGB-D, coordinate the cache outputs and optional color stream
   with **Coordinate inputs → ticket_all**. Select **Bound static** in the RGB-D
   validator's **Calibration timing** dropdown. The default **Per acquisition**
   retains the original strict header-time checks.

Different image streams still need a shared acquisition ticket and compatible
header stamps. Static calibration reuse does not implement approximate image
pairing, cross-camera registration or clock synchronization. The cache requires
matching optical frames and original ROS dimensions, and rejects transformed
pixel coordinates that lack corresponding updated calibration.

## Reset and freshness

**Max age seconds** measures monotonic time since calibration arrived at this
cache, checked when binding each image. It does not measure exposure age or time
spent later downstream. Zero explicitly permits unlimited static reuse. Identical
new CameraInfo events refresh receipt age; image events do not. Expired calibration
is removed, and subsequent images wait until calibration is supplied again.

For run/reset protection, connect one **Run identity** authority to Context, set
**Require context**, and tag incoming images using **Attach acquisition identity**.
Prime the context, then calibration, then images. Every context change clears the
cache and rejects images tagged with an old generation. Tagged calibration must
also match that context. Untagged calibration is accepted as a newly received
static reference after priming; the cache cannot identify its historical origin.
Standard CameraInfo does not carry the application's run context.

A rising true Reset clears only the calibration; held true does not repeat the
reset. Flow reset, configuration/rewiring reset and project reopen also clear
runtime state. Stop/resume preserves it, including elapsed receipt age. Runtime
copies are private; projects/XML never contain cached samples or clock values.
The cache is serial and live preview is disabled.

Incoming calibration is validated against its original decoded CameraInfo.
Invalid, failed or previously image-bound calibration is rejected and the cache
is cleared. Managed recipe revisions must agree; a cached revision is retained
on bound outputs. Existing per-input failure gates still apply before processing.

## Binding provenance

Camera output keeps `camera_model_v1` and the unchanged `ros2` calibration record.
A separate `ros2_static_binding` object contains:

| Field | Meaning |
| --- | --- |
| `model` | `ros_static_camera_binding_v1` |
| `calibration_revision` | SHA-256 of canonical calibration/grid fields, excluding the acquisition stamp |
| `calibration_header`, `calibration_ticket`, `calibration_recipe_ref` | Original calibration provenance; ticket/recipe can be absent |
| `image_header`, `image_ticket`, `image_size` | Acquisition and grid receiving this calibration |
| `receipt_age_seconds`, `max_age_seconds` | Receipt age and configured bound at binding time |
| `context`, `context_scope` | Optional run identity and scope digest |
| `timing` | `static_calibration_applied_to_image_not_new_measurement` |

The same binding travels in the output payloads. The RGB-D validator checks its
revision, source header, image ticket/grid/frame, optional run scope and receipt
age before using **image_header** in acquisition checks. Its report exposes the
static calibration record separately. A binding indicates which calibration was
applied; it is not a new calibration measurement or cryptographic authorization.

Positive image timestamps are required by default. **Allow zero stamps** explicitly
supports recorded inputs without ROS time; negative stamps are rejected. This
option does not establish synchronization. The original calibration stamp can be
zero because it is retained as calibration provenance rather than image time.

## Examples and checks

Open the [real T-LESS example](<../examples/industrial integration/ros2_fastdds/offline_static_camera.json>)
or the [disabled retained-CameraInfo subscriber](<../examples/industrial integration/ros2_fastdds/static_camera_subscriber.json>).
The offline example discloses missing ROS timestamps and reoffers calibration
on each run. The native check publishes CameraInfo **once before the subscriber
starts**, then verifies calibrated clouds for two successive image tickets:

```powershell
python -m pytest tests/test_ros2_camera_cache.py -q
$env:OPENCV_DATAFLOW_TEST_ROS2 = "1"
python -m pytest tests/test_ros2_camera_cache_native.py -q
```

Transient-local delivery requires compatible publisher/subscriber durability and
a live publisher retaining the sample. The test uses private localhost topics and
the separate Lyrical/Fast DDS runtime; no simulator or robot is launched.
