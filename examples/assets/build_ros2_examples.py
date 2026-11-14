"""Rebuild public ROS examples offline; live endpoints remain disabled."""

from pathlib import Path

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

ROOT = Path(__file__).resolve().parents[1] / "industrial integration" / "ros2_fastdds"
REGISTRY = builtin_registry()
CREDIT = (
    "VisA PCB photo: Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang and Onkar Dabeer, "
    "SPot-the-Difference (ECCV 2022). CC BY 4.0: https://registry.opendata.aws/visa/. "
    "Resized/letterboxed from bundled PCB2 0000; no defects added. "
    "Source and hashes: ../../assets/showcase/SOURCES.md. No robot/controller operation."
)


def add(graph, kind, name, position, **parameters):
    block = graph.add(REGISTRY.create(kind, block_id=name, parameters=parameters))
    block.position = list(position)
    return block


def note(graph, text):
    block = add(graph, "layout.comment", "source_and_scope", (0, -300))
    block.name = "Source, setup and scope"
    block.description = text
    block.node_size = "large"


def photo(graph):
    return add(
        graph,
        "input.file",
        "pcb_photo",
        (0, 0),
        path="../../assets/showcase/visa-pcb2-0000.png",
        path_base="project",
    )


def build():
    ROOT.mkdir(parents=True, exist_ok=True)
    graph = Graph()
    image = photo(graph)
    encoder = add(
        graph,
        "industrial.ros2_image_encode",
        "encode_rgb",
        (320, 0),
        encoding="rgb8",
        frame_id="demo_camera_optical",
    )
    decoder = add(graph, "industrial.ros2_image_decode", "decode_rgb", (640, 0))
    gray = add(graph, "cv.gray", "gray", (960, 0))
    edges = add(graph, "cv.canny", "edges", (1280, 0))
    graph.connect(image.id, "image", encoder.id, "image")
    graph.connect(encoder.id, "message", decoder.id, "message")
    graph.connect(decoder.id, "image", gray.id, "image")
    graph.connect(gray.id, "image", edges.id, "image")
    note(
        graph,
        CREDIT + " Offline converter round trip: no ROS installation is required. "
        "RGB conversion is explicit; depth and header are preserved. This is not a DDS test.",
    )
    graph.save(ROOT / "offline_image_roundtrip.json")

    graph = Graph()
    image = add(graph, "input.ros2_image", "camera", (0, 0))
    gray = add(graph, "cv.gray", "gray", (320, 0))
    edges = add(graph, "cv.canny", "edges", (640, 0))
    graph.connect(image.id, "image", gray.id, "image")
    graph.connect(gray.id, "image", edges.id, "image")
    note(
        graph,
        "Live camera template: disabled. Configure local ROS runtime (docs/ros2.md), "
        "domain 71, localhost, /opencv_dataflow/camera/image, best-effort/volatile QoS. "
        "Enable camera and use Continuous after a peer is ready. One acquired sample per cycle. "
        "No ROS image or robot connection is embedded in this project.",
    )
    graph.save(ROOT / "image_subscriber.json")

    graph = Graph()
    trigger = add(
        graph,
        "input.ros2_data",
        "trigger",
        (0, 0),
        topic="/opencv_dataflow/demo/trigger",
        data_mode="json_string",
        ticket_mode="field",
        ticket_field="part",
    )
    image = photo(graph)
    image.position = [320, 0]
    image.ticket_settings = {"mode": "external", "value": 1}
    writer = add(
        graph,
        "industrial.ros2_image_publish",
        "publish_camera",
        (640, 0),
        frame_id="demo_camera_optical",
    )
    graph.connect(trigger.id, "ticket", image.id, "ticket")
    graph.connect(image.id, "image", writer.id, "image")
    graph.connect(trigger.id, "trigger", writer.id, "enable")
    note(
        graph,
        CREDIT + ' Disabled live photo publisher, triggered by String JSON {"part":"A-1"}. '
        "Enable the trigger and Write enabled only on an isolated test domain "
        "with a camera subscriber. Uses best-effort images and reliable JSON triggers. "
        "Ticket remains local; ROS header frame "
        "does not encode the ticket. Trigger tickets must distinguish acquisitions.",
    )
    graph.save(ROOT / "triggered_image_publisher.json")

    graph = Graph()
    reader = add(
        graph,
        "input.ros2_data",
        "joint_state",
        (0, 0),
        topic="/opencv_dataflow/demo/joint_states",
        message_type="sensor_msgs/msg/JointState",
    )
    writer = add(
        graph,
        "industrial.ros2_data_publish",
        "relay_state",
        (320, 0),
        topic="/opencv_dataflow/demo/observed_joint_states",
        message_type="sensor_msgs/msg/JointState",
    )
    graph.connect(reader.id, "data", writer.id, "data")
    graph.connect(reader.id, "trigger", writer.id, "enable")
    note(
        graph,
        "Disabled simulation-data template: JointState fields relay to a separate private demo "
        "topic; not a command topic. ROS uses meters/radians/seconds, and JointState "
        "ordering comes from name[]. Use matching domain/QoS. Data stays structured "
        "with header/name/position/velocity/effort. "
        "No joint controller, simulator, TF calibration or trajectory execution is provided. "
        "Both Enabled and Write enabled must be turned on deliberately.",
    )
    graph.save(ROOT / "joint_state_relay.json")

    build_cloud_examples()
    build_calibration_examples()
    build_pose_context_examples()
    build_static_camera_examples()
    build_timestamp_pair_examples()


def real_depth_graph():
    source = Graph.load(ROOT.parents[1] / "demos/rgbd_cloud/rgbd_cloud.json", REGISTRY)
    graph = Graph()
    for key in ("rgb", "depth", "camera", "project", "showcase_source"):
        graph.add(source.blocks[key])
    for edge in source.connections:
        if edge.source in graph.blocks and edge.target in graph.blocks:
            graph.connect(edge.source, edge.output, edge.target, edge.input)
    graph.blocks["showcase_source"].description += (
        " This ROS codec example reconstructs organized pixels with NaN holes. "
        "It performs no ROS or robot operation, and does not establish TF calibration. "
        "Source/hash/license: ../../Simple Plugins/3d_locate/real_tless/source/ATTRIBUTION.md."
    )
    return graph


def build_cloud_examples():
    graph = real_depth_graph()
    encoded = add(
        graph,
        "industrial.ros2_cloud_encode",
        "encode_cloud",
        (660, 0),
        organization="source_pixels",
    )
    decoded = add(graph, "industrial.ros2_cloud_decode", "decode_cloud", (990, 0))
    measure = add(graph, "3d.cloud_measure", "measure_cloud", (1320, 0))
    graph.connect("project", "cloud", encoded.id, "cloud")
    graph.connect(encoded.id, "message", decoded.id, "message")
    graph.connect(decoded.id, "cloud", measure.id, "cloud")
    graph.save(ROOT / "offline_cloud_roundtrip.json")

    graph = Graph()
    reader = add(graph, "input.ros2_cloud", "cloud_source", (0, 0))
    measure = add(graph, "3d.cloud_measure", "measure_cloud", (330, 0))
    view = add(graph, "3d.cloud_render", "cloud_view", (660, 0))
    graph.connect(reader.id, "cloud", measure.id, "cloud")
    graph.connect(reader.id, "cloud", view.id, "cloud")
    relay = add(
        graph,
        "industrial.ros2_cloud_publish",
        "raw_relay",
        (330, 340),
        input_mode="message",
        topic="/opencv_dataflow/demo/observed_points",
    )
    graph.connect(reader.id, "message", relay.id, "message")
    graph.connect(reader.id, "trigger", relay.id, "enable")
    note(
        graph,
        "Disabled PointCloud2 subscriber on /opencv_dataflow/camera/points; "
        "domain 71, localhost, best-effort/volatile. Declare incoming XYZ units explicitly. "
        "Cloud output projects finite XYZ/RGB/normals; the Message output retains all fields "
        "and bytes, including intensity/IDs/descriptors. Header frame_id is not TF calibration. "
        "Optional raw relay is also disabled; Message mode preserves unknown fields, padding "
        "and exact bytes on /opencv_dataflow/demo/observed_points. "
        "No simulator, robot or native visualization service is launched.",
    )
    graph.save(ROOT / "cloud_subscriber.json")

    graph = real_depth_graph()
    trigger = add(
        graph,
        "input.ros2_data",
        "trigger",
        (-330, 0),
        topic="/opencv_dataflow/demo/trigger",
        data_mode="json_string",
        ticket_mode="field",
        ticket_field="part",
    )
    for key in ("rgb", "depth"):
        graph.blocks[key].ticket_settings = {"mode": "external", "value": 1}
        graph.connect(trigger.id, "ticket", key, "ticket")
    writer = add(
        graph,
        "industrial.ros2_cloud_publish",
        "publish_cloud",
        (660, 0),
        organization="source_pixels",
    )
    graph.connect("project", "cloud", writer.id, "cloud")
    graph.connect(trigger.id, "trigger", writer.id, "enable")
    graph.blocks["showcase_source"].description += (
        " Disabled live publisher. Enable trigger and Write enabled only for a configured "
        "isolated test domain and compatible subscriber. Publishes XYZ in meters, "
        "with original pixel organization; Dataflow tickets remain local."
    )
    graph.save(ROOT / "triggered_cloud_publisher.json")


def build_calibration_examples():
    graph = real_depth_graph()
    graph.remove("project")
    for key in ("rgb", "depth"):
        graph.blocks[key].ticket_settings = {"mode": "fixed", "value": "TLESS:scene1:220"}
    rgb_encode = add(
        graph, "industrial.ros2_image_encode", "rgb_encode", (330, 0), frame_id="camera"
    )
    rgb_decode = add(graph, "industrial.ros2_image_decode", "rgb_decode", (660, 0))
    depth_encode = add(
        graph, "industrial.ros2_image_encode", "depth_encode", (330, 300), frame_id="camera"
    )
    depth_decode = add(graph, "industrial.ros2_image_decode", "depth_decode", (660, 300))
    cam_encode = add(graph, "industrial.ros2_camera_encode", "camera_encode", (330, 600))
    cam_decode = add(graph, "industrial.ros2_camera_decode", "camera_decode", (660, 600))
    cam_ticket = add(graph, "flow.ticket", "camera_ticket", (990, 600), mode="external")
    graph.connect("rgb", "image", rgb_encode.id, "image")
    graph.connect(rgb_encode.id, "message", rgb_decode.id, "message")
    graph.connect("depth", "depth", depth_encode.id, "image")
    graph.connect(depth_encode.id, "message", depth_decode.id, "message")
    graph.connect("camera", "camera", cam_encode.id, "camera")
    graph.connect(cam_encode.id, "message", cam_decode.id, "message")
    graph.connect(cam_decode.id, "camera", cam_ticket.id, "value")
    graph.connect("rgb", "ticket", cam_ticket.id, "ticket")
    coord = add(graph, "flow.coordinate", "coordinate", (1320, 250), inputs=3, max_wait_ms=5000.0)
    for block, output, port in (
        (rgb_decode, "image", "input_1"),
        (depth_decode, "image", "input_2"),
        (cam_ticket, "value", "input_3"),
    ):
        graph.connect(block.id, output, coord.id, port)
    checked = add(graph, "industrial.ros2_rgbd", "checked", (1650, 250), allow_zero_stamps=True)
    project = add(graph, "3d.depth_cloud", "project", (1980, 250), sample_step=4)
    graph.connect(coord.id, "group", checked.id, "group")
    for output, port in (("color", "image"), ("depth", "depth"), ("camera", "camera")):
        graph.connect(checked.id, output, project.id, port)
    graph.blocks["showcase_source"].description += (
        " Recorded single-frame CameraInfo/RGB-D association. Dataset frame identity supplies "
        "the fixed ticket; headers are zero because no ROS exposure timestamp is supplied. "
        "Allow zero stamps is explicit and does not prove acquisition clock synchronization. "
        "Original metric depth is passed as 32FC1 meters; "
        "no synthetic acquisition time is invented."
    )
    graph.save(ROOT / "offline_calibrated_rgbd.json")

    graph = Graph()
    rgb = add(
        graph,
        "input.ros2_image",
        "rgb",
        (0, 0),
        topic="/opencv_dataflow/camera/color_rect",
        ticket_mode="header_stamp",
    )
    depth = add(
        graph,
        "input.ros2_image",
        "depth",
        (0, 300),
        topic="/opencv_dataflow/camera/depth_registered",
        ticket_mode="header_stamp",
    )
    source = add(
        graph,
        "input.ros2_data",
        "camera_info",
        (0, 600),
        topic="/opencv_dataflow/camera/camera_info",
        message_type="sensor_msgs/msg/CameraInfo",
        ticket_mode="header_stamp",
        reliability="best_effort",
    )
    camera = add(graph, "industrial.ros2_camera_decode", "camera", (330, 600), space="rectified")
    coord = add(graph, "flow.coordinate", "coordinate", (660, 250), inputs=3, max_wait_ms=2000.0)
    checked = add(graph, "industrial.ros2_rgbd", "checked", (990, 250))
    project = add(graph, "3d.depth_cloud", "project", (1320, 250), sample_step=4)
    graph.connect(source.id, "data", camera.id, "message")
    for block, output, port in (
        (rgb, "image", "input_1"),
        (depth, "image", "input_2"),
        (camera, "camera", "input_3"),
    ):
        graph.connect(block.id, output, coord.id, port)
    graph.connect(coord.id, "group", checked.id, "group")
    for output, port in (("color", "image"), ("depth", "depth"), ("camera", "camera")):
        graph.connect(checked.id, output, project.id, port)
    note(
        graph,
        "Disabled registered RGB-D acquisition template. All three topics must provide "
        "the same nonzero acquisition stamp and optical frame, with already registered color/depth "
        "pixels. CameraInfo uses rectified P intrinsics, identity R, "
        "no stereo baseline translation. "
        "Choose the exact depth scale; uint16 defaults to 0.001 m/sample, floats to meters. "
        "Coordinator queues matching tickets for two seconds, then discards incomplete sets. "
        "Header equality does not prove clock synchronization or registration. "
        "No TF lookup, simulator or robot motion is performed.",
    )
    graph.save(ROOT / "registered_rgbd_subscriber.json")


def build_pose_context_examples():
    graph = Graph()
    pose = add(
        graph,
        "3d.pose",
        "pose",
        (0, 0),
        source_frame="tool0",
        target_frame="base_link",
        translation=[100, -200, 300],
        rvec=[0.0, 0.0, 1.5707963267948966],
        units="mm",
    )
    encoded = add(graph, "industrial.ros2_pose_encode", "encode", (330, 0))
    decoded = add(graph, "industrial.ros2_pose_decode", "decode", (660, 0), units="mm")
    inverse = add(graph, "industrial.ros2_pose_decode", "inverse", (660, 300), invert=True)
    graph.connect(pose.id, "transform", encoded.id, "pose")
    graph.connect(encoded.id, "message", decoded.id, "message")
    graph.connect(encoded.id, "message", inverse.id, "message")
    note(
        graph,
        "Original synthetic pose fixture: tool0 into base_link, XYZ in millimeters. "
        "ROS translation is meters, quaternion order xyzw, transform direction parent_from_child. "
        "Inverse swaps source/target frames and inverts the matrix. Zero header stamp is retained "
        "for this offline numeric example; zero is never accepted by exact-time TF lookup. "
        "No ROS runtime, robot, simulator or motion is used.",
    )
    graph.save(ROOT / "offline_pose_roundtrip.json")

    graph = Graph()
    image = photo(graph)
    run = add(
        graph,
        "industrial.ros2_run",
        "run",
        (0, 300),
        dataset_id="VisA-PCB",
        run_id="inspection-demo",
    )
    attach = add(graph, "industrial.ros2_acquisition", "attach", (330, 0))
    guard = add(graph, "industrial.ros2_run_guard", "guard", (660, 0))
    gray = add(graph, "cv.gray", "gray", (990, 0))
    graph.connect(image.id, "image", attach.id, "value")
    graph.connect(run.id, "context", attach.id, "context")
    graph.connect(attach.id, "value", guard.id, "value")
    graph.connect(run.id, "context", guard.id, "context")
    graph.connect(guard.id, "value", gray.id, "image")
    note(
        graph,
        CREDIT + " Local run identity example. Attach scope before queues/branches and guard "
        "results against the current context afterward. New execution graph, project reopen or "
        "flow reset rotates the unsaved session nonce; a rising Reset input advances generation. "
        "Run/acquisition tags are local payloads, not standard ROS header fields "
        "or motion permission.",
    )
    graph.save(ROOT / "offline_run_identity.json")

    graph = Graph()
    image = add(graph, "input.ros2_image", "camera", (0, 0), ticket_mode="header_stamp")
    run = add(graph, "industrial.ros2_run", "run", (0, 300))
    attach = add(
        graph, "industrial.ros2_acquisition", "attach", (330, 0), capture_mode="header_stamp"
    )
    lookup = add(graph, "industrial.ros2_tf", "lookup", (660, 0), target_frame="base_link")
    encode = add(graph, "industrial.ros2_pose_encode", "encode", (990, 0))
    graph.connect(image.id, "image", attach.id, "value")
    graph.connect(run.id, "context", attach.id, "context")
    graph.connect(attach.id, "value", lookup.id, "value")
    graph.connect(run.id, "context", lookup.id, "context")
    graph.connect(lookup.id, "pose", encode.id, "pose")
    note(
        graph,
        "Disabled exact-time TF template. Enable only an isolated configured ROS domain. "
        "The camera header supplies nonzero acquisition time/source frame; Target is base_link. "
        "Start the persistent TF listener before dynamic captures. No latest fallback exists. "
        "Default node-clock freshness is one second; select Use sim time and the correct clock "
        "for simulated headers. Observed backward clock jumps invalidate the listener until reset "
        "or a new validated run context. Static lookup stamps are query times, not measurements. "
        "Pose is base_link_from_camera in meters. Nothing publishes robot motion.",
    )
    graph.save(ROOT / "exact_time_tf_subscriber.json")


def build_static_camera_examples():
    graph = real_depth_graph()
    graph.remove("project")
    encoded = add(
        graph, "industrial.ros2_image_encode", "encode_depth", (330, 0), frame_id="camera"
    )
    decoded = add(graph, "industrial.ros2_image_decode", "decode_depth", (660, 0))
    cam_encode = add(graph, "industrial.ros2_camera_encode", "encode_camera", (330, 400))
    cam_decode = add(graph, "industrial.ros2_camera_decode", "decode_camera", (660, 400))
    cache = add(graph, "industrial.ros2_camera_cache", "cache", (990, 150), allow_zero_stamps=True)
    coordinate = add(graph, "flow.coordinate", "coordinate", (1320, 150))
    checked = add(
        graph,
        "industrial.ros2_rgbd",
        "checked",
        (1650, 150),
        use_color=False,
        depth_label="input_1",
        camera_label="input_2",
        calibration_timing="bound_static",
        allow_zero_stamps=True,
    )
    project = add(graph, "3d.depth_cloud", "project", (1980, 150), sample_step=4)
    graph.connect("depth", "depth", encoded.id, "image")
    graph.connect(encoded.id, "message", decoded.id, "message")
    graph.connect("camera", "camera", cam_encode.id, "camera")
    graph.connect(cam_encode.id, "message", cam_decode.id, "message")
    # Supporting arrival first, then one binding per image event.
    graph.connect(cam_decode.id, "camera", cache.id, "camera")
    graph.connect(decoded.id, "image", cache.id, "image")
    graph.connect(cache.id, "image", coordinate.id, "input_1")
    graph.connect(cache.id, "camera", coordinate.id, "input_2")
    graph.connect(coordinate.id, "group", checked.id, "group")
    graph.connect(checked.id, "depth", project.id, "depth")
    graph.connect(checked.id, "camera", project.id, "camera")
    graph.connect("rgb", "image", project.id, "image")
    graph.blocks["showcase_source"].description += (
        " Static CameraInfo reuse with an explicit per-image binding. The recorded T-LESS "
        "depth and original intrinsics share the known dataset grid. No ROS exposure stamp "
        "exists, so zero stamps are explicitly allowed without claiming clock synchronization. "
        "The calibration header remains unchanged; binding records the image ticket, grid "
        "and calibration revision. Static updates alone emit no images. This graph reoffers "
        "its offline calibration on each run with a new processing ticket for the same recorded "
        "image; the native test covers one retained update "
        "followed by multiple image tickets. No ROS runtime is needed here."
    )
    graph.save(ROOT / "offline_static_camera.json")

    graph = Graph()
    source = add(
        graph,
        "input.ros2_data",
        "camera_info",
        (0, 350),
        topic="/opencv_dataflow/camera/camera_info",
        message_type="sensor_msgs/msg/CameraInfo",
        durability="transient_local",
    )
    camera = add(graph, "industrial.ros2_camera_decode", "camera", (330, 350), space="rectified")
    depth = add(
        graph,
        "input.ros2_image",
        "depth",
        (0, 0),
        topic="/opencv_dataflow/camera/depth_registered",
        ticket_mode="header_stamp",
    )
    cache = add(graph, "industrial.ros2_camera_cache", "cache", (660, 150))
    coordinate = add(graph, "flow.coordinate", "coordinate", (990, 150))
    checked = add(
        graph,
        "industrial.ros2_rgbd",
        "checked",
        (1320, 150),
        use_color=False,
        depth_label="input_1",
        camera_label="input_2",
        calibration_timing="bound_static",
    )
    project = add(graph, "3d.depth_cloud", "project", (1650, 150), sample_step=4)
    graph.connect(source.id, "data", camera.id, "message")
    graph.connect(camera.id, "camera", cache.id, "camera")
    graph.connect(depth.id, "image", cache.id, "image")
    graph.connect(cache.id, "image", coordinate.id, "input_1")
    graph.connect(cache.id, "camera", coordinate.id, "input_2")
    graph.connect(coordinate.id, "group", checked.id, "group")
    graph.connect(checked.id, "depth", project.id, "depth")
    graph.connect(checked.id, "camera", project.id, "camera")
    note(
        graph,
        "Disabled static CameraInfo/depth template. A live transient-local CameraInfo publisher "
        "can replay its retained calibration to this late subscriber. Prime the cache before "
        "images; early images are discarded, not replayed. CameraInfo is reused explicitly, "
        "without changing its original header; image acquisition stamps must be nonzero. "
        "Default receipt-age limit zero allows unlimited static reuse. Set a finite limit "
        "when updates are expected. Share Run identity and tagged images with Require context "
        "for run/reset invalidation. No robot, simulator or cross-frame registration is started.",
    )
    graph.save(ROOT / "static_camera_subscriber.json")


def build_timestamp_pair_examples():
    graph = Graph()
    for index in (1, 2):
        source = add(
            graph,
            "input.file",
            f"photo_{index}",
            (0, (index - 1) * 350),
            path="../../assets/showcase/visa-pcb2-0000.png",
            path_base="project",
        )
        tagged = add(
            graph,
            "core.attach_payload",
            f"fixture_stamp_{index}",
            (330, (index - 1) * 350),
            payload={
                "ros2": {
                    "header": {
                        "frame_id": "fixture_camera",
                        "stamp": {"sec": 1, "nanosec": (index - 1) * 2_000_000},
                    }
                }
            },
        )
        graph.connect(source.id, "image", tagged.id, "value")
    pair = add(
        graph,
        "industrial.ros2_image_pair",
        "pair",
        (660, 150),
        tolerance_ms=3,
        clock_domain="synthetic_example_time",
    )
    gray = add(graph, "cv.gray", "gray", (990, 150))
    edges = add(graph, "cv.canny", "edges", (1320, 150))
    for index in (1, 2):
        graph.connect(f"fixture_stamp_{index}", "value", pair.id, f"input_{index}")
    graph.connect(pair.id, "input_1", gray.id, "image")
    graph.connect(gray.id, "image", edges.id, "image")
    note(
        graph,
        CREDIT + " Timestamp-pairing control fixture: both streams reuse the SAME "
        "PCB photograph with artificial 1.000 s and 1.002 s headers. These are not two physical "
        "captures or evidence of real camera synchronization. The 3 ms tolerance pairs them "
        "using ordered FIFO and retains both original headers/tickets. This is a one-shot "
        "fixed-time fixture: Reset flow state before replay, otherwise duplicates are ignored. "
        "No ROS installation or network activity is required.",
    )
    graph.save(ROOT / "offline_timestamp_pair.json")

    graph = Graph.load(ROOT / "static_camera_subscriber.json", REGISTRY)
    for key in ("cache", "coordinate", "checked", "project"):
        graph.remove(key)
    add(graph, "industrial.ros2_camera_cache", "cache", (660, 150))
    add(graph, "flow.coordinate", "coordinate", (990, 150), inputs=3)
    add(
        graph,
        "industrial.ros2_rgbd",
        "checked",
        (1320, 150),
        calibration_timing="bound_static",
        max_skew_ms=5,
    )
    add(graph, "3d.depth_cloud", "project", (1650, 150), sample_step=4)
    color = add(
        graph,
        "input.ros2_image",
        "color",
        (0, -300),
        topic="/opencv_dataflow/camera/color_rect",
        ticket_mode="header_stamp",
    )
    pair = add(graph, "industrial.ros2_image_pair", "pair", (330, -150), tolerance_ms=5)
    graph.connect(color.id, "image", pair.id, "input_1")
    graph.connect("depth", "image", pair.id, "input_2")
    graph.connect("camera", "camera", "cache", "camera")
    graph.connect(pair.id, "input_2", "cache", "image")
    graph.blocks["coordinate"].parameters["inputs"] = 3
    graph.connect(pair.id, "input_1", "coordinate", "input_1")
    graph.connect("cache", "image", "coordinate", "input_2")
    graph.connect("cache", "camera", "coordinate", "input_3")
    graph.connect("coordinate", "group", "checked", "group")
    graph.connect("checked", "depth", "project", "depth")
    graph.connect("checked", "camera", "project", "camera")
    graph.blocks["checked"].parameters.update(
        use_color=True, depth_label="input_2", camera_label="input_3", max_skew_ms=5
    )
    graph.connect("checked", "color", "project", "image")
    graph.blocks["source_and_scope"].description = (
        "Disabled approximate RGB-D template: ordered FIFO pairing within 5 ms, then static "
        "CameraInfo binding to the matched depth, exact ticket coordination, registered RGB-D "
        "validation and a colored meter cloud. Declared common ROS clock is not verified clock "
        "synchronization. Original image stamps remain intact; common pair tickets preserve "
        "original ticket provenance. Prime calibration before images. Expiry is cooperative; "
        "duplicate stamps are ignored and backward stamps hold until reset/new run context. "
        "All live endpoints remain disabled. No simulator or robot is started."
    )
    graph.save(ROOT / "approximate_rgbd_subscriber.json")


if __name__ == "__main__":
    build()
