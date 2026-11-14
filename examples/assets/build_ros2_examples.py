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


if __name__ == "__main__":
    build()
