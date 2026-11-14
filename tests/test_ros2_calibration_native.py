"""Opt-in native CameraInfo and staggered three-stream calibrated RGB-D acquisition."""

import os
import uuid

import numpy as np
import pytest
from test_ros2_calibration import HEADER, model
from test_ros2_native import pair, poll, publish

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.ros2_blocks import ROSBlock
from opencv_dataflow.ros2_calibration import camera_from_info, camera_to_info
from opencv_dataflow.ros2_client import ROSClient
from opencv_dataflow.ros2_images import encode_image

pytestmark = pytest.mark.skipif(
    os.environ.get("OPENCV_DATAFLOW_TEST_ROS2") != "1", reason="Opt-in local ROS runtime required"
)


def test_actual_camera_info_fields_roundtrip_and_pixel_calibration():
    fields = camera_to_info(model())
    with pair("sensor_msgs/msg/CameraInfo") as (reader, writer):
        publish(writer, fields)
        received = poll(reader)["sample"]
        assert received == fields
        converted, details = camera_from_info(received)
        assert converted["camera_matrix"] == model()["camera_matrix"]
        assert details["header"] == HEADER


def test_actual_staggered_rgb_depth_calibration_match_one_acquisition():
    registry, graph = builtin_registry(), Graph()
    prefix = "/opencv_dataflow/test_" + uuid.uuid4().hex
    sources = []
    clients = []
    owned = []
    try:
        rgb = graph.add(
            registry.create(
                "input.ros2_image",
                block_id="rgb",
                parameters={
                    "enabled": True,
                    "topic": prefix + "/color",
                    "ticket_mode": "header_stamp",
                    "reliability": "reliable",
                    "timeout_seconds": 0.02,
                },
            )
        )
        depth = graph.add(
            registry.create(
                "input.ros2_image",
                block_id="depth",
                parameters={
                    "enabled": True,
                    "topic": prefix + "/depth",
                    "ticket_mode": "header_stamp",
                    "reliability": "reliable",
                    "timeout_seconds": 0.02,
                },
            )
        )
        source = graph.add(
            registry.create(
                "input.ros2_data",
                block_id="info",
                parameters={
                    "enabled": True,
                    "topic": prefix + "/info",
                    "message_type": "sensor_msgs/msg/CameraInfo",
                    "ticket_mode": "header_stamp",
                    "timeout_seconds": 0.02,
                },
            )
        )
        sources = [rgb, depth, source]
        decoder = graph.add(registry.create("industrial.ros2_camera_decode", block_id="camera"))
        coord = graph.add(
            registry.create(
                "flow.coordinate",
                block_id="coord",
                parameters={"inputs": 3, "max_wait_ms": 10000.0},
            )
        )
        checked = graph.add(registry.create("industrial.ros2_rgbd", block_id="checked"))
        cloud = graph.add(
            registry.create("3d.depth_cloud", block_id="cloud", parameters={"sample_step": 1})
        )
        graph.connect(source.id, "data", decoder.id, "message")
        for block, output, port in (
            (rgb, "image", "input_1"),
            (depth, "image", "input_2"),
            (decoder, "camera", "input_3"),
        ):
            graph.connect(block.id, output, coord.id, port)
        graph.connect(coord.id, "group", checked.id, "group")
        for output, port in (("depth", "depth"), ("camera", "camera"), ("color", "image")):
            graph.connect(checked.id, output, cloud.id, port)
        graph.run()  # Initialize three independent subscribers, with no acquisitions yet.
        assert all(b.state == "waiting" for b in sources)
        for suffix, kind, msg in (
            ("color", "image", "sensor_msgs/msg/Image"),
            ("depth", "image", "sensor_msgs/msg/Image"),
            ("info", "data", "sensor_msgs/msg/CameraInfo"),
        ):
            client = ROSClient(
                {
                    **ROSBlock.defaults,
                    "role": "write",
                    "kind": kind,
                    "message_type": msg,
                    "topic": prefix + "/" + suffix,
                },
                30,
            )
            clients.append(client)
        publish(clients[2], camera_to_info(model()))
        graph.run()
        assert coord.state == "waiting" and cloud.state == "inactive"
        pixels = np.full((4, 6), 1000, np.uint16)
        pixels[0, 0] = 0
        publish(
            clients[1],
            encode_image(ImageArray(pixels, "gray", {"ros2": {"header": HEADER}})).to_wire(),
        )
        graph.run()
        assert coord.state == "waiting" and cloud.state == "inactive"
        colors = ImageArray(np.full((4, 6, 3), 70, np.uint8), "rgb", {"ros2": {"header": HEADER}})
        publish(clients[0], encode_image(colors).to_wire())
        graph.run()
        assert checked.state == cloud.state == "pass", {
            b.id: b.error for b in graph.blocks.values()
        }
        result = cloud.last_result.outputs["cloud"]
        assert len(result.points) == 23 and result.units == "m" and result.frame == "camera_optical"
        np.testing.assert_allclose(result.points[0], [-0.1, -1 / 12, 1], rtol=1e-6)
        assert checked.last_result.outputs["info"]["header_spread_ns"] == 0
        assert (
            checked.last_result.payloads["depth"]["ticket"]["value"]
            == "camera_optical|17:987654321"
        )
    finally:
        owned = clients + [s._client for s in sources if s._client is not None]
        graph.close()
        for client in clients:
            client.close()
        assert all(c.process.poll() == 0 for c in owned)
        assert all(not t.is_alive() for c in owned for t in c._threads)
