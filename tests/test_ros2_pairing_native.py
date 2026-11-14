"""Opt-in mismatched DDS image stamps -> FIFO pair -> static RGB-D -> meter cloud."""

import os
import time
import uuid

import numpy as np
import pytest
from test_ros2_calibration import model
from test_ros2_native import publish

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.ros2_blocks import ROSBlock
from opencv_dataflow.ros2_calibration import camera_to_info
from opencv_dataflow.ros2_client import ROSClient
from opencv_dataflow.ros2_images import encode_image

pytestmark = pytest.mark.skipif(
    os.environ.get("OPENCV_DATAFLOW_TEST_ROS2") != "1", reason="Opt-in localhost Lyrical required"
)


def test_staggered_near_time_rgb_depth_with_static_calibration():
    registry, graph = builtin_registry(), Graph()
    prefix = "/opencv_dataflow/time_pair_test_" + uuid.uuid4().hex
    sources, clients = [], []
    try:
        for name in ("color", "depth"):
            sources.append(
                graph.add(
                    registry.create(
                        "input.ros2_image",
                        block_id=name,
                        parameters={
                            "enabled": True,
                            "topic": prefix + "/" + name,
                            "ticket_mode": "header_stamp",
                            "reliability": "reliable",
                            "timeout_seconds": 0.03,
                        },
                    )
                )
            )
        info = graph.add(
            registry.create(
                "input.ros2_data",
                block_id="info",
                parameters={
                    "enabled": True,
                    "topic": prefix + "/info",
                    "message_type": "sensor_msgs/msg/CameraInfo",
                    "timeout_seconds": 0.03,
                },
            )
        )
        sources.append(info)
        decoder = graph.add(registry.create("industrial.ros2_camera_decode", block_id="decoder"))
        pair = graph.add(registry.create("industrial.ros2_image_pair", block_id="pair"))
        cache = graph.add(registry.create("industrial.ros2_camera_cache", block_id="cache"))
        coordinate = graph.add(
            registry.create("flow.coordinate", block_id="coordinate", parameters={"inputs": 3})
        )
        checked = graph.add(
            registry.create(
                "industrial.ros2_rgbd",
                block_id="checked",
                parameters={
                    "calibration_timing": "bound_static",
                    "max_skew_ms": 5,
                },
            )
        )
        cloud = graph.add(
            registry.create("3d.depth_cloud", block_id="cloud", parameters={"sample_step": 1})
        )
        graph.connect(info.id, "data", decoder.id, "message")
        graph.connect(decoder.id, "camera", cache.id, "camera")
        graph.connect("color", "image", pair.id, "input_1")
        graph.connect("depth", "image", pair.id, "input_2")
        graph.connect(pair.id, "input_2", cache.id, "image")
        graph.connect(pair.id, "input_1", coordinate.id, "input_1")
        graph.connect(cache.id, "image", coordinate.id, "input_2")
        graph.connect(cache.id, "camera", coordinate.id, "input_3")
        graph.connect(coordinate.id, "group", checked.id, "group")
        for name, port in (("color", "image"), ("depth", "depth"), ("camera", "camera")):
            graph.connect(checked.id, name, cloud.id, port)
        graph.run()
        for name, kind, message in (
            ("color", "image", "sensor_msgs/msg/Image"),
            ("depth", "image", "sensor_msgs/msg/Image"),
            ("info", "data", "sensor_msgs/msg/CameraInfo"),
        ):
            clients.append(
                ROSClient(
                    {
                        **ROSBlock.defaults,
                        "kind": kind,
                        "role": "write",
                        "message_type": message,
                        "topic": prefix + "/" + name,
                    },
                    30,
                )
            )

        def poll_until(predicate):
            end = time.monotonic() + 4
            while True:
                graph.run()
                assert not any(b.state == "error" for b in graph.blocks.values()), {
                    b.id: b.error for b in graph.blocks.values()
                }
                if predicate():
                    return
                assert time.monotonic() < end, "DDS graph did not reach expected state"

        calibration = camera_to_info(model())
        calibration["header"]["stamp"] = {"sec": 0, "nanosec": 0}
        publish(clients[2], calibration)
        poll_until(lambda: cache.runtime["cached"] is not None)

        def send(index, ns):
            header = {"stamp": {"sec": 18, "nanosec": ns}, "frame_id": "camera_optical"}
            pixels = (
                np.full((4, 6, 3), 70, np.uint8) if index == 0 else np.full((4, 6), 1000, np.uint16)
            )
            value = ImageArray(
                pixels, "bgr" if index == 0 else "gray", {"ros2": {"header": header}}
            )
            publish(clients[index], encode_image(value).to_wire())

        send(0, 0)
        send(1, 6_000_000)
        poll_until(lambda: pair.runtime["dropped"] == 1)
        assert not cloud.current_emissions.get("cloud")
        send(0, 7_000_000)
        poll_until(lambda: bool(cloud.current_emissions.get("cloud")))
        paired = pair.last_result.outputs["info"]
        assert paired["spread_ns"] == 1_000_000
        assert paired["members"]["input_1"]["ticket"] != paired["members"]["input_2"]["ticket"]
        result = checked.last_result
        assert result.outputs["info"]["header_spread_ns"] == 1_000_000
        binding = result.outputs["info"]["static_calibration"]
        assert binding["calibration_header"]["stamp"] == {"sec": 0, "nanosec": 0}
        assert binding["image_header"]["stamp"]["nanosec"] == 6_000_000
        points = cloud.last_result.outputs["cloud"]
        assert len(points.points) == 24 and points.units == "m"
        np.testing.assert_allclose(points.points[0], [-0.2, -1 / 12, 1], rtol=1e-6)
        np.testing.assert_array_equal(points.colors, np.full((24, 3), 70, np.uint8))
    finally:
        owned = clients + [s._client for s in sources if s._client is not None]
        graph.close()
        for client in clients:
            client.close()
        assert all(c.process.poll() == 0 for c in owned)
        assert all(not t.is_alive() for c in owned for t in c._threads)
