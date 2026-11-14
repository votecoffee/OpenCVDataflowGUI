"""Opt-in retained CameraInfo received once and bound to multiple real DDS images."""

import copy
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


def test_retained_camera_info_replays_once_and_calibrates_multiple_image_tickets():
    prefix = "/opencv_dataflow/static_test_" + uuid.uuid4().hex
    registry, graph, clients, sources = builtin_registry(), Graph(), [], []
    try:
        calibration = camera_to_info(model())
        calibration["header"]["stamp"] = {"sec": 0, "nanosec": 0}
        writer = ROSClient(
            {
                **ROSBlock.defaults,
                "kind": "data",
                "role": "write",
                "message_type": "sensor_msgs/msg/CameraInfo",
                "topic": prefix + "/info",
                "durability": "transient_local",
            },
            30,
        )
        clients.append(writer)
        receipt = writer.request(
            "publish",
            5,
            value=calibration,
            timeout=1,
            require_subscriber=False,
            wait_for_ack=False,
            stamp_mode="preserve",
        )
        assert receipt["published"] and receipt["matched_subscribers"] == 0
        info = graph.add(
            registry.create(
                "input.ros2_data",
                block_id="info",
                parameters={
                    "enabled": True,
                    "message_type": "sensor_msgs/msg/CameraInfo",
                    "topic": prefix + "/info",
                    "durability": "transient_local",
                    "timeout_seconds": 0.05,
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
                    "timeout_seconds": 0.05,
                },
            )
        )
        sources = [info, depth]
        decoder = graph.add(registry.create("industrial.ros2_camera_decode", block_id="decoder"))
        cache = graph.add(registry.create("industrial.ros2_camera_cache", block_id="cache"))
        coordinate = graph.add(registry.create("flow.coordinate", block_id="coordinate"))
        checked = graph.add(
            registry.create(
                "industrial.ros2_rgbd",
                block_id="checked",
                parameters={
                    "use_color": False,
                    "depth_label": "input_1",
                    "camera_label": "input_2",
                    "calibration_timing": "bound_static",
                },
            )
        )
        cloud = graph.add(
            registry.create("3d.depth_cloud", block_id="cloud", parameters={"sample_step": 1})
        )
        graph.connect(info.id, "data", decoder.id, "message")
        graph.connect(decoder.id, "camera", cache.id, "camera")
        graph.connect(depth.id, "image", cache.id, "image")
        graph.connect(cache.id, "image", coordinate.id, "input_1")
        graph.connect(cache.id, "camera", coordinate.id, "input_2")
        graph.connect(coordinate.id, "group", checked.id, "group")
        graph.connect(checked.id, "depth", cloud.id, "depth")
        graph.connect(checked.id, "camera", cloud.id, "camera")
        end = time.monotonic() + 10
        while True:
            graph.run()
            assert not any(b.state == "error" for b in graph.blocks.values()), {
                b.id: b.error for b in graph.blocks.values()
            }
            if cache.runtime["cached"] is not None:
                break
            assert time.monotonic() < end, "Retained CameraInfo was not delivered to late reader"
        assert not any(cache.current_emissions.values())
        original_ticket = copy.deepcopy(cache.runtime["cached"]["payload"]["ticket"])
        image_writer = ROSClient(
            {
                **ROSBlock.defaults,
                "kind": "image",
                "role": "write",
                "topic": prefix + "/depth",
                "message_type": "sensor_msgs/msg/Image",
            },
            30,
        )
        clients.append(image_writer)
        revision = None
        for stamp in (18, 19):
            header = {"frame_id": "camera_optical", "stamp": {"sec": stamp, "nanosec": 0}}
            value = ImageArray(
                np.full((4, 6), 1000, np.uint16), "gray", {"ros2": {"header": header}}
            )
            publish(image_writer, encode_image(value).to_wire())
            end = time.monotonic() + 3
            while True:
                graph.run()
                assert not any(b.state == "error" for b in graph.blocks.values()), {
                    b.id: b.error for b in graph.blocks.values()
                }
                if cloud.current_emissions.get("cloud"):
                    break
                assert time.monotonic() < end, "No cloud produced for fresh image"
            result = checked.last_result.outputs["info"]
            binding = result["static_calibration"]
            assert binding["image_header"] == header
            assert binding["calibration_header"]["stamp"] == {"sec": 0, "nanosec": 0}
            assert binding["calibration_ticket"] == original_ticket
            assert revision is None or revision == binding["calibration_revision"]
            revision = binding["calibration_revision"]
            points = cloud.last_result.outputs["cloud"].points
            assert len(points) == 24
            np.testing.assert_allclose(points[0], [-0.2, -1 / 12, 1], rtol=1e-6)
        assert cache.runtime["cached"]["payload"]["ticket"] == original_ticket
    finally:
        owned = clients + [s._client for s in sources if s._client is not None]
        graph.close()
        for client in clients:
            client.close()
        assert all(c.process.poll() == 0 for c in owned)
        assert all(not t.is_alive() for c in owned for t in c._threads)
