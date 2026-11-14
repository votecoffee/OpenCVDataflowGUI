"""Opt-in localhost Lyrical/Fast DDS qualification; no robot commands or UI."""

import os
import time
import uuid
from contextlib import contextmanager

import numpy as np
import pytest

from opencv_dataflow.ros2_blocks import ReadROSData, ROSBlock
from opencv_dataflow.ros2_client import ROSClient, ROSRemoteError
from opencv_dataflow.ros2_images import decode_image, encode_image
from opencv_dataflow.ros2_support.wire import ROSImage

pytestmark = pytest.mark.skipif(
    os.environ.get("OPENCV_DATAFLOW_TEST_ROS2") != "1", reason="Opt-in local ROS runtime required"
)
HEADER = {"stamp": {"sec": 42, "nanosec": 987654321}, "frame_id": "camera_optical"}


@contextmanager
def pair(message_type="std_msgs/msg/String", kind="data", **overrides):
    config = {
        **ROSBlock.defaults,
        "topic": "/opencv_dataflow/test_" + uuid.uuid4().hex,
        "domain_id": 71,
        "discovery_range": "localhost",
        "kind": kind,
        "message_type": message_type,
        **overrides,
    }
    clients = []
    try:
        reader = ROSClient({**config, "role": "read"}, 30)
        clients.append(reader)
        writer = ROSClient({**config, "role": "write"}, 30)
        clients.append(writer)
        assert reader.info["rmw"] == writer.info["rmw"] == "rmw_fastrtps_cpp"
        assert reader.info["ros_distro"] == "lyrical"
        yield reader, writer
    finally:
        for client in clients:
            client.close()
        assert all(c.process.poll() == 0 for c in clients)
        assert all(not t.is_alive() for c in clients for t in c._threads)


def publish(writer, value):
    result = writer.request(
        "publish",
        8,
        value=value,
        timeout=2,
        require_subscriber=True,
        wait_for_ack=True,
        stamp_mode="preserve",
    )
    assert result["published"] and result["dds_acknowledged"]
    assert result["application_acknowledged"] is False
    return result


def poll(reader):
    return reader.request("poll", 4, timeout=2)


def test_actual_color_depth_and_padded_images():
    with pair("sensor_msgs/msg/Image", "image") as (reader, writer):
        from opencv_dataflow.images import ImageArray

        images = [
            ImageArray(np.arange(60, dtype=np.uint8).reshape(4, 5, 3), "bgr"),
            ImageArray(np.arange(60, dtype=np.uint16).reshape(4, 5, 3), "rgb"),
            ImageArray(np.array([[0, 0.5, 50], [2, -1, np.nan]], np.float32), "gray"),
        ]
        packets = [encode_image(i) for i in images]
        packets.append(
            ROSImage(2, 2, "mono16", 6, 1, b"\x00\x01\x01\x00xx\xff\xff\x00\x02yy", HEADER)
        )
        for packet in packets:
            packet.header = HEADER
            delivery = publish(writer, packet.to_wire())
            received = ROSImage.from_wire(poll(reader)["sample"])
            assert received.header == delivery["header"] == HEADER
            assert received.encoding == packet.encoding and received.data == packet.data
            assert received.step == packet.step and received.is_bigendian == packet.is_bigendian
            np.testing.assert_array_equal(decode_image(received), decode_image(packet))


@pytest.mark.parametrize(
    "message_type,value",
    [
        ("std_msgs/msg/String", {"data": '{"part_id":"trial-1","pass":true}'}),
        (
            "geometry_msgs/msg/PoseStamped",
            {
                "header": HEADER,
                "pose": {
                    "position": {"x": 0.1, "y": -0.2, "z": 0.3},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
            },
        ),
        (
            "sensor_msgs/msg/JointState",
            {
                "header": HEADER,
                "name": ["joint_1", "joint_2"],
                "position": [0.1, -0.2],
                "velocity": [],
                "effort": [],
            },
        ),
        ("rosgraph_msgs/msg/Clock", {"clock": {"sec": 10, "nanosec": 987654321}}),
        ("std_msgs/msg/Int64", {"data": 2**60 + 1}),
    ],
)
def test_actual_structured_messages(message_type, value):
    with pair(message_type) as (reader, writer):
        publish(writer, value)
        assert poll(reader)["sample"] == value


def test_actual_invalid_field_is_rejected_before_publish():
    with pair("std_msgs/msg/Int32") as (reader, writer):
        with pytest.raises(ROSRemoteError):
            publish(writer, {"data": 3.7})
        assert reader.request("poll", 1, timeout=0.05)["sample"] is None
        publish(writer, {"data": 3})
        assert poll(reader)["sample"] == {"data": 3}


def test_actual_latest_queue_drops_old_samples_and_rejects_stale():
    with pair(consume="latest", queue_depth=2, max_age_seconds=0.2) as (reader, writer):
        for index in range(4):
            publish(writer, {"data": str(index)})
        result = poll(reader)
        assert result["sample"] == {"data": "3"}
        assert result["info"]["received"] == 4 and result["info"]["dropped"] == 3
        publish(writer, {"data": "old"})
        time.sleep(0.25)
        assert reader.request("poll", 1, timeout=0.05)["sample"] is None


def test_actual_qos_mismatch_does_not_publish():
    config = {
        **ROSBlock.defaults,
        "topic": "/opencv_dataflow/test_" + uuid.uuid4().hex,
        "role": "read",
        "kind": "data",
        "message_type": "std_msgs/msg/String",
    }
    reader = ROSClient(config, 30)  # Requests reliable delivery.
    writer = None
    try:
        writer = ROSClient({**config, "role": "write", "reliability": "best_effort"}, 30)
        result = writer.request(
            "publish",
            3,
            value={"data": "no compatible subscriber"},
            timeout=0.2,
            require_subscriber=True,
            wait_for_ack=False,
            stamp_mode="preserve",
        )
        assert not result["published"] and not result["attempted"]
        assert reader.request("poll", 1, timeout=0.05)["sample"] is None
    finally:
        reader.close()
        if writer:
            writer.close()


def test_actual_ticketed_block_reader_and_missing_peer():
    topic = "/opencv_dataflow/test_" + uuid.uuid4().hex
    reader = ReadROSData(
        parameters={
            "enabled": True,
            "topic": topic,
            "data_mode": "json_string",
            "ticket_mode": "field",
            "ticket_field": "part",
        }
    )
    writer = None
    try:
        from opencv_dataflow.payloads import NO_VALUE

        assert reader.execute({}).outputs["data"] is NO_VALUE
        config = {
            **ROSBlock.defaults,
            "topic": topic,
            "role": "write",
            "kind": "data",
            "message_type": "std_msgs/msg/String",
        }
        writer = ROSClient(config, 30)
        publish(writer, {"data": '{"part":"native-test","score":0.9}'})
        result = reader.execute({})
        assert result.outputs["ticket"]["value"] == "native-test"
        assert result.outputs["data"]["score"] == 0.9
        reader.close()
        # Discovery removal may propagate asynchronously; use a new topic with no peer.
        writer.close()
        writer = ROSClient({**config, "topic": topic + "_absent"}, 30)
        result = writer.request(
            "publish",
            2,
            value={"data": "none"},
            timeout=0.05,
            require_subscriber=True,
            wait_for_ack=True,
            stamp_mode="preserve",
        )
        assert result == {"published": False, "attempted": False, "matched_subscribers": 0}
    finally:
        reader.close()
        if writer:
            writer.close()


def test_actual_image_blocks_best_effort_and_sim_clock_stamping():
    from opencv_dataflow.payloads import NO_VALUE
    from opencv_dataflow.ros2_blocks import PublishROSImage, ReadROSImage

    topic = "/opencv_dataflow/test_" + uuid.uuid4().hex
    reader = ReadROSImage(parameters={"enabled": True, "topic": topic})
    writer = PublishROSImage(
        parameters={
            "write_enabled": True,
            "topic": topic,
            "use_sim_time": True,
            "stamp_mode": "node_clock",
            "frame_id": "camera_optical",
            "timeout_seconds": 2.0,
        }
    )
    try:
        assert reader.execute({}).outputs["image"] is NO_VALUE
        image = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
        delivery = writer.execute({"image": image, "enable": True})
        assert delivery is not None, writer.error
        assert delivery.outputs["published"]
        assert delivery.outputs["delivery"]["dds_acknowledged"] is False
        result = reader.execute({})
        assert result is not None, reader.error
        np.testing.assert_array_equal(result.outputs["image"], image)
        # No /clock publisher: simulated node time remains zero; no wall-time substitution.
        assert result.outputs["image"].metadata["ros2"]["header"] == {
            "stamp": {"sec": 0, "nanosec": 0},
            "frame_id": "camera_optical",
        }
    finally:
        reader.close()
        writer.close()
