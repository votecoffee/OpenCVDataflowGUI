"""Opt-in real Fast DDS PointCloud2 transport on localhost/private test topics."""

import os
import uuid

import numpy as np
import pytest
from test_ros2_clouds import packet
from test_ros2_native import pair, poll, publish

from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.ros2_blocks import PublishROSCloud, ReadROSCloud
from opencv_dataflow.ros2_client import ROSRemoteError
from opencv_dataflow.ros2_clouds import decode_cloud
from opencv_dataflow.ros2_support.cloud_wire import ROSPointCloud

pytestmark = pytest.mark.skipif(
    os.environ.get("OPENCV_DATAFLOW_TEST_ROS2") != "1", reason="Opt-in local ROS runtime required"
)


@pytest.mark.parametrize("big", [False, True])
def test_actual_padded_organized_cloud_keeps_exact_fields_bytes_and_header(big):
    with pair("sensor_msgs/msg/PointCloud2", "cloud") as (reader, writer):
        source = packet(big)
        publish(writer, source.to_wire())
        received = ROSPointCloud.from_wire(poll(reader)["sample"])
        assert received.to_wire() == source.to_wire()
        # Includes 64-bit integer extras and packed float RGBA whose bits encode NaN.
        cloud, details = decode_cloud(received)
        assert len(cloud.points) == 3 and details["ignored_fields"] == ["id", "intensity"]
        np.testing.assert_array_equal(cloud.colors[0], [144, 32, 48])


def test_actual_cloud_blocks_publish_meter_coordinates_normals_and_ticket():
    topic = "/opencv_dataflow/test_" + uuid.uuid4().hex
    reader = ReadROSCloud(
        parameters={
            "enabled": True,
            "topic": topic,
            "reliability": "reliable",
            "ticket_mode": "header_stamp",
        }
    )
    writer = PublishROSCloud(
        parameters={
            "write_enabled": True,
            "topic": topic,
            "reliability": "reliable",
            "timeout_seconds": 2.0,
            "organization": "source_pixels",
            "byte_order": "big",
        }
    )
    clients = []
    try:
        assert reader.execute({}).outputs["cloud"] is NO_VALUE
        cloud, _ = decode_cloud(packet())
        cloud.points *= 1000
        cloud.units = "mm"
        cloud.normals = np.tile([0.0, 0.0, 1.0], (len(cloud.points), 1))
        delivery = writer.execute({"cloud": cloud, "enable": True})
        assert delivery is not None, writer.error
        assert delivery.outputs["delivery"]["dds_acknowledged"]
        result = reader.execute({})
        assert result is not None, reader.error
        np.testing.assert_allclose(result.outputs["cloud"].points, cloud.points * 0.001, rtol=1e-7)
        np.testing.assert_array_equal(result.outputs["cloud"].normals, cloud.normals)
        assert result.outputs["cloud"].units == "m"
        assert result.outputs["ticket"]["value"] == "camera_optical|12:345678901"
        assert result.outputs["message"].height == 2 and result.outputs["message"].is_bigendian
    finally:
        clients = [c for c in (reader._client, writer._client) if c is not None]
        reader.close()
        writer.close()
        assert all(c.process.poll() == 0 for c in clients)
        assert all(not t.is_alive() for c in clients for t in c._threads)


def test_actual_cloud_writer_rejects_bad_layout_before_publication():
    with pair("sensor_msgs/msg/PointCloud2", "cloud") as (reader, writer):
        value = packet().to_wire()
        value["fields"][0]["offset"] = value["point_step"]
        with pytest.raises(ROSRemoteError, match="beyond"):
            publish(writer, value)
        assert reader.request("poll", 1, timeout=0.05)["sample"] is None
        publish(writer, packet().to_wire())
        assert poll(reader)["sample"] is not None


def test_actual_raw_publisher_preserves_unknown_fields_and_padded_nan_bits():
    with pair("sensor_msgs/msg/PointCloud2", "cloud") as (reader, endpoint):
        writer = PublishROSCloud(
            parameters={
                "write_enabled": True,
                "input_mode": "message",
                "reliability": "reliable",
                "timeout_seconds": 2.0,
                "topic": reader.info["topic"],
            }
        )
        try:
            source = packet(True)
            result = writer.execute({"message": source, "enable": True})
            assert result is not None, writer.error
            assert result.outputs["delivery"]["dds_acknowledged"]
            assert ROSPointCloud.from_wire(poll(reader)["sample"]).to_wire() == source.to_wire()
        finally:
            writer.close()
