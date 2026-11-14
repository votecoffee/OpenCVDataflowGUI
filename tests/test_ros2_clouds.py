"""Independent binary-layout, units, field-projection and graph checks for PointCloud2."""

import copy
import json
import struct

import numpy as np
import pytest

from opencv_dataflow import ros2_blocks as ros
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE, ValuePacket
from opencv_dataflow.pointclouds import PointCloud
from opencv_dataflow.ros2_clouds import cloud_records, decode_cloud, encode_cloud
from opencv_dataflow.ros2_support.cloud_wire import FIELD_TYPES, ROSPointCloud
from opencv_dataflow.runner import describe
from opencv_dataflow.toolbox import block_entry

HEADER = {"stamp": {"sec": 12, "nanosec": 345678901}, "frame_id": "camera_optical"}


def packet(big=False, invalid=True):
    endian = ">" if big else "<"
    fields = [
        {"name": n, "offset": offset, "datatype": kind, "count": 1}
        for n, offset, kind in [
            ("x", 0, 8),
            ("y", 8, 8),
            ("z", 16, 8),
            ("rgba", 24, 7),
            ("intensity", 28, 7),
            ("id", 32, 10),
        ]
    ]
    raw = bytearray(b"P" * 176)
    for i in range(4):
        offset = (i // 2) * 88 + (i % 2) * 40
        struct.pack_into(
            endian + "dddIfQ",
            raw,
            offset,
            np.nan if invalid and i == 1 else i + 0.5,
            i + 2.0,
            i + 10.0,
            0xFF902030 + i,
            3.25,
            2**60 + i,
        )
    return ROSPointCloud(2, 2, fields, big, 40, 88, bytes(raw), not invalid, HEADER)


@pytest.mark.parametrize("big", [False, True])
def test_padded_float64_packed_float_rgba_and_uint64_extra_fields(big):
    source = packet(big)
    received = ROSPointCloud.from_wire(source.to_wire())
    assert received.data == source.data and received.fields == source.fields
    cloud, details = decode_cloud(received)
    np.testing.assert_array_equal(cloud.points, [[0.5, 2, 10], [2.5, 4, 12], [3.5, 5, 13]])
    np.testing.assert_array_equal(cloud.colors, [[144, 32, 48], [144, 32, 50], [144, 32, 51]])
    assert np.isnan(cloud_records(source)["rgba"]).all()
    np.testing.assert_array_equal(cloud.pixels, [[0, 0], [0, 1], [1, 1]])
    assert cloud.source_size == (2, 2) and cloud.frame == "camera_optical" and cloud.units == "m"
    assert details["dropped"] == 1 and details["ignored_fields"] == ["id", "intensity"]
    assert cloud_records(source)["id"][1, 1] == 2**60 + 3
    assert source.data[80:88] == b"P" * 8
    assert cloud.metadata["ros2"]["header"] == HEADER
    assert json.dumps(source.describe()) and "data_base64" not in json.dumps(details)


@pytest.mark.parametrize("code", FIELD_TYPES)
def test_all_lyrical_field_datatypes_and_vector_counts(code):
    # XYZ plus an arbitrary three-component field; no field information is discarded from packet.
    shape, size = FIELD_TYPES[code]
    fields = [
        {"name": n, "offset": i * 4, "datatype": 7, "count": 1}
        for i, n in enumerate(("x", "y", "z"))
    ]
    fields.append({"name": "descriptor", "offset": 12, "datatype": code, "count": 3})
    source = ROSPointCloud(
        1,
        1,
        fields,
        False,
        12 + size * 3,
        12 + size * 3,
        struct.pack("<fff", 1, 2, 3) + bytes(size * 3),
        True,
        HEADER,
    )
    values = cloud_records(source)
    assert values["descriptor"].shape == (1, 1, 3)
    assert values["descriptor"].dtype == np.dtype(shape)
    cloud, info = decode_cloud(source)
    assert len(cloud.points) == 1 and info["ignored_fields"] == ["descriptor"]


@pytest.mark.parametrize(
    "change",
    [
        {"height": 0},
        {"width": -1},
        {"point_step": 0},
        {"row_step": 79},
        {"data": b"short"},
        {"is_bigendian": 1},
        {"is_dense": 1},
        {"fields": []},
        {"metadata": []},
    ],
)
def test_invalid_cloud_envelope_rejected(change):
    values = copy.deepcopy(vars(packet()))
    with pytest.raises(ValueError):
        ROSPointCloud(**{**values, **change})


@pytest.mark.parametrize(
    "change",
    [
        {"name": ""},
        {"datatype": 12},
        {"datatype": True},
        {"offset": -1},
        {"count": 0},
        {"count": 1025},
        {"offset": 40},
    ],
)
def test_invalid_field_descriptors_rejected(change):
    source = packet()
    source.fields[0].update(change)
    with pytest.raises(ValueError):
        source.to_wire()


def test_aliases_are_allowed_but_duplicate_names_and_missing_xyz_are_rejected():
    source = packet(invalid=False)
    source.fields.append({"name": "rgb_alias", "offset": 24, "datatype": 6, "count": 1})
    source.__post_init__()  # ROS permits aliases into the same record storage.
    source.fields.append(copy.deepcopy(source.fields[0]))
    with pytest.raises(ValueError):
        source.__post_init__()
    source = packet(invalid=False)
    source.fields.pop(0)
    with pytest.raises(ValueError, match="x needs"):
        decode_cloud(source)


def test_dense_claim_invalid_point_rejection_and_explicit_units():
    source = packet()
    with pytest.raises(ValueError, match="Drop"):
        decode_cloud(source, invalid_points="reject")
    source.is_dense = True
    with pytest.raises(ValueError, match="Dense"):
        decode_cloud(source)
    source.is_dense = False
    cloud, _ = decode_cloud(source, units="mm")
    encoded = encode_cloud(cloud)
    values = cloud_records(encoded)
    np.testing.assert_allclose(values["x"][0], cloud.points[:, 0] * 0.001, rtol=1e-7)
    assert encoded.header == HEADER
    cloud.units = "unitless"
    with pytest.raises(ValueError, match="physical"):
        encode_cloud(cloud)


@pytest.mark.parametrize("big", [False, True])
def test_canonical_organization_reconstructs_holes_without_inventing_points(big):
    cloud, _ = decode_cloud(packet(big))
    encoded = encode_cloud(
        cloud, organization="source_pixels", byte_order="big" if big else "little"
    )
    assert encoded.height == 2 and encoded.width == 2 and not encoded.is_dense
    assert np.isnan(cloud_records(encoded)["x"][0, 1])
    decoded, details = decode_cloud(encoded)
    np.testing.assert_array_equal(decoded.points, cloud.points)
    np.testing.assert_array_equal(decoded.colors, cloud.colors)
    np.testing.assert_array_equal(decoded.pixels, cloud.pixels)
    assert details["dropped"] == 1
    cloud.pixels[1] = cloud.pixels[0]
    with pytest.raises(ValueError, match="same pixel"):
        encode_cloud(cloud, organization="source_pixels")


def test_normals_are_normalized_and_invalid_normals_follow_drop_policy():
    fields = [
        {"name": n, "offset": i * 4, "datatype": 7, "count": 1}
        for i, n in enumerate(("x", "y", "z", "normal_x", "normal_y", "normal_z"))
    ]
    raw = struct.pack("<12f", 1, 2, 3, 0, 0, 2, 4, 5, 6, 0, 0, 0)
    source = ROSPointCloud(1, 2, fields, False, 24, 48, raw, True, HEADER)
    cloud, details = decode_cloud(source)
    np.testing.assert_array_equal(cloud.normals, [[0, 0, 1]])
    assert details["invalid_normals"] == 1
    ignored, _ = decode_cloud(source, normals="ignore")
    assert len(ignored.points) == 2 and ignored.normals is None
    with pytest.raises(ValueError, match="Drop"):
        decode_cloud(source, invalid_points="reject")
    np.testing.assert_array_equal(decode_cloud(encode_cloud(cloud))[0].normals, cloud.normals)


def test_empty_cloud_and_float64_overflow_are_explicit():
    source = PointCloud(np.zeros((0, 3)), frame="empty")
    cloud, details = decode_cloud(encode_cloud(source))
    assert len(cloud.points) == 0 and details["point_count"] == 0
    source = packet(invalid=False)
    raw = bytearray(source.data)
    struct.pack_into("<d", raw, 0, 1e100)
    source.data = bytes(raw)
    with pytest.raises(ValueError, match="overflow"):
        decode_cloud(source)


def test_color_bytes_frame_fallback_and_raw_data_retention():
    fields = [
        {"name": n, "offset": i * 4, "datatype": 7, "count": 1}
        for i, n in enumerate(("x", "y", "z"))
    ]
    fields += [
        {"name": n, "offset": i + 12, "datatype": 2, "count": 1}
        for i, n in enumerate(("r", "g", "b"))
    ]
    source = ROSPointCloud(
        1,
        1,
        fields,
        False,
        15,
        15,
        struct.pack("<fffBBB", 1, 2, 3, 5, 6, 7),
        True,
        {**HEADER, "frame_id": ""},
        {"custom": "preserved"},
    )
    cloud, _ = decode_cloud(source, empty_frame="sensor")
    assert cloud.frame == "sensor" and cloud.metadata["custom"] == "preserved"
    np.testing.assert_array_equal(cloud.colors, [[5, 6, 7]])
    assert decode_cloud(source, colors="ignore")[0].colors is None


def test_block_registry_parameters_tickets_waiting_and_failed_acquisition(monkeypatch):
    class Connection:
        sample = None

        def request(self, *args, **kwargs):
            return {"sample": self.sample, "info": {"rmw": "fake", "topic": "/test"}}

        def close(self):
            pass

    connection = Connection()
    monkeypatch.setattr(ros, "ROSClient", lambda *args: connection)
    source = ros.ReadROSCloud(parameters={"enabled": True, "ticket_mode": "header_stamp"})
    try:
        assert source.execute({}).outputs["cloud"] is NO_VALUE
        connection.sample = packet().to_wire()
        result = source.execute({})
        assert result is not None, source.error
        assert result.outputs["ticket"]["value"] == "camera_optical|12:345678901"
        assert len(result.outputs["cloud"].points) == 3
        assert result.payloads["cloud"]["ticket"] == result.outputs["ticket"]
        assert result.outputs["message"].data == packet().data
        assert block_entry(source).path == ("Industrial integration", "ROS 2")
        source.parameters["invalid_points"] = "reject"
        assert source.execute({}) is None
        assert not source.current_emissions
    finally:
        source.close()


def test_converters_xml_reports_and_writer_enable_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(ros, "ROSClient", lambda *args: pytest.fail("Offline blocks started ROS"))
    registry = builtin_registry()
    for kind in (
        "input.ros2_cloud",
        "industrial.ros2_cloud_encode",
        "industrial.ros2_cloud_decode",
        "industrial.ros2_cloud_publish",
    ):
        registry.create(kind).validate_parameters()
    cloud, _ = decode_cloud(packet())
    encoded = ros.EncodeROSCloud().execute({"cloud": cloud})
    result = ros.DecodeROSCloud().execute({"message": encoded.outputs["message"]})
    np.testing.assert_array_equal(result.outputs["cloud"].points, cloud.points)
    writer = ros.PublishROSCloud()
    assert writer.execute({"cloud": cloud, "enable": True}).outputs["published"] is NO_VALUE
    writer.parameters["write_enabled"] = True
    assert writer.execute({"cloud": cloud, "enable": False}).outputs["published"] is NO_VALUE
    graph = Graph()
    graph.add(registry.create("industrial.ros2_cloud_decode", parameters={"units": "mm"}))
    path = tmp_path / "cloud.xml"
    graph.save(path)
    restored = Graph.load(path, registry)
    assert next(iter(restored.blocks.values())).parameters["units"] == "mm"
    # Report helper describes binary messages, never the point data/base64.
    decoder = next(iter(restored.blocks.values()))
    decoder.execute({"message": packet()})
    receipt = json.dumps(describe(restored))
    assert "data_base64" not in receipt and "point_count" in receipt
    restored.close()
    graph.close()


def test_local_payload_overrides_survive_codec_projection_without_mutating_caller():
    cloud, _ = decode_cloud(packet())
    original_metadata = copy.deepcopy(cloud.metadata)
    wrapped = ValuePacket(cloud, {"inspection": {"recipe": "A", "revision": 3}})
    encoder = ros.EncodeROSCloud()
    encoded = encoder.execute({"cloud": wrapped})
    assert cloud.metadata == original_metadata
    portable = encoded.outputs["message"]
    assert portable.metadata["inspection"]["revision"] == 3
    decoder = ros.DecodeROSCloud()
    decoded = decoder.execute({"message": ValuePacket(portable, {"station": "test"})})
    assert "station" not in portable.metadata
    assert decoded.outputs["cloud"].metadata["station"] == "test"
    assert decoded.outputs["cloud"].metadata["inspection"]["recipe"] == "A"


def test_raw_publisher_retains_unknown_fields_and_validates_selected_input(monkeypatch):
    published = []

    class Connection:
        def request(self, operation, *args, **kwargs):
            published.append(kwargs["value"])
            return {"published": True}

        def close(self):
            pass

    monkeypatch.setattr(ros, "ROSClient", lambda *args: Connection())
    writer = ros.PublishROSCloud(parameters={"write_enabled": True, "input_mode": "message"})
    source = packet(True)
    result = writer.execute({"message": source, "enable": True})
    assert result is not None, writer.error
    assert published == [source.to_wire()]
    writer.parameters["stamp_mode"] = "zero"
    writer.execute({"message": source, "enable": True})
    assert published[-1]["header"]["stamp"] == {"sec": 0, "nanosec": 0}
    assert source.header == HEADER
    assert writer.execute({"enable": True}) is None
    assert len(published) == 2
    writer.close()
