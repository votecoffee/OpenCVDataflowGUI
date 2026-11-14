"""ROS contracts without ROS imports, equipment, discovery or network access."""

import io
import json
import struct
import subprocess
import sys

import numpy as np
import pytest

from opencv_dataflow import ros2_blocks as ros
from opencv_dataflow import ros2_client as transport
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, frame, metadata
from opencv_dataflow.ros2_images import decode_image, encode_image
from opencv_dataflow.ros2_support.native import preserved
from opencv_dataflow.ros2_support.wire import (
    MAX_FRAME_BYTES,
    ROSImage,
    json_bytes,
    parse_json,
    read_frame,
    write_frame,
)
from opencv_dataflow.runner import describe
from opencv_dataflow.toolbox import block_entry

HEADER = {"stamp": {"sec": 42, "nanosec": 123456789}, "frame_id": "camera_optical"}


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16, np.int32, np.float32, np.float64])
def test_scalar_images_preserve_samples_header_and_payload(dtype):
    pixels = np.arange(63, dtype=dtype).reshape(7, 9)[:, ::2]
    payload = {"ros2": {"header": HEADER}, "spatial": frame([5, 7], "capture"), "custom": "kept"}
    image = ImageArray(pixels, "gray", payload)
    packet = encode_image(image)
    decoded = decode_image(packet)
    np.testing.assert_array_equal(decoded, pixels)
    assert decoded.dtype == pixels.dtype and decoded.flags.c_contiguous
    assert decoded.metadata["custom"] == "kept" and decoded.metadata["ros2"]["header"] == HEADER
    assert decoded.metadata["spatial"] == payload["spatial"]
    decoded[0, 0] = 100
    assert image[0, 0] == 0
    assert "data_base64" not in json.dumps(packet.describe())


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("color", ["bgr", "rgb", "bgra", "rgba"])
def test_color_order_is_explicit(dtype, color):
    channels = 4 if "a" in color else 3
    image = ImageArray(np.arange(6 * 4 * channels, dtype=dtype).reshape(6, 4, channels), color)
    packet = encode_image(image)
    assert packet.encoding == color + ("8" if dtype == np.uint8 else "16")
    np.testing.assert_array_equal(decode_image(packet), image)
    opposite = {"bgr": "rgb", "rgb": "bgr", "bgra": "rgba", "rgba": "bgra"}[color]
    swapped = encode_image(image, encoding=opposite + ("8" if dtype == np.uint8 else "16"))
    np.testing.assert_array_equal(decode_image(swapped)[..., :3], image[..., [2, 1, 0]])


def test_padded_big_endian_image_and_header_override():
    packet = ROSImage(2, 2, "mono16", 6, 1, b"\x00\x01\x01\x00xx\xff\xff\x00\x02yy", HEADER)
    image = decode_image(ROSImage.from_wire(packet.to_wire()))
    np.testing.assert_array_equal(image, [[1, 256], [65535, 2]])
    assert image.dtype == np.uint16
    original = image.copy()
    encoded = encode_image(image, frame_id="other", stamp_mode="zero")
    assert encoded.header == {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "other"}
    assert metadata(original)["ros2"]["header"] == HEADER
    np.testing.assert_array_equal(decode_image(encoded), image)


@pytest.mark.parametrize(
    "change",
    [
        {"height": True},
        {"width": 0},
        {"step": 1},
        {"data": b"abc"},
        {"encoding": "yuv422"},
        {"is_bigendian": 2},
        {"header": {"stamp": {"sec": 0, "nanosec": -1}, "frame_id": "x"}},
    ],
)
def test_invalid_packets_rejected(change):
    values = dict(
        height=1, width=2, encoding="mono16", step=4, is_bigendian=0, data=b"\0" * 4, header=HEADER
    )
    with pytest.raises(ValueError):
        ROSImage(**{**values, **change})


@pytest.mark.parametrize(
    "image,encoding",
    [
        (ImageArray(np.zeros((2, 3, 3), np.uint8), "hsv"), "auto"),
        (np.zeros((2, 3), np.uint16), "mono8"),
        (np.zeros((2, 3, 2), np.uint8), "auto"),
        (np.zeros((2, 3), np.bool_), "auto"),
    ],
)
def test_no_implicit_depth_or_color_guess(image, encoding):
    with pytest.raises(ValueError):
        encode_image(image, encoding=encoding)


def test_wire_frames_are_bounded_strict_and_can_be_concatenated():
    stream = io.BytesIO()
    write_frame(stream, {"value": 2**60 + 1})
    write_frame(stream, {"value": "second"})
    stream.seek(0)
    assert read_frame(stream) == {"value": 2**60 + 1}
    assert read_frame(stream) == {"value": "second"}
    with pytest.raises(EOFError):
        read_frame(stream)
    for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e309}'):
        with pytest.raises(ValueError):
            parse_json(raw)
    with pytest.raises(ValueError):
        read_frame(io.BytesIO(struct.pack("!I", MAX_FRAME_BYTES + 1)))
    with pytest.raises(ValueError):
        json_bytes({"large": "abcdef"}, 5)


def test_wire_handles_partial_os_reads_and_writes():
    class Partial(io.BytesIO):
        def write(self, value):
            return super().write(value[:3])

        def read(self, count=-1):
            return super().read(min(2, count))

    stream = Partial()
    write_frame(stream, {"data": "complete even with partial pipe operations"})
    stream.seek(0)
    assert read_frame(stream) == {"data": "complete even with partial pipe operations"}


@pytest.mark.parametrize(
    "requested,actual,accepted",
    [
        ({"x": 3}, {"x": 3, "default": 0}, True),
        (3.7, 3, False),
        (True, 1, False),
        ("3", 3, False),
        (2**53 + 1, float(2**53 + 1), False),
        (0.1, float(np.float32(0.1)), True),
        ([1, 2], [1], False),
    ],
)
def test_native_field_setter_cannot_silently_coerce_data(requested, actual, accepted):
    assert preserved(requested, actual) == accepted


class Connection:
    def __init__(self, *args):
        self.sample = None
        self.calls = []
        self.closed = False
        self.error = None
        self.published = True

    def request(self, operation, *args, **kwargs):
        self.calls.append((operation, kwargs))
        if self.error:
            raise self.error
        if operation == "poll":
            value, self.sample = self.sample, None
            return {"sample": value, "info": {"topic": "/test", "receive_sequence": 1}}
        return {"published": self.published, "dds_acknowledged": True}

    def close(self):
        self.closed = True


def test_readers_only_issue_ticket_after_message_and_stop_is_silent(monkeypatch):
    connection = Connection()
    monkeypatch.setattr(ros, "ROSClient", lambda *args: connection)
    reader = ros.ReadROSData(
        parameters={
            "enabled": True,
            "data_mode": "json_string",
            "ticket_mode": "field",
            "ticket_field": "part.id",
        }
    )
    assert reader.execute({}).outputs["data"] is NO_VALUE
    assert reader.state == "waiting" and not reader.current_emissions["ticket"]
    connection.sample = {"data": '{"part":{"id":"A-14"},"ok":true}'}
    result = reader.execute({})
    assert result.outputs["ticket"]["value"] == "A-14"
    assert result.outputs["data"]["ok"] is True
    assert result.payloads["data"]["ros2"]["topic"] == "/test"
    connection.error = InterruptedError()
    assert reader.execute({}).outputs["data"] is NO_VALUE
    assert connection.closed


def test_image_source_records_ros_header_without_claiming_calibration(monkeypatch):
    connection = Connection()
    connection.sample = encode_image(np.ones((3, 5), np.uint16), frame_id="camera").to_wire()
    monkeypatch.setattr(ros, "ROSClient", lambda *args: connection)
    reader = ros.ReadROSImage(parameters={"enabled": True, "ticket_mode": "header_stamp"})
    result = reader.execute({})
    assert result.outputs["ticket"]["value"] == "camera|0:000000000"
    image = result.outputs["image"]
    assert image.metadata["ros2"]["header"]["frame_id"] == "camera"
    assert image.metadata["spatial"]["root_id"] == f"{reader.id}:image"
    assert "calibration" not in image.metadata
    reader.close()


def test_writer_double_gate_validation_unknown_latch_and_reset(monkeypatch):
    connection = Connection()
    opened = []

    def factory(*args):
        opened.append(True)
        return connection

    monkeypatch.setattr(ros, "ROSClient", factory)
    writer = ros.PublishROSData()
    inputs = {"data": {"data": "ok"}, "enable": True}
    assert writer.execute(inputs).outputs["published"] is NO_VALUE
    writer.parameters["write_enabled"] = True
    assert writer.execute({**inputs, "enable": False}).outputs["published"] is NO_VALUE
    assert writer.execute({**inputs, "data": [1, 2]}) is None
    assert not opened
    connection.published = False
    assert writer.execute(inputs).outputs["published"] is NO_VALUE
    connection.error = transport.ROSPublishUnknown("unconfirmed")
    assert writer.execute(inputs) is None
    assert connection.closed and writer._uncertain
    count = len(connection.calls)
    assert writer.execute(inputs) is None and len(connection.calls) == count
    clone = ros.PublishROSData()
    clone.copy_runtime_from(writer)
    assert clone._uncertain
    writer.reset_runtime()
    connection.error = None
    connection.published = True
    assert writer.execute(inputs).outputs["published"] is True
    writer.close()


@pytest.mark.parametrize(
    "settings",
    [
        {"domain_id": 233},
        {"queue_depth": 0},
        {"timeout_seconds": -1},
        {"topic": "bad//name"},
        {"namespace": "/bad-name"},
        {"enabled": "yes"},
        {"message_type": "custom", "custom_message_type": "x"},
        {"data_mode": "json_string", "message_type": "geometry_msgs/msg/Twist"},
        {"ticket_mode": "field"},
        {"ticket_field": "a..b"},
    ],
)
def test_settings_reject_without_runtime(settings):
    with pytest.raises(ValueError):
        ros.ReadROSData(parameters=settings).validate_parameters()


def test_registry_xml_receipts_and_offline_converters(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ros, "ROSClient", lambda *args: pytest.fail("No ROS worker during offline use")
    )
    registry = builtin_registry()
    graph = Graph()
    for cls in ros.ROS2_BLOCKS:
        block = registry.create(cls.type_id)
        block.validate_parameters()
        assert block_entry(block).path == ("Industrial integration", "ROS 2")
    source = graph.add(registry.create("input.demo"))
    encoder = graph.add(registry.create("industrial.ros2_image_encode"))
    decoder = graph.add(registry.create("industrial.ros2_image_decode"))
    graph.connect(source.id, "image", encoder.id, "image")
    graph.connect(encoder.id, "message", decoder.id, "message")
    try:
        graph.run()
        np.testing.assert_array_equal(
            source.last_result.outputs["image"], decoder.last_result.outputs["image"]
        )
        packet = encoder.last_result.outputs["message"]
        assert packet.metadata["spatial"]["root_id"] == source.id + ":image"
        receipt = json.dumps(describe(graph))
        assert "byte_count" in receipt and "data_base64" not in receipt
        path = tmp_path / "codec.xml"
        graph.save(path)
        restored = Graph.load(path, registry)
        try:
            restored.run()
            assert all(b.state == "pass" for b in restored.blocks.values())
        finally:
            restored.close()
    finally:
        graph.close()


def fake_worker(tmp_path, monkeypatch, action):
    script = tmp_path / "fake.py"
    script.write_text(
        "import sys,time\n"
        "from opencv_dataflow.ros2_support.wire import read_frame,write_frame\n"
        "r=read_frame(sys.stdin.buffer)\n"
        "write_frame(sys.stdout.buffer,{'id':r['id'],'ok':True,'result':{'rmw':'fake'}})\n"
        "r=read_frame(sys.stdin.buffer)\n" + action,
        encoding="utf-8",
    )
    popen = transport.subprocess.Popen
    monkeypatch.setattr(
        transport, "runtime_environment", lambda: (sys.executable, __import__("os").environ.copy())
    )
    monkeypatch.setattr(
        transport.subprocess,
        "Popen",
        lambda args, **kwargs: popen([sys.executable, "-u", str(script)], **kwargs),
    )
    return transport.ROSClient({**ros.ROSBlock.defaults}, 5)


@pytest.mark.parametrize(
    "action",
    [
        "time.sleep(30)\n",
        "sys.stdout.buffer.write(b'bad!');sys.stdout.buffer.flush()\n",
        "write_frame(sys.stdout.buffer,{'id':999,'ok':True,'result':{}})\n",
        "write_frame(sys.stdout.buffer,{'id':r['id'],'ok':True,'result':{}})\n",
    ],
)
def test_transport_timeout_corruption_reaps_worker_and_marks_publish_unknown(
    tmp_path, monkeypatch, action
):
    client = fake_worker(tmp_path, monkeypatch, action)
    with pytest.raises(transport.ROSPublishUnknown):
        client.request("publish", 0.15, value={})
    assert client.process.poll() is not None
    assert all(not t.is_alive() for t in client._threads)
    client.close()


def test_remote_validation_rejection_is_known_before_publish(tmp_path, monkeypatch):
    client = fake_worker(
        tmp_path,
        monkeypatch,
        "error={'message':'invalid field'}\n"
        "write_frame(sys.stdout.buffer,{'id':r['id'],'ok':False,'error':error})\n"
        "time.sleep(30)\n",
    )
    try:
        with pytest.raises(transport.ROSRemoteError, match="invalid field"):
            client.request("publish", 1, value={})
        assert not client._closed
    finally:
        client._terminate()


def test_cancelled_read_reaps_worker_without_ambiguous_publish(tmp_path, monkeypatch):
    client = fake_worker(tmp_path, monkeypatch, "time.sleep(30)\n")
    with pytest.raises(InterruptedError):
        client.request("poll", 5, lambda: True, timeout=4)
    assert client.process.poll() is not None
    assert all(not t.is_alive() for t in client._threads)


def test_cancelled_publish_is_ambiguous_and_reaps_worker(tmp_path, monkeypatch):
    client = fake_worker(tmp_path, monkeypatch, "time.sleep(30)\n")
    with pytest.raises(transport.ROSPublishUnknown):
        client.request("publish", 5, lambda: True, value={})
    assert client.process.poll() is not None


def test_runtime_profile_is_local_and_never_names_worker_script(tmp_path, monkeypatch):
    path = tmp_path / "runtime.json"
    path.write_text(
        json.dumps({"python": sys.executable, "ros_distro": "lyrical", "worker": "evil.py"})
    )
    monkeypatch.setenv("OPENCV_DATAFLOW_ROS2_PROFILE", str(path))
    with pytest.raises(ValueError, match="fields"):
        transport.runtime_environment()


def test_optional_registry_import_does_not_import_rclpy():
    code = (
        "from opencv_dataflow.blocks import builtin_registry;import sys;"
        "r=builtin_registry();assert 'input.ros2_image' in r.types;"
        "assert 'rclpy' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30, capture_output=True)
