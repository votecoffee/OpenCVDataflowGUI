"""Native byte-stream decoding, independent pixels/times, resource and payload contracts."""

import io
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.block_xml import save_block_xml
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import COLOR_FORMATS, PIXEL_DEPTHS, format_image, image_format
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.project_xml import read_block_xml
from opencv_dataflow.tickets import TicketClock, ticket_key
from opencv_dataflow.toolbox import block_entry
from opencv_dataflow.video_buffer import BoundedVideoReader, DecodeVideo

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/video_buffer"
DATA = (BASE / "assets/color-motion.avi").read_bytes()


def run(block, inputs=None):
    result = block.execute({"encoded": DATA} if inputs is None else inputs)
    assert result is not None, block.error
    return result


@pytest.mark.parametrize("color", COLOR_FORMATS)
@pytest.mark.parametrize("depth", PIXEL_DEPTHS)
def test_native_strided_pixels_formats_timestamps_and_no_input_mutation(color, depth):
    data = np.frombuffer(DATA, np.uint8).copy()
    block = DecodeVideo(
        parameters={
            "start_frame": 1,
            "frame_count": 4,
            "stride": 2,
            "preview_frame": 2,
            "color_format": color,
            "pixel_depth": depth,
            "read_chunk_bytes": 257,
        }
    )
    if depth == "uint16" and color in ("hsv", "lab"):
        assert block.execute({"encoded": data}) is None
        assert block.error_kind == "invalid" and not block.current_emissions
        return
    result = run(block, {"encoded": data})
    np.testing.assert_array_equal(data, np.frombuffer(DATA, np.uint8))
    assert result.outputs["info"]["selected_indexes"] == [1, 3, 5, 7]
    assert result.outputs["info"]["reader"]["closed"]
    assert result.outputs["info"]["reader"]["short_reads"] > 0
    for index, packet in zip((1, 3, 5, 7), result.outputs["frames"].items, strict=True):
        expected = cv2.imread(str(BASE / f"assets/frame_{index:02d}.png"))
        expected = format_image(expected, color, depth)
        np.testing.assert_array_equal(packet.value, expected)
        assert image_format(packet.value) == image_format(expected)
        assert packet.payload["video_frame"]["timestamp_ms"] == pytest.approx(index * 100)
        assert packet.payload["video_frame"]["timestamp_source"] == "decoder"
        assert packet.payload["processing_time"]["elapsed_ms"] >= 0
        assert packet.payload["outcome"]["chain_passed"]
    np.testing.assert_array_equal(result.outputs["image"], result.outputs["frames"].items[2].value)
    assert ticket_key(result.payloads["image"]["ticket"]) == ticket_key(
        result.outputs["frames"].items[2].payload["ticket"]
    )


@pytest.mark.parametrize("chunk", [1, 257, 4096])
def test_partial_reads_and_fresh_frame_tickets_preserve_parent_and_replace_pixel_grid(chunk):
    parent = TicketClock().issue("plc", "external", "assembly-42")
    context = {
        "ticket": parent,
        "lot": "A",
        "spatial": frame([64, 48], "old-camera"),
        "pyramid": {"level": 0, "level_count": 1, "largest_size": [64, 48]},
    }
    # Use a valid ordinary spatial payload; pyramid semantics belong to decoded frames.
    context.pop("pyramid")
    block = DecodeVideo(parameters={"frame_count": 8, "read_chunk_bytes": chunk})
    first = run(block, {"encoded": ValuePacket(bytearray(DATA), context)})
    second = run(block, {"encoded": ValuePacket(np.frombuffer(DATA, np.uint8), context)})
    keys = []
    for result in (first, second):
        roots = set()
        for packet in result.outputs["frames"].items:
            keys.append(ticket_key(packet.payload["ticket"]))
            assert packet.payload["video_source_ticket"] == parent
            assert packet.payload["lot"] == "A" and "pyramid" not in packet.payload
            assert packet.payload["spatial"] != context["spatial"]
            roots.add(packet.payload["spatial"]["root_id"])
        assert len(roots) == 1
    assert len(set(keys)) == 16


@pytest.mark.parametrize(
    "parameters",
    [
        {"frame_count": 65},
        {"start_frame": -1},
        {"read_chunk_bytes": 0},
        {"maximum_scan_frames": 4},
        {"preview_frame": 16},
        {"stride": 1.0},
        {"orientation_auto": "true"},
        {"backend": "auto"},
        {"path": []},
        {"timestamp_policy": "guess"},
        {"short_clip": "truncate"},
    ],
)
def test_invalid_static_controls_never_construct_native_capture(monkeypatch, parameters):
    def forbidden():
        raise AssertionError("Native capture must not be constructed")

    monkeypatch.setattr(cv2, "VideoCapture", forbidden)
    block = DecodeVideo(parameters=parameters)
    assert block.execute({"encoded": DATA}) is None
    assert block.error_kind == "invalid" and not block.current_emissions


@pytest.mark.parametrize(
    "value",
    [
        b"",
        b"not video",
        np.zeros((2, 2), np.uint8),
        np.zeros(4, np.float32),
        "http://unused.invalid",
        None,
    ],
)
def test_invalid_encoded_input_is_contained(value):
    block = DecodeVideo()
    assert block.execute({"encoded": value}) is None and not block.current_emissions


@pytest.mark.parametrize(
    "parameters",
    [
        {"maximum_encoded_bytes": 1},
        {"maximum_frame_pixels": 1},
        {"maximum_decoded_bytes": 1},
        {"maximum_io_calls": 1},
        {"maximum_read_bytes": 1},
        {"frame_count": 16, "short_clip": "error"},
        {"preview_frame": 15},
        {"start_frame": 50, "frame_count": 1},
    ],
)
def test_runtime_budget_or_missing_frame_rejects_all_outputs(parameters):
    block = DecodeVideo(parameters=parameters)
    assert block.execute({"encoded": DATA}) is None
    assert block.error_kind == "invalid" and not block.current_emissions


def test_file_snapshot_save_as_xml_choices_and_short_clip_report(app, tmp_path):
    from PySide6.QtWidgets import QComboBox

    from opencv_dataflow.app import MainWindow

    registry = builtin_registry()
    limited = DecodeVideo(
        parameters={"path": str(BASE / "assets/color-motion.avi"), "maximum_encoded_bytes": 1}
    )
    assert limited.execute({}) is None and limited.error_kind == "invalid"
    graph = Graph()
    block = graph.add(
        registry.create(
            "cv.decode_video",
            parameters={"path": "assets/color-motion.avi", "path_base": "project"},
        )
    )
    graph.project_dir = BASE
    assert block_entry(block).path[:2] == ("OpenCV", "Video I/O (videoio)")
    try:
        graph.save(tmp_path / "project.json")
        reopened = Graph.load(tmp_path / "project.json", registry)
        try:
            reopened.run()
            result = reopened.blocks[block.id].last_result
            assert result is not None, reopened.blocks[block.id].error
            assert len(result.outputs["frames"].items) == 8
            assert result.outputs["info"]["stop_reason"] == "native_end_or_decode_failure"
            assert not result.outputs["info"]["selection_satisfied"]
        finally:
            reopened.close()
        path = tmp_path / "decode.cv.decode_video.xml"
        save_block_xml(block, path)
        assert read_block_xml(path)["parameters"] == block.parameters
        window = MainWindow()
        try:
            window.replace_graph(graph)
            app.processEvents()
            window.inspector.select(block)
            for name in block.parameter_choices:
                assert isinstance(window.inspector.fields[name], QComboBox)
        finally:
            window.dirty = False
            window.close()
            app.processEvents()
    finally:
        graph.close()


@pytest.mark.parametrize("shape", [(1, -1), (-1, 1)])
def test_raw_vector_layout_and_encoded_precedence_over_unused_file_path(shape):
    block = DecodeVideo(
        parameters={"path": "missing-video-must-not-be-opened.avi", "frame_count": 1}
    )
    result = run(block, {"encoded": np.frombuffer(DATA, np.uint8).reshape(shape)})
    np.testing.assert_array_equal(
        result.outputs["image"], cv2.imread(str(BASE / "assets/frame_00.png"))
    )


def test_native_reader_closed_eof_out_of_range_and_callback_exceptions():
    reader = BoundedVideoReader(b"abcdef", 2, 6, 20)
    assert reader.read(4) == b"ab" and reader.short_reads == 1
    assert reader.seek(-2, io.SEEK_END) == 4
    assert reader.read(5) == b"ef" and reader.read(1) == b""
    assert reader.seek(7) == -1 and reader.failure
    reader.close()
    assert reader.read(2) == b"" and reader.seek(0) == -1 and not reader.data

    def broken():
        raise RuntimeError("Controlled cancellation callback error")

    reader = BoundedVideoReader(b"abc", 2, 6, 20, broken)
    assert reader.read(2) == b"" and "callback failed" in reader.failure
    reader.close()


@pytest.mark.parametrize(
    "failure",
    ["open", "read", "header", "shape", "orientation", "pixel_limit", "read_budget", "cancel"],
)
def test_every_native_exit_releases_capture_and_reader_without_partial_publication(
    monkeypatch, failure
):
    original = cv2.VideoCapture
    observed = []

    class RecordingCapture:
        def __init__(self):
            self.native = original()
            self.reader = None
            self.released = False
            self.read_calls = 0
            observed.append(self)

        def __getattr__(self, name):
            return getattr(self.native, name)

        def open(self, reader, backend, options):
            self.reader = reader
            assert backend == cv2.CAP_FFMPEG
            assert dict(zip(options[::2], options[1::2])) == {
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC: 5000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC: 5000,
                cv2.CAP_PROP_N_THREADS: 1,
            }
            result = self.native.open(reader, backend, options)
            if failure == "open":
                raise cv2.error("Controlled native open exception after allocation")
            return result

        def get(self, key):
            return (
                float("nan")
                if failure == "header" and key == cv2.CAP_PROP_FRAME_WIDTH
                else self.native.get(key)
            )

        def set(self, key, value):
            return False if failure == "orientation" else self.native.set(key, value)

        def read(self):
            self.read_calls += 1
            if failure == "read" and self.read_calls == 2:
                raise cv2.error("Controlled native read exception after one selected frame")
            ok, value = self.native.read()
            return ok, value[:, :-1] if ok and failure == "shape" else value

        def release(self):
            self.native.release()
            self.released = True

    monkeypatch.setattr(cv2, "VideoCapture", RecordingCapture)
    extra = (
        {"maximum_frame_pixels": 1}
        if failure == "pixel_limit"
        else {"maximum_read_bytes": 1}
        if failure == "read_budget"
        else {}
    )
    block = DecodeVideo(parameters={"frame_count": 8, **extra})
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return failure == "cancel" and calls > 8

    assert block.execute({"encoded": DATA}, cancelled=cancelled) is None
    assert not block.current_emissions and block.state == "error"
    assert len(observed) == 1 and observed[0].released
    assert not observed[0].native.isOpened()
    assert observed[0].reader.closed and observed[0].reader.data == b""
    if failure in ("pixel_limit", "header", "orientation"):
        assert not observed[0].read_calls


@pytest.mark.parametrize(
    "policy,expected",
    [("decoder_or_fps", 200.0), ("decoder_only", "error"), ("allow_missing", None)],
)
def test_missing_presentation_time_has_explicit_policy(policy, expected):
    from opencv_dataflow.parameter_validation import ValidationError
    from opencv_dataflow.video_buffer import timestamp

    class MissingTime:
        def get(self, key):
            return 0.0

    if expected == "error":
        with pytest.raises(ValidationError, match="timestamp unavailable"):
            timestamp(MissingTime(), 2, 10, policy)
    else:
        value, origin = timestamp(MissingTime(), 2, 10, policy)
        assert value == expected
        assert origin == ("unavailable" if expected is None else "frame_rate_estimate")
