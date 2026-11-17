"""Independent EBML timestamp parsing and native saved-project diagnostic acceptance."""

import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.tickets import ticket_key

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/video_timing"
pytestmark = pytest.mark.local_examples


def elements(data):
    offset = 0
    while offset < len(data):
        width = 8 - data[offset].bit_length() + 1
        tag = int.from_bytes(data[offset : offset + width], "big")
        offset += width
        width = 8 - data[offset].bit_length() + 1
        size = int.from_bytes(data[offset : offset + width], "big") & ((1 << (7 * width)) - 1)
        offset += width
        assert offset + size <= len(data)
        yield tag, data[offset : offset + size]
        offset += size
    assert offset == len(data)


def authored_times(data):
    root = dict(elements(data))
    segment = dict(elements(root[0x18538067]))
    info = dict(elements(segment[0x1549A966]))
    scale = int.from_bytes(info[0x2AD7B1], "big")
    track = dict(elements(dict(elements(segment[0x1654AE6B]))[0xAE]))
    assert track[0x86] == b"V_MJPEG"
    cluster = list(elements(segment[0x1F43B675]))
    origin = int.from_bytes(next(v for k, v in cluster if k == 0xE7), "big")
    result = []
    for key, value in cluster:
        if key == 0xA3:
            assert value[0] == 0x81 and value[3] == 0x80 and value[4:6] == b"\xff\xd8"
            result.append((origin + int.from_bytes(value[1:3], "big", signed=True)) * scale / 1e6)
    return result, track


def test_binary_container_times_are_authored_independently_and_regenerate_exactly(tmp_path):
    truth = json.loads((BASE / "truth.json").read_bytes())
    for name, digest in truth["asset_sha256"].items():
        data = (BASE / "assets" / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
        stamps, track = authored_times(data)
        assert stamps == truth["clocks_ms"][Path(name).stem]
        if name == "zero_time.mkv":
            assert 0x23E383 not in track
        elif name == "wrong_rate.mkv":
            assert int.from_bytes(track[0x23E383], "big") == 100000000
    spec = importlib.util.spec_from_file_location("timing_example", BASE / "build_example.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = tmp_path
    module.build()
    for path in [*BASE.glob("*.json"), *(BASE / "assets").glob("*.mkv")]:
        assert (tmp_path / path.relative_to(BASE)).read_bytes() == path.read_bytes()


class Clock:
    def __init__(self):
        self.value = 100.0

    def now(self):
        return self.value

    def sleep(self, amount):
        self.value += amount


@pytest.mark.parametrize(
    "name",
    [
        "constant",
        "variable",
        "duplicate",
        "backward",
        "zero_time",
        "wrong_rate",
        "wrong_reference",
        "paced_variable",
    ],
)
def test_saved_projects_preserve_each_current_image_and_expose_expected_timing_judgments(name):
    from opencv_dataflow.paced_video import MediaPacer

    truth = json.loads((BASE / "truth.json").read_bytes())
    options = truth["cases"][name]
    graph = Graph.load(BASE / f"{name}.json", builtin_registry())
    source, block = graph.blocks["video"], graph.blocks["timing"]
    clock = Clock()
    if options.get("paced"):
        source._pacer = MediaPacer(clock.now, clock.sleep)
    identifiers = set()
    try:
        for i in range(8):
            graph.run()
            assert all(b.state in ("pass", "fail") for b in graph.processing_blocks()), [
                (b.id, b.state, b.error) for b in graph.processing_blocks()
            ]
            captured = source.last_result
            expected = truth["clocks_ms"][options["clip"]][i]
            assert captured.outputs["info"]["decoder_timestamp_ms"] == pytest.approx(expected)
            observed = block.last_result
            report = observed.outputs["report"]
            assert report["counts"]["frames"] == i + 1
            np.testing.assert_array_equal(observed.outputs["image"], captured.outputs["image"])
            assert observed.payloads["image"]["ticket"] == captured.outputs["ticket"]
            assert report["current"]["decoder_timestamp_ms"] == expected
            assert report["history_rows"] == i + 1
            measured = graph.blocks["blobs"].last_result.outputs
            assert measured["centroids"] == [truth["geometry"][i]["center"]]
            assert measured["data"]["components"][0]["area"] == 672
            assert "Overlaid 2 layers" in graph.blocks["overlay"].last_result.message
            assert graph.blocks["filmstrip"].last_result is not None
            identifiers.add(ticket_key(captured.outputs["ticket"]))
            if options.get("paced"):
                assert clock.now() == pytest.approx(100 + expected / 2000)
        assert len(identifiers) == 8 and source.exhausted
        assert report["counts"]["rejected"] == options["rejected"]
        assert observed.passed is (options["rejected"] == 0)
    finally:
        graph.close()


def test_variable_timing_paced_loop_reset_stride_resume_and_sampling_have_known_pixels():
    from opencv_dataflow.paced_video import MediaPacer

    truth = json.loads((BASE / "truth.json").read_bytes())
    graph = Graph.load(BASE / "paced_variable.json", builtin_registry())
    source = graph.blocks["video"]
    source.configure_replay("loop")
    clock = Clock()
    source._pacer = MediaPacer(clock.now, clock.sleep)
    try:
        for run in range(10):
            graph.run()
            i = run % 8
            report = graph.blocks["timing"].last_result.outputs["report"]
            assert report["ok"] and report["counts"]["frames"] == i + 1
            assert report["current"]["timestamp_ms"] == truth["clocks_ms"]["variable"][i]
    finally:
        graph.close()
    registry = builtin_registry()
    path = BASE / "assets/variable.mkv"
    for start in (1, 3, 5):
        source = registry.create(
            "input.video_file",
            parameters={
                "path": str(path),
                "backend": "ffmpeg",
                "start_frame": start,
                "stride": 2,
            },
        )
        try:
            for i in range(start, 8, 2):
                result = source.execute({})
                assert result is not None, source.error
                assert result.outputs["info"]["timestamp_ms"] == truth["clocks_ms"]["variable"][i]
                gray = cv2.cvtColor(result.outputs["image"], cv2.COLOR_BGR2GRAY)
                yy, xx = np.nonzero(gray > 127)
                np.testing.assert_allclose([xx.mean(), yy.mean()], truth["geometry"][i]["center"])
                source.close()  # Reopen at the retained ordinal on the next activation.
        finally:
            source.close()
    sampler = registry.create(
        "input.video_at_time",
        parameters={
            "path": str(path),
            "seek_method": "scan_from_start",
        },
    )
    for request, expected in ((145, 2), (175, 3), (350, 4), (400, 5)):
        result = sampler.execute({"time_ms": request})
        assert result is not None, sampler.error
        assert result.outputs["info"]["frame_index"] == expected
        assert result.outputs["info"]["timestamp_ms"] == truth["clocks_ms"]["variable"][expected]
