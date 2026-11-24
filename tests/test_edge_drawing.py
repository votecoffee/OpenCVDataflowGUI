"""Known primitive geometry, native path parity and bounded Edge Drawing execution."""

from pathlib import Path

import cv2
import numpy as np
import pytest

import opencv_dataflow.edge_drawing as edge_drawing
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

TYPE = "inspect.edge_drawing"
EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/edge_drawing"


@pytest.fixture(scope="module")
def registry():
    return builtin_registry()


@pytest.fixture(scope="module")
def plate():
    return cv2.imread(str(EXAMPLE / "assets/plate.png"))


@pytest.mark.parametrize(
    ("gradient", "pf", "color"),
    [
        ("prewitt", False, False), ("sobel", False, False), ("scharr", False, False),
        ("lsd", False, False), ("prewitt", True, False), ("prewitt", False, True),
    ],
)
def test_native_maps_path_order_and_line_links(registry, plate, gradient, pf, color):
    native = cv2.ximgproc.createEdgeDrawing()
    params = cv2.ximgproc.EdgeDrawing.Params()
    params.EdgeDetectionOperator = getattr(cv2.ximgproc, f"EdgeDrawing_{gradient.upper()}")
    params.PFmode = pf
    native.setParams(params)
    native.detectEdges(plate if color else cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY))
    expected_paths = native.getSegments()
    expected_edges, expected_gradient = native.getEdgeImage(), native.getGradientImage()
    expected_lines = native.detectLines()
    expected_indices = native.getSegmentIndicesOfLines()
    block = registry.create(
        TYPE,
        parameters={
            "gradient": gradient, "parameter_free": pf, "input_mode": "color" if color else "grayscale",
        },
    )
    threads = cv2.getNumThreads()
    result = block.execute({"image": plate})
    assert result and result.passed, block.error
    assert cv2.getNumThreads() == threads
    outputs = result.outputs
    np.testing.assert_array_equal(outputs["image"], plate)
    np.testing.assert_array_equal(outputs["edges"], expected_edges)
    np.testing.assert_array_equal(outputs["gradient"], expected_gradient)
    assert outputs["gradient"].dtype == np.uint16
    assert len(expected_paths) == len(outputs["segments"]) > 0
    for index, (actual, expected) in enumerate(zip(outputs["segments"], expected_paths, strict=True)):
        assert actual["id"] == index
        np.testing.assert_array_equal(actual["points"], np.asarray(expected).reshape(-1, 2))
    for actual, expected, index in zip(outputs["lines"]["items"], expected_lines, expected_indices, strict=True):
        assert actual["geometry"]["edge_segment_id"] == index
        np.testing.assert_allclose(actual["geometry"]["points"], expected.reshape(2, 2), atol=1e-5)


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_example_fits_known_circle_ellipse_and_long_lines(registry, profile):
    graph = Graph.load(EXAMPLE / "edge_drawing.json", registry)
    graph.set_run_profile(profile)
    try:
        graph.run()
        block = graph.blocks["edge"]
        assert block_entry(block).path[:2] == ("OpenCV", "Extended Image Processing (ximgproc)")
        assert all(node.state == "pass" for node in graph.processing_blocks()), block.error
        result = block.last_result
        assert result.outputs["report"]["segments"] == 3
        assert [len(path["points"]) for path in result.outputs["segments"]] == [240, 196, 340]
        assert result.outputs["circles"]["count"] == result.outputs["ellipses"]["count"] == 1
        circle = result.outputs["circles"]["items"][0]["geometry"]
        assert circle["center"] == pytest.approx((85, 154), abs=0.5)
        assert circle["radius"] == pytest.approx(34.61, abs=0.2)
        ellipse = result.outputs["ellipses"]["items"][0]["geometry"]
        assert ellipse["center"] == pytest.approx((270, 115), abs=0.5)
        assert ellipse["axis_lengths"] == pytest.approx((107.31, 61.31), abs=0.2)
        assert ellipse["angle_degrees"] == pytest.approx(27.22, abs=0.3)
        long_lines = graph.blocks["long"].last_result.outputs["collection"]["items"]
        assert len(long_lines) == 2
        assert sorted(line["metrics"]["center_y"] for line in long_lines) == pytest.approx((27, 62), abs=0.15)
        assert all(abs(line["metrics"]["length"] - 136) < 1 for line in long_lines)
        assert {line["geometry"]["edge_segment_id"] for line in long_lines} == {2}
        assert bool(result.drawings) == (profile == "debug")
        if profile == "debug":
            assert set(result.drawings) == {"edge paths", "fitted lines", "circles", "ellipses"}
    finally:
        graph.close()


def test_selection_keeps_path_ids_and_raster_and_ticket_alignment(registry, plate):
    spatial = frame([360, 240], "camera")
    ticket = TicketClock("edge").issue("camera", "fixed", "plate")
    mask = np.zeros(plate.shape[:2], np.uint8)
    mask[60:175, 212:] = 255
    area = {
        "model": "image_region_v1", "shape": "rectangle",
        "image_size": [360, 240], "rectangle": [205, 60, 100, 115],
    }
    block = registry.create(TYPE)
    result = block.execute(
        {
            "image": ValuePacket(plate, {"spatial": spatial, "ticket": ticket}),
            "mask": ValuePacket(mask, {"spatial": spatial}),
            "region": area,
        }
    )
    assert result and result.passed, block.error
    outputs = result.outputs
    [path] = outputs["segments"]
    assert path["id"] == 0 and max(x for x, _ in path["points"]) > 305
    assert outputs["lines"]["count"] > 0
    for line in outputs["lines"]["items"]:
        assert line["geometry"]["edge_segment_id"] == path["id"]
        assert all(212 <= round(point[0]) < 305 for point in line["geometry"]["points"])
    assert outputs["circles"]["count"] == 0
    assert outputs["ellipses"]["count"] == 1
    assert not outputs["edges"][:, 305:].any()
    assert not outputs["gradient"][:, :212].any()
    assert outputs["image"][100, 320].tolist() == plate[100, 320].tolist()
    for port in ("edges", "gradient", "segments", "lines", "ellipses"):
        assert result.payloads[port]["ticket"] == ticket
        assert result.payloads[port]["spatial"] == spatial


def test_flat_and_path_only_modes_are_valid(registry, plate):
    block = registry.create(TYPE)
    flat = np.full((64, 80), 80, np.uint8)
    result = block.execute({"image": flat})
    assert result and result.passed, block.error
    assert result.outputs["segments"] == [] and not result.outputs["edges"].any()

    block = registry.create(TYPE, parameters={"fit_round": False, "fit_lines": False})
    source16 = plate.astype(np.uint16) * 257
    result = block.execute({"image": source16})
    assert result and result.passed, block.error
    outputs = result.outputs
    assert len(outputs["segments"]) == 3
    assert all(outputs[name]["count"] == 0 for name in ("lines", "circles", "ellipses"))
    np.testing.assert_array_equal(outputs["image"], source16)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"scan_interval": 0}, "Scan interval"),
        ({"minimum_line_length": 4}, "Minimum line length"),
        ({"input_mode": "color", "parameter_free": True}, "requires grayscale"),
    ],
)
def test_invalid_settings_are_rejected_before_the_worker(registry, plate, monkeypatch, params, message):
    def unexpected(*args, **kwargs):
        raise AssertionError("Invalid settings reached OpenCV")

    monkeypatch.setattr(edge_drawing, "ManagedProcess", unexpected)
    block = registry.create(TYPE, parameters=params)
    assert block.execute({"image": plate}) is None
    assert message in block.error


@pytest.mark.parametrize(
    ("body", "timeout", "message"),
    [("raise SystemExit(17)", 10, "worker failed (exit 17)"), ("import time; time.sleep(10)", 0.2, "timed out")],
)
def test_worker_failure_and_timeout_are_block_errors(registry, plate, monkeypatch, tmp_path, body, timeout, message):
    script = tmp_path / "worker_error.py"
    script.write_text(body, encoding="utf-8")
    monkeypatch.setattr(edge_drawing, "WORKER_SCRIPT", script)
    block = registry.create(TYPE, parameters={"timeout_seconds": timeout})
    assert block.execute({"image": plate}) is None
    assert block.state == "error" and message in block.error and not block.current_emissions
