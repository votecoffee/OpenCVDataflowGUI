"""Verify native accumulator parity and analytic source-line geometry."""

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/fast_hough_lines"
HORIZONTAL = ((0, 48), (147, 48))
VERTICAL = ((36, 0), (36, 103))
SHALLOW_DOWN = ((0, 25), (147, 82))
SHALLOW_UP = ((0, 78), (147, 21))
STEEP_DOWN = ((42, 0), (75, 103))
STEEP_UP = ((120, 0), (54, 103))


@pytest.fixture(scope="module")
def registry():
    return builtin_registry()


@pytest.mark.parametrize("layout", ["deskewed", "raw"])
@pytest.mark.parametrize(
    ("angle", "native_angle", "line", "rows"),
    [
        ("full", "ARO_315_135", SHALLOW_DOWN, 501),
        ("horizontal", "ARO_45_135", SHALLOW_DOWN, 295),
        ("vertical", "ARO_315_45", STEEP_DOWN, 207),
        ("near_horizontal", "ARO_CTR_HOR", SHALLOW_DOWN, 148),
        ("near_vertical", "ARO_CTR_VER", STEEP_DOWN, 104),
        ("vertical_down_right", "ARO_315_0", STEEP_DOWN, 104),
        ("vertical_down_left", "ARO_0_45", STEEP_UP, 104),
        ("horizontal_up_right", "ARO_45_90", SHALLOW_UP, 148),
        ("horizontal_down_right", "ARO_90_135", SHALLOW_DOWN, 148),
    ],
)
def test_accumulator_and_line_geometry(registry, angle, native_angle, line, rows, layout):
    source = np.zeros((104, 148), np.uint8)
    cv2.line(source, *line, 255, 1)
    block = registry.create(
        "inspect.fast_hough_lines",
        parameters={
            "angle_range": angle,
            "layout": layout,
            "maximum_lines": 1,
        },
    )
    result = block.execute({"image": source})
    assert result and result.passed, block.error
    native = cv2.ximgproc
    expected = native.FastHoughTransform(
        source,
        cv2.CV_32F,
        angleRange=getattr(native, native_angle),
        makeSkew=native.HDO_RAW if layout == "raw" else native.HDO_DESKEW,
    )
    accumulator = result.outputs["accumulator"]
    assert accumulator.shape == (rows, 252) and accumulator.dtype == np.float32
    np.testing.assert_array_equal(accumulator, expected)
    (peak,) = result.outputs["peaks"]
    x, y = peak["bin"]
    assert peak["response"] == accumulator[y, x] == accumulator.max()
    assert sorted(map(tuple, peak["points"])) == sorted(line)
    angle_radians = math.radians(peak["normal_angle_degrees"])
    normal = (math.cos(angle_radians), math.sin(angle_radians))
    np.testing.assert_allclose(np.array(line) @ normal, peak["rho_pixels"], atol=1e-10)


@pytest.mark.parametrize(
    ("operation", "native_op", "depth", "native_depth"),
    [
        ("average", "FHT_AVE", "float64", cv2.CV_64F),
        ("minimum", "FHT_MIN", "float32", cv2.CV_32F),
        ("maximum", "FHT_MAX", "int32", cv2.CV_32S),
    ],
)
def test_accumulation_operations(registry, operation, native_op, depth, native_depth):
    image = np.arange(9 * 17, dtype=np.uint8).reshape(9, 17)
    block = registry.create(
        "inspect.fast_hough_lines",
        parameters={
            "operation": operation,
            "accumulator_depth": depth,
            "minimum_lines": 0,
        },
    )
    result = block.execute({"image": image})
    assert result and result.passed, block.error
    expected = cv2.ximgproc.FastHoughTransform(
        image, native_depth, op=getattr(cv2.ximgproc, native_op)
    )
    accumulator = result.outputs["accumulator"]
    assert accumulator.dtype == np.dtype(depth)
    np.testing.assert_array_equal(accumulator, expected)


def test_selected_edge_votes_keep_ticket_and_separate_bin_frame(registry):
    image = np.zeros((104, 148), np.uint8)
    cv2.line(image, *HORIZONTAL, 255, 1)
    cv2.line(image, *VERTICAL, 255, 1)
    mask = np.zeros_like(image)
    mask[:, 30:42] = 255
    region = {
        "model": "image_region_v1",
        "shape": "rectangle",
        "image_size": [148, 104],
        "rectangle": [22, 12, 31, 76],
    }
    spatial = frame([148, 104], "camera")
    ticket = TicketClock("fast-hough").issue("camera", "fixed", "edge")
    block = registry.create("inspect.fast_hough_lines", parameters={"minimum_response": 18_000})
    result = block.execute(
        {
            "image": ValuePacket(image, {"ticket": ticket, "spatial": spatial}),
            "mask": ValuePacket(mask, {"spatial": spatial}),
            "region": region,
        }
    )
    assert result and result.passed, block.error
    (peak,) = result.outputs["peaks"]
    assert sorted(map(tuple, peak["points"])) == sorted(VERTICAL)
    assert peak["response"] == 76 * 255
    np.testing.assert_array_equal(result.outputs["image"], image)
    for name in ("image", "peaks", "collection"):
        assert result.payloads[name]["spatial"] == spatial
        assert result.payloads[name]["ticket"] == ticket
    accumulator_payload = result.payloads["accumulator"]
    assert accumulator_payload["ticket"] == ticket
    assert accumulator_payload["spatial"]["root_id"] != spatial["root_id"]
    assert accumulator_payload["spatial"]["image_size"] == [252, 501]
    assert accumulator_payload["fast_hough"]["source_size"] == [148, 104]
    assert result.outputs["collection"]["context"]["spatial"] == spatial
    assert result.drawings["Fast Hough lines"].source == "image"


def test_blank_and_accumulator_limit(registry):
    block = registry.create("inspect.fast_hough_lines")
    blank = block.execute({"image": np.zeros((25, 43), np.uint8)})
    assert blank and not blank.passed
    assert blank.fail_reasons[0]["code"] == "fast_hough_lines_below_minimum"
    assert blank.outputs["count"] == 0 and not blank.outputs["accumulator"].any()
    assert block.execute({"image": np.zeros((1080, 1920), np.uint8)}) is None
    assert "exceeds 8,388,608 cells" in block.error


def test_saved_example_with_ranked_lines_and_accumulator_display(registry):
    graph = Graph.load(EXAMPLE / "fast_hough_lines.json", registry)
    try:
        graph.run()
        detect = graph.blocks["detect"]
        result = detect.last_result
        assert result and result.passed, detect.error
        assert block_entry(detect).path[:2] == ("OpenCV", "Extended Image Processing (ximgproc)")
        assert len(result.drawings["Fast Hough lines"].objects) == 3
        for peak, line in zip(
            result.outputs["peaks"], [HORIZONTAL, SHALLOW_DOWN, VERTICAL], strict=True
        ):
            assert sorted(map(tuple, peak["points"])) == sorted(line)
        display = graph.blocks["display"].last_result.outputs["image"]
        assert display.shape == (501, 252) and display.dtype == np.uint8
        assert display.min() == 0 and display.max() == 255
        assert graph.blocks["vertical"].last_result.outputs["count"] == 1
        overlay_mask = graph.blocks["draw"].last_result.outputs["mask"]
        assert overlay_mask[70, 36] and not overlay_mask[48, 130]
    finally:
        graph.close()
