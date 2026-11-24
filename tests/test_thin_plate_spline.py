"""Verify raster direction, held-out geometry and output frames for spline registration."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

EXAMPLE = (
    Path(__file__).parents[1] / "examples/opencv tools/thin_plate_spline/thin_plate_spline.json"
)
TARGET_POINTS = [[8, 8], [50, 8], [50, 34], [8, 34], [20, 21]]
SOURCE_POINTS = (np.asarray(TARGET_POINTS) - [4, 6]).tolist()


@pytest.fixture(scope="module")
def registry():
    return builtin_registry()


def test_shift_to_larger_frame_preserves_encoding_and_ticket(registry):
    samples = np.arange(50 * 70 * 3, dtype=np.uint16).reshape(50, 70, 3)
    source = ImageArray(samples, "rgb")
    target = np.zeros((54, 74), np.uint8)
    source_frame = frame([70, 50], "source")
    target_frame = frame([74, 54], "target")
    ticket = TicketClock("tps").issue("camera", "fixed", "source")
    other_ticket = TicketClock("tps").issue("reference", "fixed", "target")
    block = registry.create(
        "cv.thin_plate_spline",
        parameters={
            "interpolation": "nearest",
            "border_value": [200, 400, 600],
            "probe_points": [[13, 20], [57, 37]],
            "expected_source_points": [[9, 14], [53, 31]],
        },
    )
    result = block.execute(
        {
            "image": ValuePacket(source, {"spatial": source_frame, "ticket": ticket}),
            "target": ValuePacket(target, {"spatial": target_frame, "ticket": other_ticket}),
            "source_points": ValuePacket(SOURCE_POINTS, {"spatial": source_frame}),
            "target_points": ValuePacket(TARGET_POINTS, {"spatial": target_frame}),
        }
    )
    assert result and result.passed, block.error
    expected = cv2.warpAffine(
        samples,
        np.array([[1, 0, 4], [0, 1, 6]], float),
        (74, 54),
        flags=cv2.INTER_NEAREST,
        borderValue=(200, 400, 600),
    )
    output = result.outputs["image"]
    np.testing.assert_array_equal(output, expected)
    assert output.dtype == np.uint16 and output.color_format == "rgb"
    np.testing.assert_allclose(result.outputs["mapped_points"], [[9, 14], [53, 31]], atol=1e-4)
    assert result.payloads["image"]["spatial"] == target_frame
    assert result.payloads["report"]["spatial"] == target_frame
    assert result.payloads["mapped_points"]["spatial"] == source_frame
    assert result.payloads["image"]["ticket"] == ticket
    assert all(drawing.from_output for drawing in result.drawings.values())


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"source_points": [[4, 2], [4, 2], [46, 28], [4, 28], [16, 15]]}, "distinct"),
        ({"target_points": [[10, 8], [20, 8], [30, 8], [40, 8], [50, 8]]}, "span both axes"),
        ({"target_points": TARGET_POINTS[:4]}, "one target point per"),
        ({"expected_source_points": [[9, 14]]}, "for each target probe"),
        ({"source_points": [[0, 0], [1, 1], [700, 50], [2, 1]]}, "in its image grid"),
    ],
)
def test_invalid_control_geometry_fails_through_executor(registry, updates, message):
    block = registry.create(
        "cv.thin_plate_spline",
        parameters={"source_points": SOURCE_POINTS, "target_points": TARGET_POINTS, **updates},
    )
    assert block.execute({"image": np.zeros((50, 70), np.uint8)}) is None
    assert message in block.error


def test_spatial_mismatch_and_work_budget(registry):
    block = registry.create(
        "cv.thin_plate_spline",
        parameters={"source_points": SOURCE_POINTS, "target_points": TARGET_POINTS},
    )
    image = np.zeros((50, 70), np.uint8)
    assert (
        block.execute(
            {
                "image": ValuePacket(image, {"spatial": frame([70, 50], "source")}),
                "source_points": ValuePacket(SOURCE_POINTS, {"spatial": frame([70, 50], "other")}),
            }
        )
        is None
    )
    assert "source_points must be in the corresponding image grid" in block.error
    many = [[x, y] for x in range(10, 90, 10) for y in range(10, 90, 10)]
    assert (
        block.execute(
            {
                "image": np.zeros((1200, 1200), np.uint8),
                "source_points": many,
                "target_points": many,
            }
        )
        is None
    )
    assert "Spline work limit exceeded" in block.error


def test_saved_warp_improves_analytic_image_and_held_out_landmarks(registry):
    graph = Graph.load(EXAMPLE, registry)
    try:
        graph.run()
        block = graph.blocks["spline"]
        assert block_entry(block).path[:2] == ("OpenCV", "Shape Distance and Matching (shape)")
        result = block.last_result
        assert result and result.passed, block.error
        report = result.outputs["report"]
        assert report["control_count"] == 30
        assert report["probe_count"] == 6
        assert report["control_error"]["max_pixels"] < 0.001
        assert report["probe_error"]["rms_pixels"] < 0.33
        assert report["probe_error"]["max_pixels"] < 0.37
        source = graph.blocks["source"].last_result.outputs["image"]
        target = graph.blocks["target"].last_result.outputs["image"]
        baseline = np.mean(np.abs(np.asarray(source, dtype=float) - target))
        residual = graph.blocks["difference"].last_result.outputs["image"]
        assert np.mean(residual) < 0.1 * baseline
        assert result.payloads["image"]["spatial"]["root_id"] == "target:image"
        assert result.payloads["mapped_points"]["spatial"]["root_id"] == "source:image"
        block.parameters["regularization"] = 0.02
        graph.run()
        smooth = block.last_result
        assert smooth and smooth.passed, block.error
        assert smooth.outputs["report"]["control_error"]["rms_pixels"] > 0.05
    finally:
        graph.close()
