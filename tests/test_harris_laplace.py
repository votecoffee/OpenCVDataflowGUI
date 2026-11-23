"""Verify native scale and response, and repeatability across a known resize."""

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

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/harris_laplace"


@pytest.fixture(scope="module")
def registry():
    return builtin_registry()


@pytest.mark.parametrize("layers", ["2", "4"])
def test_masked_native_scale_response_and_frame(registry, layers):
    gray = cv2.imread(str(EXAMPLE / "assets/corners.png"), cv2.IMREAD_GRAYSCALE)
    wide = np.broadcast_to((gray.astype(np.uint16) * 257)[..., None], (*gray.shape, 3))
    source = ImageArray(wide, "rgb")
    width, height = gray.shape[::-1]
    spatial = frame([width, height], "camera")
    ticket = TicketClock("harris-laplace").issue("camera", "fixed", "scale")
    mask = np.zeros_like(gray)
    mask[:, 100:225] = 255
    region = {
        "model": "image_region_v1",
        "shape": "rectangle",
        "image_size": [width, height],
        "rectangle": [90, 16, 148, 150],
    }
    native_mask = np.zeros_like(gray)
    native_mask[16:166] = mask[16:166]
    native = cv2.xfeatures2d.HarrisLaplaceFeatureDetector_create(5, 0.015, 0.006, 24, int(layers))
    expected = native.detect(gray, native_mask)
    block = registry.create(
        "inspect.harris_laplace",
        parameters={
            "octaves": 5,
            "layers": layers,
            "corner_threshold": 0.015,
            "dog_threshold": 0.006,
            "max_corners": 24,
            "preview_count": 3,
        },
    )
    result = block.execute(
        {
            "image": ValuePacket(source, {"spatial": spatial, "ticket": ticket}),
            "mask": ValuePacket(mask, {"spatial": spatial}),
            "region": region,
        }
    )
    assert result and result.passed, block.error
    assert len(expected) == result.outputs["count"] == 24
    data = result.outputs["keypoints"]
    assert data["image_size"] == [width, height]
    assert data["keypoints"] == [
        {
            "point": list(key.pt),
            "size": key.size,
            "angle": -1.0,
            "response": key.response,
            "octave": key.octave,
        }
        for key in expected
    ]
    np.testing.assert_array_equal(result.outputs["image"], source)
    assert result.outputs["image"].color_format == "rgb"
    assert result.outputs["image"].dtype == np.uint16
    assert result.payloads["keypoints"]["spatial"] == spatial
    assert result.payloads["keypoints"]["ticket"] == ticket
    drawing = result.drawings["Harris–Laplace scale circles"]
    assert len(drawing.objects) == 3
    np.testing.assert_allclose(
        np.linalg.norm(np.subtract(drawing.objects[0].points[0], expected[0].pt)),
        expected[0].size / 2,
    )


def test_blank_and_input_limits(registry):
    block = registry.create("inspect.harris_laplace")
    result = block.execute({"image": np.zeros((23, 31), np.uint8)})
    assert result and result.passed and result.outputs["count"] == 0
    assert result.outputs["keypoints"]["keypoints"] == [] and not result.drawings
    assert block.execute({"image": np.zeros((12, 15), np.uint8)}) is None
    assert "16–4096 pixels per axis" in block.error
    assert block.execute({"image": np.zeros((1024, 1025), np.uint8)}) is None
    assert "1,048,576 pixels" in block.error


def test_saved_graph_repeats_corners_and_scales_across_resize(registry):
    graph = Graph.load(EXAMPLE / "harris_laplace.json", registry)
    try:
        graph.run()
        detector = graph.blocks["query_corners"]
        assert block_entry(detector).path[:2] == ("OpenCV", "Extra 2D Features (xfeatures2d)")
        query = detector.last_result
        train = graph.blocks["train_corners"].last_result
        assert query.outputs["count"] == 205 and train.outputs["count"] == 177
        assert (
            query.payloads["keypoints"]["spatial"]["root_id"]
            == (train.payloads["keypoints"]["spatial"]["root_id"])
        )
        assert train.payloads["keypoints"]["spatial"]["image_size"] == [640, 472]
        data = graph.blocks["evaluate"].last_result
        assert data and data.passed, graph.blocks["evaluate"].error
        assert data.outputs["repeatability"] > 0.9
        assert len(data.outputs["correspondences"]) > 155
        assert data.outputs["metrics"]["operating"]["true_positives"] > 120
        a = graph.blocks["query_descriptor"].last_result.outputs["features"]["keypoints"]
        b = graph.blocks["train_descriptor"].last_result.outputs["features"]["keypoints"]
        ratios = [
            b[pair["train_index"]]["size"] / a[pair["query_index"]]["size"]
            for pair in data.outputs["correspondences"]
        ]
        assert np.median(ratios) == pytest.approx(2.0, abs=0.01)
    finally:
        graph.close()
