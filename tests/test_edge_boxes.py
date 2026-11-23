"""Verify ranked EdgeBoxes proposals, scores, and inspection graph integration."""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/edge_boxes"
TRUTH = ((43, 44, 93, 69), (215, 112, 75, 77))
PARAMETERS = {"beta": 0.65, "max_boxes": 12, "min_box_area": 1100}


@pytest.fixture(scope="module")
def registry():
    return builtin_registry()


def iou(rectangle, truth):
    x, y, w, h = rectangle
    tx, ty, tw, th = truth
    iw = max(0, min(x + w, tx + tw) - max(x, tx))
    ih = max(0, min(y + h, ty + th) - max(y, ty))
    return iw * ih / (w * h + tw * th - iw * ih)


def fixture_maps():
    source = cv2.imread(str(EXAMPLE / "assets/parts.png"))
    gray = cv2.GaussianBlur(cv2.cvtColor(source, cv2.COLOR_BGR2GRAY), (5, 5), 1)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    _, angles = cv2.cartToPolar(dx, dy, angleInDegrees=True)
    return source, cv2.Canny(gray, 45, 100), angles


def test_native_controls_scores_and_cap_without_nms(registry):
    _, edge_mask, degrees = fixture_maps()
    strengths = edge_mask.astype(np.float32) / 255
    radians = np.deg2rad(np.remainder(degrees, 180))
    block = registry.create(
        "inspect.edge_boxes",
        parameters={
            "alpha": 0.7,
            "beta": 1.0,
            "eta": 0.95,
            "min_score": 0.03,
            "max_boxes": 3,
            "edge_min_mag": 0.12,
            "edge_merge_threshold": 0.4,
            "cluster_min_mag": 0.65,
            "max_aspect_ratio": 2.7,
            "min_box_area": 900,
            "gamma": 2.3,
            "kappa": 1.4,
        },
    )
    result = block.execute({"edge_map": strengths, "orientation": radians})
    assert result and result.passed, block.error
    native = cv2.ximgproc.createEdgeBoxes(
        0.7, 1.0, 0.95, 0.03, 3, 0.12, 0.4, 0.65, 2.7, 900, 2.3, 1.4
    )
    rectangles, scores = native.getBoundingBoxes(strengths, radians)
    assert len(rectangles) > 3 and result.outputs["count"] == 3
    np.testing.assert_array_equal(
        [item["bounds"] for item in result.outputs["proposals"]], rectangles[:3]
    )
    np.testing.assert_array_equal(
        [item["score"] for item in result.outputs["proposals"]], scores.reshape(-1)[:3]
    )
    assert [item["metrics"]["strength"] for item in result.outputs["collection"]["items"]] == [
        item["score"] for item in result.outputs["proposals"]
    ]


def test_selection_keeps_native_rank_and_source_frame(registry):
    source, strengths, angles = fixture_maps()
    spatial = frame([340, 240], "camera")
    ticket = TicketClock("edge-boxes").issue("camera", "fixed", "parts")
    payload = {"ticket": ticket, "spatial": spatial}
    inputs = {
        "image": ValuePacket(source, payload),
        "edge_map": ValuePacket(strengths, payload),
        "orientation": ValuePacket(angles, {**payload, "angle_units": "degrees"}),
    }
    block = registry.create("inspect.edge_boxes", parameters=PARAMETERS)
    full = block.execute(inputs)
    assert full and full.passed, block.error
    mask = np.zeros(source.shape[:2], np.uint8)
    mask[:, 197:] = 255
    region = {
        "model": "image_region_v1",
        "shape": "rectangle",
        "image_size": [340, 240],
        "rectangle": [190, 98, 117, 92],
    }
    selected = block.execute({**inputs, "mask": mask, "region": region})
    assert selected and selected.passed, block.error
    assert selected.outputs["proposals"][0] == full.outputs["proposals"][1]
    assert selected.outputs["count"] < full.outputs["count"]
    np.testing.assert_array_equal(selected.outputs["image"], source)
    for name in ("image", "collection", "proposals"):
        assert selected.payloads[name]["spatial"] == spatial
        assert selected.payloads[name]["ticket"] == ticket
    assert selected.outputs["collection"]["context"]["spatial"] == spatial
    assert selected.drawings["EdgeBoxes proposals"].source == "image"
    bad = ValuePacket(angles, {"spatial": frame([340, 240], "other")})
    assert block.execute({**inputs, "orientation": bad}) is None
    assert "different frames" in block.error


def test_blank_and_search_budget(registry, monkeypatch):
    def unexpected_create(*args, **kwargs):
        pytest.fail("A blank map or excessive search reached the native detector")

    monkeypatch.setattr(cv2.ximgproc, "createEdgeBoxes", unexpected_create)
    block = registry.create("inspect.edge_boxes")
    blank = np.zeros((16, 24), np.float32)
    result = block.execute({"edge_map": blank, "orientation": blank})
    assert result and result.passed and result.outputs["count"] == 0
    huge = np.ones((1000, 1400), np.float32)
    expensive = registry.create(
        "inspect.edge_boxes", parameters={"alpha": 0.95, "min_box_area": 16}
    )
    assert expensive.execute({"edge_map": huge, "orientation": huge}) is None
    assert "estimated windows" in expensive.error


def test_saved_example_ranks_both_parts(registry):
    graph = Graph.load(EXAMPLE / "edge_boxes.json", registry)
    try:
        graph.run()
        block = graph.blocks["propose"]
        assert block_entry(block).path[:2] == ("OpenCV", "Extended Image Processing (ximgproc)")
        result = block.last_result
        assert result and result.passed, block.error
        proposals = result.outputs["proposals"]
        assert len(proposals) == 12
        assert proposals[0]["score"] > proposals[1]["score"] > proposals[2]["score"]
        assert iou(proposals[0]["bounds"], TRUTH[0]) > 0.9
        assert iou(proposals[1]["bounds"], TRUTH[1]) > 0.85
        assert len(result.drawings["EdgeBoxes proposals"].objects) == 2
        assert graph.blocks["filter"].last_result.outputs["count"] == 2
        assert graph.blocks["draw"].last_result.outputs["count"] == 2
    finally:
        graph.close()


def test_structured_edges_feed_ranked_proposals(registry):
    local_model = EXAMPLE.parent / "structured_edges/.local/models/model.yml.gz"
    model = Path(os.getenv("OPENCV_STRUCTURED_EDGE_MODEL", local_model))
    if not model.is_file():
        pytest.skip("Prepare the companion structured_edges example's reference forest")
    source = cv2.imread(str(EXAMPLE / "assets/parts.png"))
    edges = registry.create("cv.structured_edges", parameters={"model_path": str(model)})
    try:
        maps = edges.execute({"image": source})
        assert maps and maps.passed, edges.error
        block = registry.create("inspect.edge_boxes", parameters=PARAMETERS)
        result = block.execute(
            {
                "edge_map": maps.outputs["thinned"],
                "orientation": maps.outputs["orientation"],
                "image": source,
            }
        )
        assert result and result.passed, block.error
        proposals = result.outputs["proposals"]
        assert len(proposals) == 12
        assert iou(proposals[0]["bounds"], TRUTH[0]) > 0.83
        assert iou(proposals[1]["bounds"], TRUTH[1]) > 0.8
    finally:
        edges.close()
