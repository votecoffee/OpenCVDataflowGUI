"""Check native LBD bytes, endpoint provenance and a known view transform."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/line_descriptors"


@pytest.fixture(scope="module")
def registry():
    return builtin_registry()


def open_query():
    return cv2.imread(str(EXAMPLE / "assets/query.png"), cv2.IMREAD_GRAYSCALE)


def test_native_descriptor_order_and_bytes(registry):
    gray = open_query()
    block = registry.create(
        "inspect.line_descriptors",
        parameters={"band_width": 9, "min_length": 31, "max_features": 12},
    )
    result = block.execute({"image": gray})
    assert result and result.passed, block.error
    native = cv2.line_descriptor.BinaryDescriptor_createBinaryDescriptor()
    native.setWidthOfBand(9)
    keys = sorted(
        (key for key in native.detect(gray) if key.lineLength >= 31),
        key=lambda key: -key.lineLength,
    )[:12]
    keys, expected = native.compute(gray, keys)
    assert len(keys) == result.outputs["count"] == 12
    features = result.outputs["features"]
    assert features["band_width"] == 9
    np.testing.assert_array_equal(features["descriptors"], expected)
    np.testing.assert_array_equal(
        features["segments"], [[key.getStartPoint(), key.getEndPoint()] for key in keys]
    )
    items = result.outputs["collection"]["items"]
    assert [item["geometry"]["points"] for item in items] == features["segments"]


def test_selected_level_geometry_and_match_context(registry):
    gray = open_query()
    root = frame([352, 258], "camera")
    root["root_size"] = [704, 516]
    root["to_root"] = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]
    ticket = TicketClock("line-features").issue("camera", "fixed", "part")
    pyramid = {"level": 1, "largest_size": [704, 516], "to_largest": root["to_root"]}
    payload = {"spatial": root, "ticket": ticket, "pyramid": pyramid}
    region = {
        "model": "image_region_v1",
        "shape": "rectangle",
        "image_size": [352, 258],
        "rectangle": [20, 25, 121, 106],
    }
    mask = np.zeros_like(gray)
    mask[:, :118] = 255
    block = registry.create("inspect.line_descriptors")
    result = block.execute(
        {
            "image": ValuePacket(gray, payload),
            "mask": ValuePacket(mask, {"spatial": root}),
            "region": region,
        }
    )
    assert result and result.passed and result.outputs["count"] > 0, block.error
    segments = np.rint(result.outputs["features"]["segments"]).astype(int)
    assert np.all((20 <= segments[..., 0]) & (segments[..., 0] < 118))
    assert np.all((25 <= segments[..., 1]) & (segments[..., 1] < 131))
    assert {key: result.payloads["features"][key] for key in payload} == payload
    assert result.outputs["collection"]["context"] == {"spatial": root, "pyramid": pyramid}

    features = ValuePacket(result.outputs["features"], payload)
    matcher = registry.create("inspect.match_line_descriptors", parameters={"mode": "cross_check"})
    matched = matcher.execute({"query": features, "train": features})
    assert matched and matched.passed and matched.outputs["count"] > 0, matcher.error
    assert matched.outputs["matches"]["query_context"] == payload
    assert matched.outputs["matches"]["train_context"] == payload


def test_empty_ambiguous_and_incompatible_sets(registry):
    detector = registry.create("inspect.line_descriptors")
    result = detector.execute({"image": np.zeros((48, 56), np.uint8)})
    assert result and result.passed and result.outputs["count"] == 0, detector.error
    empty = result.outputs["features"]
    one = {**empty, "segments": [[[1, 5], [24, 5]]], "descriptors": [[0] * 32]}
    ratio = registry.create("inspect.match_line_descriptors", parameters={"min_matches": 1})
    result = ratio.execute({"query": empty, "train": one})
    assert result and not result.passed and result.outputs["count"] == 0
    assert result.fail_reasons[0]["code"] == "too_few_line_matches"
    result = ratio.execute({"query": one, "train": one})
    assert result and not result.passed and result.outputs["count"] == 0
    incompatible = {**one, "band_width": 9}
    assert ratio.execute({"query": one, "train": incompatible}) is None
    assert "same band width" in ratio.error
    cross_check = registry.create(
        "inspect.match_line_descriptors", parameters={"mode": "cross_check"}
    )
    result = cross_check.execute({"query": one, "train": one})
    assert result and result.outputs["count"] == 1
    assert result.outputs["matches"]["matches"][0]["distance"] == 0


def test_saved_matches_preserve_native_distances_and_known_motion(registry):
    graph = Graph.load(EXAMPLE / "line_descriptors.json", registry)
    try:
        graph.run()
        match = graph.blocks["match"]
        assert block_entry(match).path[:2] == ("OpenCV", "Line Descriptors (line_descriptor)")
        result = match.last_result
        assert result and result.passed, match.error
        assert result.outputs["count"] >= 20
        data = result.outputs["matches"]
        query = graph.blocks["query_lines"].last_result.outputs["features"]
        train = graph.blocks["train_lines"].last_result.outputs["features"]
        warp = cv2.getRotationMatrix2D((176, 129), 3, 1)
        warp[:, 2] += (16, -9)
        for record, source, target in zip(
            data["matches"], data["query_segments"], data["train_segments"], strict=True
        ):
            q, t = record["query_index"], record["train_index"]
            assert source == query["segments"][q] and target == train["segments"][t]
            qd = np.array(query["descriptors"][q], np.uint8)
            td = np.array(train["descriptors"][t], np.uint8)
            assert record["distance"] == np.unpackbits(qd ^ td).sum()
            predicted = np.array(source) @ warp[:, :2].T + warp[:, 2]
            direct = np.linalg.norm(predicted - target, axis=1).max()
            swapped = np.linalg.norm(predicted - target[::-1], axis=1).max()
            assert min(direct, swapped) < 3
        review = graph.blocks["review"].last_result
        assert review.outputs["info"]["pairs"] == result.outputs["count"]
        assert review.outputs["canvas"].shape == (258, 720, 3)
        drawing = review.drawings["Unclassified matches"]
        assert len(drawing.objects) == 3 * result.outputs["count"]
        assert drawing.source == "canvas"
    finally:
        graph.close()
