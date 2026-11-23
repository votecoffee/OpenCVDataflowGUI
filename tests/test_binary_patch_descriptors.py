"""Verify native binary descriptors, matching layouts, and transformed-image use."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/binary_patch_descriptors"


@pytest.fixture(scope="module")
def registry():
    return builtin_registry()


@pytest.mark.parametrize(
    ("family", "width"),
    [("LATCH", 4), ("LATCH", 64), ("BEBLID", 32), ("BEBLID", 64), ("TEBLID", 32), ("TEBLID", 64)],
)
def test_native_parity_and_source_rows(registry, family, width):
    image = np.random.default_rng(912).integers(0, 256, (133, 157), np.uint8)
    original = image.copy()
    data = [
        {"point": [10, 11], "size": 24, "angle": 19},
        {"point": [74, 63], "size": 31, "angle": 43},
        {"point": [96, 79], "size": 28, "angle": 255},
    ]
    cv = cv2.xfeatures2d
    if family == "LATCH":
        settings = {"descriptor_bytes": str(width), "half_ssd_size": 4, "sigma": 0.8}
        native = cv.LATCH_create(width, True, 4, 0.8)
    else:
        settings = {"descriptor_bits": str(width * 8), "scale_factor": 1.25}
        enum = getattr(cv, f"{family}_SIZE_{width * 8}_BITS")
        native = getattr(cv, f"{family}_create")(1.25, enum)
    keys = [
        cv2.KeyPoint(*row["point"], row["size"], row["angle"], 0, 0, index)
        for index, row in enumerate(data)
    ]
    expected_keys, expected = native.compute(image, keys)
    spatial = frame([157, 133], "camera")
    ticket = TicketClock("binary-patches").issue("camera", "fixed", family)
    context = {"spatial": spatial, "ticket": ticket}
    block = registry.create(f"inspect.{family.lower()}_descriptor", parameters=settings)
    result = block.execute(
        {
            "image": ValuePacket(image, context),
            "keypoints": ValuePacket(data, context),
        }
    )
    assert result and result.passed, block.error
    features = result.outputs["features"]
    assert features["algorithm"] == family and features["descriptor_size"] == width
    assert [key["source_index"] for key in features["keypoints"]] == [
        key.class_id for key in expected_keys
    ]
    assert result.outputs["count"] == (2 if family == "LATCH" else 3)
    np.testing.assert_array_equal(features["descriptors"], expected)
    np.testing.assert_array_equal(result.outputs["image"], original)
    assert result.payloads["features"]["spatial"] == spatial
    assert result.payloads["features"]["ticket"] == ticket


def test_oriented_latch_and_incompatible_hamming_layouts(registry):
    image = np.random.default_rng(22).integers(0, 256, (81, 83), np.uint8)
    inputs = {"image": image, "keypoints": [[40, 41]]}
    latch = registry.create("inspect.latch_descriptor")
    assert latch.execute(inputs) is None
    assert "measured angle" in latch.error
    latch = registry.create("inspect.latch_descriptor", parameters={"use_orientation": False})
    first = latch.execute(inputs).outputs["features"]
    beblid = registry.create("inspect.beblid_descriptor", parameters={"descriptor_bits": "256"})
    boosted = beblid.execute(inputs).outputs["features"]
    matcher = registry.create("inspect.match_descriptor_radius", parameters={"max_distance": 0})
    assert matcher.execute({"query": first, "train": first}).outputs["count"] == 1
    assert matcher.execute({"query": first, "train": boosted}) is None
    assert "same descriptor family and settings" in matcher.error
    different = {**boosted, "descriptor_settings": {"scale_factor": 2.0}}
    assert matcher.execute({"query": boosted, "train": different}) is None
    assert "same descriptor family and settings" in matcher.error


def test_saved_comparison_measures_repeatability_and_hamming_distances(registry):
    graph = Graph.load(EXAMPLE / "binary_patch_descriptors.json", registry)
    try:
        graph.run()
        for family, minimum, precision in (
            ("latch", 90, 0.84),
            ("beblid", 110, 0.98),
            ("teblid", 165, 0.94),
        ):
            descriptor = graph.blocks[f"{family}_query"]
            assert block_entry(descriptor).path[:2] == ("OpenCV", "Extra 2D Features (xfeatures2d)")
            query = descriptor.last_result.outputs["features"]
            train = graph.blocks[f"{family}_train"].last_result.outputs["features"]
            result = graph.blocks[f"{family}_evaluate"].last_result
            assert result and result.passed, graph.blocks[f"{family}_evaluate"].error
            assert result.outputs["repeatability"] > 0.68
            assert result.outputs["precision"] >= precision
            matches = graph.blocks[f"{family}_match"].last_result.outputs["matches"]["matches"]
            assert len(matches) >= minimum
            for match in matches:
                q = np.array(query["descriptors"][match["query_index"]], np.uint8)
                t = np.array(train["descriptors"][match["train_index"]], np.uint8)
                assert match["distance"] == np.unpackbits(q ^ t).sum() <= 48
    finally:
        graph.close()
