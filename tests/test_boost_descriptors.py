"""Verify native BoostDesc layouts and known-pair matching through the built-in registry."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.registration import radius_data
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/boost_descriptors"
MODES = (
    ("BGM", 100, 32, np.uint8, cv2.NORM_HAMMING),
    ("BGM_HARD", 101, 16, np.uint8, cv2.NORM_HAMMING),
    ("BGM_BILINEAR", 102, 32, np.uint8, cv2.NORM_HAMMING),
    ("LBGM", 200, 64, np.float32, cv2.NORM_L2),
    ("BINBOOST_64", 300, 8, np.uint8, cv2.NORM_HAMMING),
    ("BINBOOST_128", 301, 16, np.uint8, cv2.NORM_HAMMING),
    ("BINBOOST_256", 302, 32, np.uint8, cv2.NORM_HAMMING),
)


@pytest.fixture(scope="module")
def registry():
    return builtin_registry()


@pytest.fixture
def texture():
    return np.random.default_rng(741).integers(0, 256, (126, 153), dtype=np.uint8)


@pytest.mark.parametrize(("variant", "code", "width", "dtype", "norm"), MODES)
def test_all_variants_match_native_and_keep_selected_source_indices(
    registry, texture, variant, code, width, dtype, norm,
):
    spatial = frame([153, 126], "camera")
    ticket = TicketClock("boostdesc").issue("camera", "fixed", variant)
    context = {"spatial": spatial, "ticket": ticket}
    rows = [
        {"point": [15, 14], "size": 10, "angle": 25},
        {"point": [67, 60], "size": 8, "angle": 68},
        {"point": [106, 89], "size": 12, "angle": 311},
    ]
    mask = np.zeros(texture.shape, np.uint8)
    mask[16:120, 18:145] = 255
    region = {
        "model": "image_region_v1", "image_size": [153, 126],
        "shape": "rectangle", "rectangle": [45, 35, 75, 70],
    }
    native = cv2.xfeatures2d.BoostDesc.create(code, True, 2.75)
    keys = [
        cv2.KeyPoint(*row["point"], row["size"], row["angle"], 0, 0, index)
        for index, row in enumerate(rows) if index in (1, 2)
    ]
    _, expected = native.compute(texture, keys)
    block = registry.create(
        "inspect.boost_descriptor", parameters={"variant": variant, "scale_factor": 2.75}
    )
    result = block.execute({
        "image": ValuePacket(texture, context), "keypoints": ValuePacket(rows, context),
        "mask": mask, "region": region,
    })
    assert result and result.passed, block.error
    features = result.outputs["features"]
    assert features["descriptor_size"] == width
    assert [point["source_index"] for point in features["keypoints"]] == [1, 2]
    descriptors, _, _, actual_norm = radius_data(features)
    assert descriptors.dtype == dtype and actual_norm == norm
    np.testing.assert_array_equal(descriptors, expected)
    np.testing.assert_array_equal(result.outputs["image"], texture)
    assert result.payloads["features"]["spatial"] == spatial
    assert result.payloads["features"]["ticket"] == ticket


def test_empty_sets_and_incompatible_patterns(registry, texture):
    block = registry.create("inspect.boost_descriptor", parameters={"use_orientation": False})
    emptied = block.execute({"image": texture, "keypoints": []})
    descriptors, _, _, norm = radius_data(emptied.outputs["features"])
    assert descriptors.shape == (0, 32) and norm == cv2.NORM_HAMMING
    settings = {"image": texture, "keypoints": [[45, 45], [83, 92]]}
    first = block.execute(settings).outputs["features"]
    matcher = registry.create("inspect.match_descriptor_radius", parameters={"max_distance": 0})
    assert matcher.execute({"query": first, "train": first}).outputs["count"] == 2
    for parameters in (
        {"variant": "BGM"},
        {"variant": "LBGM"},
        {"scale_factor": 5.0},
    ):
        other = registry.create(
            "inspect.boost_descriptor", parameters={"use_orientation": False, **parameters}
        ).execute(settings).outputs["features"]
        assert matcher.execute({"query": first, "train": other}) is None
        assert "same descriptor family and settings" in matcher.error
    wrong_length = {**first, "descriptor_size": 16}
    assert matcher.execute({"query": first, "train": wrong_length}) is None
    assert "Invalid BoostDesc variant, width" in matcher.error


def test_rejects_unmeasured_orientation_and_oversized_patches(registry, texture):
    descriptor = registry.create("inspect.boost_descriptor")
    assert descriptor.execute({"image": texture, "keypoints": [[50, 50]]}) is None
    assert "needs a measured angle" in descriptor.error
    keys = [{"point": [50, 50], "size": 10000, "angle": 0}]
    assert descriptor.execute({"image": texture, "keypoints": keys}) is None
    assert "sampling diameter" in descriptor.error


@pytest.mark.parametrize(("profile", "workers"), [("debug", 1), ("lean", 3)])
def test_saved_comparison_matches_known_geometry_with_native_norms(registry, profile, workers):
    graph = Graph.load(EXAMPLE / "boost_descriptors.json", registry)
    try:
        graph.set_run_profile(profile)
        graph.set_branch_workers(workers)
        graph.run()
        for key, minimum, limit in (("binboost", 175, 48), ("bgm", 170, 48), ("lbgm", 155, 1.3)):
            block = graph.blocks[f"{key}_query"]
            assert block_entry(block).path[:2] == ("OpenCV", "Extra 2D Features (xfeatures2d)")
            descriptor = block.last_result.outputs["features"]
            train = graph.blocks[f"{key}_train"].last_result.outputs["features"]
            evaluation = graph.blocks[f"{key}_evaluate"].last_result
            assert evaluation and evaluation.passed, graph.blocks[f"{key}_evaluate"].error
            assert evaluation.outputs["precision"] > 0.98
            assert evaluation.outputs["repeatability"] > 0.65
            matches = graph.blocks[f"{key}_match"].last_result.outputs["matches"]["matches"]
            assert len(matches) >= minimum
            for match in matches:
                q = descriptor["descriptors"][match["query_index"]]
                t = train["descriptors"][match["train_index"]]
                distance = (
                    np.linalg.norm(np.subtract(q, t)) if key == "lbgm"
                    else np.unpackbits(np.array(q, np.uint8) ^ np.array(t, np.uint8)).sum()
                )
                assert match["distance"] == pytest.approx(distance, abs=2e-6)
                assert match["distance"] <= limit
    finally:
        graph.close()
