"""Check native texture descriptor rows, numeric distances and known-view matching."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.registration import radius_data
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/texture_descriptors"


@pytest.fixture(scope="module")
def registry():
    return builtin_registry()


@pytest.mark.parametrize(("code", "width"), [(100, 120), (101, 80), (102, 64), (103, 48)])
@pytest.mark.parametrize("normalized", [False, True])
def test_vgg_matches_all_native_variants_and_uses_l2_for_bytes(registry, code, width, normalized):
    image = np.random.default_rng(751).integers(0, 256, (120, 154), np.uint8)
    rows = [
        {"point": [44, 48], "size": 7, "angle": 30},
        {"point": [101, 74], "size": 9, "angle": 50},
    ]
    keys = [cv2.KeyPoint(*row["point"], row["size"], row["angle"]) for row in rows]
    _, expected = cv2.xfeatures2d.VGG_create(code, .85, False, True, 2.5, normalized).compute(
        image, keys
    )
    block = registry.create("inspect.vgg_descriptor", parameters={
        "variant": f"VGG_{width}", "normalize_descriptor": normalized, "scale_factor": 2.5,
        "sigma": .85, "normalize_image": False,
    })
    result = block.execute({"image": image, "keypoints": rows})
    assert result and result.passed, block.error
    features = result.outputs["features"]
    assert features["descriptor_size"] == width
    assert type(features["descriptors"][0][0]) is (int if normalized else float)
    values, _, _, norm = radius_data(features)
    assert values.dtype == np.float32 and norm == cv2.NORM_L2
    np.testing.assert_array_equal(values, expected)


def test_lucid_matches_native_color_and_preserves_source_and_selected_indices(registry):
    bgr = np.random.default_rng(752).integers(0, 256, (73, 94, 3), dtype=np.uint8)
    rgba = np.concatenate((bgr[..., ::-1], np.full((73, 94, 1), 100, np.uint8)), axis=2)
    rgba = ImageArray(rgba.astype(np.uint16) * 257, "rgba")
    rows = [[0, 0], [11, 11], [93, 72]]
    ticket = TicketClock("lucid-check").issue("camera", "fixed", "color")
    context = {"spatial": frame([94, 73], "camera"), "ticket": ticket}
    mask = np.full(bgr.shape[:2], 255, np.uint8)
    mask[11, 11] = 0
    result = registry.create("inspect.lucid_descriptor", parameters={
        "patch_radius": 3, "blur_radius": 1,
    }).execute({
        "image": ValuePacket(rgba, context), "keypoints": ValuePacket(rows, context), "mask": mask,
    })
    assert result and result.passed
    _, expected = cv2.xfeatures2d.LUCID_create(3, 1).compute(
        bgr, [cv2.KeyPoint(*rows[i], 7) for i in (0, 2)]
    )
    features = result.outputs["features"]
    parsed, _, _, norm = radius_data(features)
    assert parsed.dtype == np.uint8 and norm == cv2.NORM_HAMMING
    assert [row["source_index"] for row in features["keypoints"]] == [0, 2]
    np.testing.assert_array_equal(parsed, expected)
    assert result.outputs["image"].color_format == "rgba"
    np.testing.assert_array_equal(result.outputs["image"], rgba)
    assert result.payloads["features"]["spatial"] == context["spatial"]
    assert result.payloads["features"]["ticket"] == ticket


def test_lucid_rejects_invalid_color_patch_size_and_wire_width(registry):
    block = registry.create("inspect.lucid_descriptor", parameters={"patch_radius": 4})
    assert block.execute({"image": np.zeros((7, 8), np.uint8), "keypoints": [[1, 1]]}) is None
    assert "needs RGB or BGR" in block.error
    assert block.execute({"image": np.zeros((4, 8, 3), np.uint8), "keypoints": [[1, 1]]}) is None
    assert "smaller than both image axes" in block.error
    data = block.execute({"image": np.zeros((11, 12, 3), np.uint8), "keypoints": []})
    features = data.outputs["features"]
    values, _, _, norm = radius_data(features)
    assert values.shape == (0, 243) and norm == cv2.NORM_HAMMING
    with pytest.raises(ValueError, match="LUCID width"):
        radius_data({**features, "descriptor_size": 27})


def test_vgg_matcher_rejects_different_normalization_and_patch_settings(registry):
    image = np.random.default_rng(753).integers(0, 256, (84, 93), np.uint8)
    inputs = {"image": image, "keypoints": [[40, 40], [53, 56]]}

    def descriptor(**params):
        block = registry.create(
            "inspect.vgg_descriptor", parameters={"use_orientation": False, **params}
        )
        return block.execute(inputs).outputs["features"]

    query = descriptor()
    matcher = registry.create("inspect.match_descriptor_radius", parameters={"max_distance": 0})
    assert matcher.execute({"query": query, "train": query}).outputs["count"] == 2
    for other in (descriptor(normalize_descriptor=True), descriptor(sigma=.9)):
        assert matcher.execute({"query": query, "train": other}) is None
        assert "same descriptor family and settings" in matcher.error


@pytest.mark.parametrize(("profile", "workers"), [("debug", 1), ("lean", 3)])
def test_saved_texture_comparison_checks_native_distance_and_geometry(registry, profile, workers):
    graph = Graph.load(EXAMPLE / "texture_descriptors.json", registry)
    try:
        graph.set_run_profile(profile)
        graph.set_branch_workers(workers)
        graph.run()
        for key, cutoff in (("vgg_float", 1.0), ("vgg_byte", 40.0), ("lucid", 128.0)):
            block = graph.blocks[f"{key}_query"]
            assert block_entry(block).path[:2] == ("OpenCV", "Extra 2D Features (xfeatures2d)")
            q = block.last_result.outputs["features"]
            t = graph.blocks[f"{key}_train"].last_result.outputs["features"]
            q_rows, _, _, norm = radius_data(q)
            t_rows, _, _, _ = radius_data(t)
            matches = graph.blocks[f"{key}_match"].last_result.outputs["matches"]["matches"]
            assert len(matches) > 190
            for match in matches:
                first, second = q_rows[match["query_index"]], t_rows[match["train_index"]]
                assert match["distance"] == pytest.approx(cv2.norm(first, second, norm), abs=1e-5)
                assert match["distance"] <= cutoff
            evaluation = graph.blocks[f"{key}_evaluate"].last_result
            assert evaluation and evaluation.passed
            assert evaluation.outputs["precision"] > 0.98
    finally:
        graph.close()
