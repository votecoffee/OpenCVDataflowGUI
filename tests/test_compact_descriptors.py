"""Qualify native descriptor bytes, keypoint lineage and Hamming matching."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/compact_binary_descriptors"


def tool(method, **settings):
    return builtin_registry().create(f"inspect.{method}_descriptor", parameters=settings)


@pytest.mark.parametrize("width", [16, 32, 64])
def test_brief_matches_native_and_keeps_aligned_original_indices(width):
    image = np.random.default_rng(62).integers(0, 256, (128, 152), dtype=np.uint8)
    rgb = ImageArray(np.repeat((image.astype(np.uint16) * 257)[..., None], 3, axis=2), "rgb")
    points = [[3, 3], [72, 63], [95, 64], [70.5, 80.25]]
    keys = [cv2.KeyPoint(*p, 11, 30, 0.5, 1, i) for i, p in enumerate(points)]
    upstream = {
        "image_size": [152, 128],
        "keypoints": [
            {"point": p, "size": 11, "angle": 30, "response": 0.5, "octave": 1} for p in points
        ],
    }
    expected_keys, expected = cv2.xfeatures2d.BriefDescriptorExtractor_create(width, True).compute(
        image, keys
    )
    node = tool("brief", descriptor_bytes=str(width), use_orientation=True)
    result = node.execute({"image": rgb, "keypoints": upstream})
    assert result and result.passed, node.error
    data = result.outputs["features"]
    assert data["descriptor_size"] == width
    assert [k["source_index"] for k in data["keypoints"]] == [k.class_id for k in expected_keys]
    assert [k["octave"] for k in data["keypoints"]] == [1, 1, 1]
    np.testing.assert_array_equal(data["descriptors"], expected)
    np.testing.assert_array_equal(result.outputs["image"], rgb)
    assert result.outputs["image"].color_format == "rgb"
    assert block_entry(node).path[:2] == ("OpenCV", "Extra 2D Features (xfeatures2d)")


def test_freak_custom_pairs_use_mask_intersection_and_keep_frame_and_ticket():
    image = np.random.default_rng(17).integers(0, 256, (144, 160), dtype=np.uint8)
    points = [[5, 5], [65, 57], [87, 60], [95, 91], [67, 97]]
    mask = np.full_like(image, 255)
    mask[59:62, 85:89] = 0
    scale = [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
    spatial = {**frame([320, 288], "vision"), "image_size": [160, 144], "to_root": scale}
    pyramid = {"level": 1, "largest_size": [320, 288], "to_largest": scale}
    ticket = TicketClock("freak").issue("camera", "fixed", "plate")
    payload = {"spatial": spatial, "pyramid": pyramid, "ticket": ticket}
    region = {
        "model": "image_region_v1",
        "image_size": [160, 144],
        "shape": "rectangle",
        "rectangle": [0, 0, 90, 115],
        "coordinate_context": spatial,
    }
    pairs = list(range(391, 903))
    node = tool("freak", pattern_scale=17.0, octaves=3, selected_pairs=pairs)
    result = node.execute(
        {
            "image": ValuePacket(image, payload),
            "keypoints": ValuePacket(points, payload),
            "mask": ValuePacket(mask, {"spatial": spatial}),
            "region": region,
        }
    )
    assert result and result.passed, node.error
    data = result.outputs["features"]
    native = cv2.xfeatures2d.FREAK_create(True, True, 17, 3, pairs)
    keys, descriptors = native.compute(image, [cv2.KeyPoint(*points[i], 7) for i in (0, 1, 4)])
    assert [k["source_index"] for k in data["keypoints"]] == [1, 4]
    np.testing.assert_array_equal(
        [k["angle"] for k in data["keypoints"]], [k.angle % 360 for k in keys]
    )
    np.testing.assert_array_equal(data["descriptors"], descriptors)
    for field, value in payload.items():
        assert result.payloads["features"][field] == value
    assert len(result.drawings["FREAK"].objects) == 2


def test_empty_and_invalid_keypoints_and_incompatible_descriptor_layouts():
    image = np.random.default_rng(19).integers(0, 256, (96, 96), dtype=np.uint8)
    brief = tool("brief", descriptor_bytes="64")
    empty = brief.execute({"image": image, "keypoints": [[0, 0]]})
    assert empty and empty.passed and empty.outputs["features"]["descriptors"] == []
    assert brief.execute({"image": image, "keypoints": [{"point": [45, 45], "size": 0}]}) is None
    assert "valid size" in brief.error
    oriented = tool("brief", use_orientation=True)
    assert oriented.execute({"image": image, "keypoints": [[45, 45]]}) is None
    assert "measured angle" in oriented.error
    with pytest.raises(ValidationError) as error:
        tool("freak", selected_pairs=list(range(511)) + [903]).validate_parameters()
    assert error.value.fields == ("selected_pairs",)
    clock = TicketClock("checks")
    image_ctx = {
        "spatial": frame([96, 96], "source"),
        "ticket": clock.issue("camera", "fixed", "a"),
    }
    point_ctx = {**image_ctx, "spatial": frame([96, 96], "another")}
    assert (
        brief.execute(
            {
                "image": ValuePacket(image, image_ctx),
                "keypoints": ValuePacket([[45, 45]], point_ctx),
            }
        )
        is None
    )
    assert "coordinate frame" in brief.error
    point_ctx = {**image_ctx, "ticket": clock.issue("camera", "fixed", "b")}
    assert (
        brief.execute(
            {
                "image": ValuePacket(image, image_ctx),
                "keypoints": ValuePacket([[45, 45]], point_ctx),
            }
        )
        is None
    )
    assert "different acquisition tickets" in brief.error
    inputs = {"image": image, "keypoints": [[45, 45]]}
    q = brief.execute(inputs).outputs["features"]
    t = tool("freak").execute(inputs).outputs["features"]
    matcher = builtin_registry().create("inspect.match_descriptor_radius")
    assert matcher.execute({"query": q, "train": t}) is None
    assert "same descriptor family and settings" in matcher.error
    t = tool("freak", pattern_scale=18).execute(inputs).outputs["features"]
    t2 = (
        tool("freak", orientation_normalized=False, pattern_scale=18)
        .execute(inputs)
        .outputs["features"]
    )
    assert matcher.execute({"query": t, "train": t2}) is None
    assert "same descriptor family and settings" in matcher.error


@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_example_hamming_matches_agree_with_translation_truth(profile):
    graph = Graph.load(EXAMPLE / "compact_binary_descriptors.json", builtin_registry())
    graph.set_run_profile(profile)
    graph.set_branch_workers(2)
    try:
        graph.run()
        for method in ("brief", "freak"):
            match = graph.blocks[f"{method}_match"].last_result.outputs["matches"]
            evaluation = graph.blocks[f"{method}_evaluate"].last_result
            operating = evaluation.outputs["metrics"]["operating"]
            assert evaluation.passed and operating["true_positives"] >= 30
            assert operating["precision"] == operating["recall"] == 1
            shift = np.asarray(match["train_points"]) - np.asarray(match["query_points"])
            np.testing.assert_array_equal(shift, np.tile([9, 7], (len(shift), 1)))
            descriptors = {
                side: np.asarray(
                    graph.blocks[f"{method}_{side}"].last_result.outputs["features"]["descriptors"],
                    np.uint8,
                )
                for side in ("query", "train")
            }
            for row in match["matches"]:
                query = descriptors["query"][row["query_index"]]
                train = descriptors["train"][row["train_index"]]
                assert int(np.unpackbits(query ^ train).sum()) == row["distance"]
    finally:
        graph.close()
