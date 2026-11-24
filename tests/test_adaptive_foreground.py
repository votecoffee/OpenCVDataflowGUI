"""Qualify native foreground models through the same saved inspection graph."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

TOOLS = Path(__file__).parents[1] / "examples/opencv tools"
PROJECT = TOOLS / "adaptive_foreground_models/adaptive_foreground_models.json"


@pytest.fixture(scope="module")
def registry():
    return builtin_registry()


@pytest.mark.parametrize("profile_name,workers", [("debug", 1), ("lean", 3)])
def test_saved_replay_matches_known_part_masks(registry, profile_name, workers):
    assets = TOOLS / "background_reference/assets"
    mask_a, mask_b = (cv2.imread(str(assets / f"mask_{suffix}.png"), 0) for suffix in ("a", "b"))
    empty = np.zeros_like(mask_a)
    graph = Graph.load(PROJECT, registry)
    try:
        graph.set_run_profile(profile_name)
        graph.set_branch_workers(workers)
        for index in range(27):
            graph.run()
            truth = mask_a if 22 <= index < 24 else mask_b if 24 <= index < 26 else empty
            for key, first_ready in (("mog", 1), ("gmg", 13), ("cnt", 10)):
                block = graph.blocks[key]
                result = block.last_result
                assert result and result.passed, block.error
                assert result.outputs["status"]["frame_index"] == index
                assert result.outputs["ready"] is (index >= first_ready)
                observed = result.outputs["mask"]
                if index < first_ready:
                    assert observed is NO_VALUE and not result.drawings
                    continue
                assert observed.dtype == np.uint8
                expected = cv2.medianBlur(truth, 7) if key == "gmg" else truth
                np.testing.assert_array_equal(observed, expected)
                if key == "cnt":
                    assert image_format(result.outputs["background"]) == "gray"
                payload = result.payloads["mask"]
                assert payload["spatial"]["root_id"] == "video:image"
                assert payload["video_frame"]["timestamp_ms"] == pytest.approx(index * 50)
                assert payload["ticket"] == result.payloads["image"]["ticket"]
                assert bool(result.drawings) == (profile_name == "debug" and bool(truth.any()))
        for key in ("mog", "gmg", "cnt"):
            assert block_entry(graph.blocks[key]).path[:2] == (
                "OpenCV",
                "Background and Foreground Segmentation (bgsegm)",
            )
    finally:
        graph.close()


@pytest.mark.parametrize(
    "key,settings",
    [
        ("mog", {"learning_rate": 0}),
        ("gmg", {"initialization_frames": 2, "smoothing_size": 0, "update_background": False}),
        ("cnt", {"min_stability": 3, "history_scale": 0.5, "parallel": False}),
    ],
)
def test_color_mask_roi_payload_and_foreground_reset(registry, key, settings):
    block = registry.create(f"sequence.{key}_background", parameters={"warmup": 5, **settings})
    clean = ImageArray(np.full((32, 40, 4), [90, 80, 60, 255], dtype=np.uint8), "rgba")
    part = clean.copy()
    part[5:17, 8:28, :3] = [220, 180, 150]
    allowed = np.zeros((32, 40), np.uint8)
    allowed[8:22, 4:16] = 1
    expected = np.zeros_like(allowed)
    expected[8:17, 8:16] = 255
    spatial = frame([40, 32], "camera")
    clock = TicketClock("adaptive")

    def execute(image, index, **options):
        ticket = clock.issue("camera", "fixed", f"frame-{index}")
        payload = {
            "spatial": spatial,
            "ticket": ticket,
            "alpha_representation": "straight",
            "video_frame": {"frame_index": index},
        }
        result = block.execute(
            {
                "image": ValuePacket(image, payload),
                "mask": ValuePacket(allowed, {"spatial": spatial}),
                **options,
            }
        )
        assert result and result.passed, block.error
        return result

    for index in range(5):
        assert execute(clean, index).outputs["ready"] is (index == 4)
    found = execute(part, 5)
    assert found.outputs["status"]["foreground_pixels"] == 72
    np.testing.assert_array_equal(found.outputs["mask"], expected)
    actual = found.outputs["foreground"]
    assert image_format(actual) == "rgba"
    np.testing.assert_array_equal(actual[expected != 0], part[expected != 0])
    assert not actual[expected == 0].any()
    assert "alpha_representation" not in found.payloads["mask"]
    assert found.drawings["adaptive foreground"].objects
    assert execute(part, 6, stream="other").outputs["mask"] is NO_VALUE
    rewound = execute(clean, 0)
    assert rewound.outputs["mask"] is NO_VALUE
    assert rewound.outputs["status"]["model_restart"] == "video position restarted"
    assert rewound.outputs["status"]["model_frames"] == 1
    reset = execute(clean, 1, reset=True)
    assert reset.outputs["status"]["generation"] != rewound.outputs["status"]["generation"]


def test_gmg_custom_float_range_and_retained_memory_limit(registry):
    block = registry.create(
        "sequence.gmg_background",
        parameters={
            "initialization_frames": 2,
            "max_features": 4,
            "smoothing_size": 0,
            "intensity_range": "custom",
            "minimum_intensity": -10.0,
            "maximum_intensity": 30.0,
        },
    )
    clean = np.full((32, 32), -5.5, np.float32)
    for _ in range(4):
        result = block.execute({"image": clean})
        assert result and result.passed, block.error
    assert result.outputs["ready"] and not result.outputs["mask"].any()
    outside = block.execute({"image": np.full_like(clean, 60)})
    assert not outside or not outside.passed
    assert "intensity range" in str(block.error)
    assert not block.memory["streams"]
    block.parameters["max_bytes_mb"] = 0.01
    oversized = block.execute({"image": clean})
    assert not oversized or not oversized.passed
    assert "sequence memory byte limit" in str(block.error)
    assert not block.memory["streams"]


@pytest.mark.parametrize(
    "key,changes,error",
    [
        ("mog", {"learning_rate": -0.5}, "Learning rate"),
        ("gmg", {"smoothing_size": 4}, "zero or odd"),
        ("cnt", {"max_stability": 30, "history_scale": 0.2}, "Scaled history"),
    ],
)
def test_invalid_model_configuration_is_reported_without_native_state(
    registry, key, changes, error
):
    block = registry.create(f"sequence.{key}_background", parameters=changes)
    result = block.execute({"image": np.zeros((12, 12), np.uint8)})
    assert not result or not result.passed
    assert error in str(block.error)
    assert not block.memory["streams"]
