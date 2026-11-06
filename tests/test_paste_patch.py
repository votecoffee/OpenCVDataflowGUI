"""Check masked copy, clipped placement, and coordinate provenance."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import frame, metadata
from opencv_dataflow.tickets import TicketClock


def test_wired_offsets_clip_and_apply_patch_mask_and_region():
    base = np.full((4, 5), 99, np.uint16)
    patch = np.arange(12, dtype=np.uint16).reshape(3, 4)
    canvas_frame, patch_frame = frame([5, 4], "canvas"), frame([4, 3], "patch")
    mask = np.full((3, 4), 255, np.uint8)
    mask[1, 2] = 0
    region = {
        "model": "image_region_v1",
        "image_size": [4, 3],
        "shape": "rectangle",
        "rectangle": [1, 1, 3, 1],
    }
    block = builtin_registry().create("cv.paste_patch", parameters={"x": 1, "y": 1})
    result = block.execute(
        {
            "canvas": ImageArray(base, "gray", {"spatial": canvas_frame}),
            "patch": ImageArray(patch, "gray", {"spatial": patch_frame}),
            "x": -1,
            "y": -1,
            "mask": ImageArray(mask, "gray", {"spatial": patch_frame}),
            "region": region,
        }
    )
    assert result is not None and result.passed, block.error
    outputs = result.outputs
    expected = base.copy()
    expected[0, [0, 2]] = [5, 7]
    np.testing.assert_array_equal(outputs["image"], expected)
    np.testing.assert_array_equal(base, 99)
    np.testing.assert_array_equal(outputs["copied_mask"] != 0, expected != base)
    assert outputs["image"].dtype == np.uint16
    assert outputs["copied_pixel_count"] == 2
    placement = outputs["placement"]
    assert placement["requested_rect"] == [-1, -1, 4, 3]
    assert placement["source_rect"] == [1, 1, 3, 2]
    assert placement["destination_rect"] == [0, 0, 3, 2]
    assert placement["source_spatial"] == patch_frame
    assert metadata(outputs["image"])["spatial"] == canvas_frame
    assert metadata(outputs["copied_mask"])["spatial"] == canvas_frame
    np.testing.assert_array_equal(np.asarray(placement["source_to_target"]) @ [1, 1, 1], [0, 0, 1])


def test_label_patch_copies_signed_values_and_counts_identical_pixels():
    base = np.zeros((3, 4), np.int32)
    patch = np.array([[0, -1], [2**31 - 1, -(2**31)]], np.int32)
    block = builtin_registry().create(
        "cv.paste_patch", parameters={"x": 2, "y": 1, "bounds_policy": "strict"}
    )
    result = block.execute({"canvas": base, "patch": patch})
    assert result is not None and result.passed, block.error
    np.testing.assert_array_equal(result.outputs["image"][1:, 2:], patch)
    assert result.outputs["copied_pixel_count"] == 4


def test_strict_rejects_clipping_and_clip_reports_empty_overlap_or_selection():
    canvas = np.ones((3, 3), np.uint8)
    patch = np.zeros((2, 2), np.uint8)
    registry = builtin_registry()
    strict = registry.create("cv.paste_patch", parameters={"x": 2, "bounds_policy": "strict"})
    assert strict.execute({"canvas": canvas, "patch": patch}) is None
    assert "fit inside" in strict.error
    clip = registry.create("cv.paste_patch")
    outside = clip.execute({"canvas": canvas, "patch": patch, "x": 3})
    assert outside is not None and not outside.passed, clip.error
    assert outside.fail_reasons[0]["code"] == "paste_no_overlap"
    assert outside.outputs["placement"]["destination_rect"] is None
    np.testing.assert_array_equal(outside.outputs["image"], canvas)
    empty = clip.execute({"canvas": canvas, "patch": patch, "mask": np.zeros((2, 2), np.uint8)})
    assert empty is not None and not empty.passed, clip.error
    assert empty.fail_reasons[0]["code"] == "paste_empty_selection"
    assert empty.outputs["copied_pixel_count"] == 0
    assert not empty.outputs["copied_mask"].any()


def test_mismatched_color_encoding_and_wrong_mask_frame_fail():
    pixels = np.zeros((2, 2, 3), np.uint8)
    block = builtin_registry().create("cv.paste_patch")
    assert block.execute({"canvas": pixels, "patch": ImageArray(pixels, "rgb")}) is None
    assert "same pixel depth and color encoding" in block.error
    patch = ImageArray(pixels, "bgr", {"spatial": frame([2, 2], "patch")})
    mask = ImageArray(np.ones((2, 2), np.uint8), "gray", {"spatial": frame([2, 2], "canvas")})
    assert block.execute({"canvas": pixels, "patch": patch, "mask": mask}) is None
    assert "Selection coordinate frame differs" in block.error


def test_ticket_identity_checks_and_patch_ticket_propagation():
    pixels = np.zeros((2, 2), np.uint8)
    clock = TicketClock(session="test")
    a = ImageArray(pixels, "gray", {"ticket": clock.issue("a", "fixed", 1)})
    b = ImageArray(pixels, "gray", {"ticket": clock.issue("b", "fixed", 2)})
    block = builtin_registry().create("cv.paste_patch")
    assert block.execute({"canvas": a, "patch": b}) is None
    assert "same ticket" in block.error
    result = block.execute({"canvas": pixels, "patch": b})
    assert result is not None and result.passed, block.error
    assert result.payloads["copied_pixel_count"]["ticket"]["value"] == 2


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_example_copies_only_the_visible_circle_selection(profile):
    project = (
        Path(__file__).parents[1] / "examples/image_processing" / "paste_patch" / "paste_patch.json"
    )
    graph = Graph.load(project, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    canvas = graph.blocks["source"].last_result.outputs["image"]
    patch = graph.blocks["patch"].last_result.outputs["image"]
    visible = graph.blocks["mask"].last_result.outputs["mask"][:, 24:] != 0
    outputs = graph.blocks["paste"].last_result.outputs
    expected = np.asarray(canvas).copy()
    expected[40:110, :46][visible] = patch[:, 24:][visible]
    np.testing.assert_array_equal(outputs["image"], expected)
    assert outputs["placement"]["source_rect"] == [24, 0, 46, 70]
    assert outputs["placement"]["destination_rect"] == [0, 40, 46, 70]
    assert outputs["copied_pixel_count"] == int(visible.sum())
