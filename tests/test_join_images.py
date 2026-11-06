"""Verify native concatenation, padding, and per-panel provenance."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import frame, metadata
from opencv_dataflow.tickets import TicketClock


@pytest.mark.parametrize(("direction", "axis"), (("horizontal", 1), ("vertical", 0)))
def test_strict_join_matches_array_concatenation(direction, axis):
    a = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    b = np.array([[65000, 12], [34, 56]], dtype=np.uint16)
    block = builtin_registry().create("cv.join_images", parameters={"direction": direction})
    result = block.execute({"image_a": a, "image_b": b})
    assert result is not None and result.passed, block.error
    np.testing.assert_array_equal(result.outputs["image"], np.concatenate((a, b), axis=axis))
    assert result.outputs["image"].dtype == np.uint16


@pytest.mark.parametrize("direction", ("horizontal", "vertical"))
@pytest.mark.parametrize(("alignment", "before"), (("start", 0), ("center", 1), ("end", 3)))
def test_padding_and_gap_preserve_pixels_and_source_frames(direction, alignment, before):
    a = np.full((4, 2, 3), (10, 20, 30), dtype=np.uint8)
    b = np.full((1, 3, 3), (40, 50, 60), dtype=np.uint8)
    size, a_rect, b_rect = [7, 4], [0, 0, 2, 4], [4, before, 3, 1]
    if direction == "vertical":
        a, b = a.swapaxes(0, 1), b.swapaxes(0, 1)
        size = size[::-1]
        a_rect, b_rect = [0, 0, 4, 2], [before, 4, 1, 3]
    a_frame, b_frame = frame(a.shape[:2][::-1], "a"), frame(b.shape[:2][::-1], "b")
    pyramid = {"level": 1, "largest_size": [6, 8], "to_largest": np.diag([2, 2, 1]).tolist()}
    block = builtin_registry().create(
        "cv.join_images",
        parameters={
            "direction": direction,
            "size_policy": "pad",
            "alignment": alignment,
            "gap": 2,
            "fill_value": 7,
        },
    )
    result = block.execute(
        {
            "image_a": ImageArray(a, "rgb", {"spatial": a_frame}),
            "image_b": ImageArray(b, "rgb", {"spatial": b_frame, "pyramid": pyramid}),
        }
    )
    assert result is not None and result.passed, block.error
    joined, layout = result.outputs["image"], result.outputs["layout"]
    assert joined.color_format == "rgb" and joined.dtype == np.uint8
    expected = np.full((size[1], size[0], 3), 7, dtype=np.uint8)
    for image, (x, y, width, height) in zip((a, b), (a_rect, b_rect), strict=True):
        expected[y : y + height, x : x + width] = image
    np.testing.assert_array_equal(joined, expected)
    assert layout["image_size"] == size
    assert [p["rect"] for p in layout["panels"]] == [a_rect, b_rect]
    assert [p["source_spatial"] for p in layout["panels"]] == [a_frame, b_frame]
    assert layout["panels"][1]["source_pyramid"] == pyramid
    assert metadata(joined)["spatial"] == frame(size, f"{block.id}:image")
    assert "pyramid" not in metadata(joined)


@pytest.mark.parametrize(
    ("parameters", "a", "b", "message"),
    (
        ({}, np.zeros((2, 1), np.uint8), np.zeros((3, 1), np.uint8), "same image height"),
        (
            {"direction": "vertical"},
            np.zeros((1, 2), np.uint8),
            np.zeros((1, 3), np.uint8),
            "same image width",
        ),
        ({}, np.zeros((1, 1), np.uint8), np.zeros((1, 1), np.uint16), "same pixel depth"),
        (
            {"gap": 1, "fill_value": 256},
            np.zeros((1, 1), np.uint8),
            np.zeros((1, 1), np.uint8),
            "uint8 range",
        ),
        (
            {"gap": 16384},
            np.zeros((1024, 1), np.uint8),
            np.zeros((1024, 1), np.uint8),
            "exceeds 16,777,216",
        ),
    ),
)
def test_rejects_incompatible_inputs_and_unbounded_output(parameters, a, b, message):
    block = builtin_registry().create("cv.join_images", parameters=parameters)
    assert block.execute({"image_a": a, "image_b": b}) is None
    assert message in block.error


def test_rejects_mixed_channel_encodings():
    pixels = np.zeros((1, 1, 3), np.uint8)
    block = builtin_registry().create("cv.join_images")
    assert block.execute({"image_a": ImageArray(pixels, "rgb"), "image_b": pixels}) is None
    assert "color encoding" in block.error


def test_matches_tickets_across_sources_and_keeps_the_only_input_ticket():
    pixels = np.zeros((1, 1), np.uint8)
    clock = TicketClock(session="test")
    a = ImageArray(pixels, "gray", {"ticket": clock.issue("a", "fixed", 3)})
    b = ImageArray(pixels, "gray", {"ticket": clock.issue("b", "fixed", 3)})
    other = ImageArray(pixels, "gray", {"ticket": clock.issue("b", "fixed", 4)})
    block = builtin_registry().create("cv.join_images")
    for first in (a, pixels):
        result = block.execute({"image_a": first, "image_b": b})
        assert result is not None and result.passed, block.error
        assert metadata(result.outputs["image"])["ticket"]["value"] == 3
        assert result.payloads["layout"]["ticket"]["value"] == 3
    assert block.execute({"image_a": a, "image_b": other}) is None
    assert "same ticket" in block.error


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_example_contains_original_and_filtered_crop(profile):
    project = Path(__file__).parents[1] / "examples" / "join_images" / "join_images.json"
    graph = Graph.load(project, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    outputs = graph.blocks["join"].last_result.outputs
    joined = outputs["image"]
    assert joined.shape == (360, 652, 3)
    np.testing.assert_array_equal(
        joined[:, :480], graph.blocks["source"].last_result.outputs["image"]
    )
    np.testing.assert_array_equal(joined[:, 480:492], 12)
    np.testing.assert_array_equal(
        joined[80:280, 492:], graph.blocks["blur"].last_result.outputs["image"]
    )
    crop = outputs["layout"]["panels"][1]
    assert crop["rect"] == [492, 80, 160, 200]
    np.testing.assert_array_equal(
        crop["source_spatial"]["to_root"], [[1, 0, 255], [0, 1, 80], [0, 0, 1]]
    )
