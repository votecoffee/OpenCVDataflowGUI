"""Check native routing, index validation, and the false-color example."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import frame, metadata
from opencv_dataflow.tickets import TicketClock


@pytest.mark.parametrize("dtype", (np.uint8, np.uint16, np.int16, np.float32, np.float64))
def test_wired_map_reorders_duplicates_and_zero_fills_at_the_input_depth(dtype):
    a = np.arange(16, dtype=dtype).reshape(2, 2, 4)
    b = a + 50
    block = builtin_registry().create("cv.route_channels", parameters={"color_format": "rgba"})
    result = block.execute({"image_a": a, "image_b": b, "channel_map": [7, 0, 7, -1]})
    assert result is not None and result.passed, block.error
    image = result.outputs["image"]
    np.testing.assert_array_equal(
        image, np.dstack((b[..., 3], a[..., 0], b[..., 3], np.zeros_like(a[..., 0])))
    )
    assert image.dtype == dtype and image.color_format == "rgba"


def test_gray_routing_uses_raw_samples_and_intersects_mask_with_region():
    spatial = frame([2, 2], "sensor")
    source = ImageArray(np.arange(12, dtype=np.uint8).reshape(2, 2, 3), "hsv", {"spatial": spatial})
    mask = np.array([[255, 0], [255, 255]], dtype=np.uint8)
    region = {
        "model": "image_region_v1",
        "image_size": [2, 2],
        "shape": "rectangle",
        "rectangle": [0, 0, 2, 1],
    }
    block = builtin_registry().create(
        "cv.route_channels", parameters={"color_format": "gray", "channel_map": [2]}
    )
    result = block.execute({"image_a": source, "mask": mask, "region": region})
    assert result is not None and result.passed, block.error
    image = result.outputs["image"]
    np.testing.assert_array_equal(image, [[2, 0], [0, 0]])
    assert image.color_format == "gray"
    assert metadata(image)["spatial"] == spatial


@pytest.mark.parametrize("mapping", ([0, 1], [True, 1, 0], [0, 1, -2], [0, 1, 8]))
def test_invalid_maps_are_reported_in_the_editor(mapping):
    block = builtin_registry().create("cv.route_channels")
    with pytest.raises(ValidationError) as error:
        block.validate_parameters({**block.parameters, "channel_map": mapping})
    assert error.value.fields == ("channel_map",)


def test_valid_index_syntax_requires_the_referenced_source_to_exist():
    block = builtin_registry().create(
        "cv.route_channels", parameters={"color_format": "gray", "channel_map": [3]}
    )
    assert block.execute({"image_a": np.zeros((2, 2, 3), dtype=np.uint8)}) is None
    assert "Only 3 source channels" in block.error


def test_alignment_can_explicitly_pair_different_frames_by_index():
    a = ImageArray(np.array([[10]], np.uint8), "gray", {"spatial": frame([1, 1], "a")})
    b = ImageArray(np.array([[20]], np.uint8), "gray", {"spatial": frame([1, 1], "b")})
    registry = builtin_registry()
    parameters = {"color_format": "bgr", "channel_map": [1, -1, 0]}
    strict = registry.create("cv.route_channels", parameters=parameters)
    assert strict.execute({"image_a": a, "image_b": b}) is None
    assert "different coordinate frames" in strict.error
    paired = registry.create(
        "cv.route_channels", parameters={**parameters, "alignment": "pixel_grid"}
    )
    result = paired.execute({"image_a": a, "image_b": b})
    assert result is not None and result.passed, paired.error
    np.testing.assert_array_equal(result.outputs["image"], [[[20, 0, 10]]])
    assert metadata(result.outputs["image"])["spatial"]["root_id"] == "a"


def test_same_ticket_is_required_and_the_second_ticket_can_propagate():
    pixel = np.ones((1, 1), dtype=np.uint8)
    clock = TicketClock(session="test")
    a = ImageArray(pixel, "gray", {"ticket": clock.issue("a", "fixed", 1)})
    b = ImageArray(pixel, "gray", {"ticket": clock.issue("b", "fixed", 2)})
    block = builtin_registry().create("cv.route_channels", parameters={"channel_map": [0, 1, 1]})
    assert block.execute({"image_a": a, "image_b": b}) is None
    assert "same ticket" in block.error
    result = block.execute({"image_a": pixel, "image_b": b})
    assert result is not None and result.passed, block.error
    assert metadata(result.outputs["image"])["ticket"]["value"] == 2


@pytest.mark.parametrize("second", (np.zeros((1, 2), np.uint8), np.zeros((1, 1), np.uint16)))
def test_second_source_must_match_dimensions_and_depth(second):
    block = builtin_registry().create("cv.route_channels", parameters={"channel_map": [0, 1, 1]})
    assert block.execute({"image_a": np.zeros((1, 1), np.uint8), "image_b": second}) is None
    assert "same size and pixel depth" in block.error


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_example_creates_rgb_from_two_sources(profile):
    project = (
        Path(__file__).parents[1]
        / "examples/opencv tools"
        / "route_channels"
        / "route_channels.json"
    )
    graph = Graph.load(project, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    source = graph.blocks["source"].last_result.outputs["image"]
    image = graph.blocks["route"].last_result.outputs["image"]
    assert image.color_format == "rgb"
    np.testing.assert_array_equal(image[..., 0], source[..., 2])
    np.testing.assert_array_equal(image[..., 1], 0)
    np.testing.assert_array_equal(image[..., 2], source[..., 1])
