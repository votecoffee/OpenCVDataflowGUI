"""Verify selected validity counts, masks, and nonfinite preview behavior."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray, preview_rgb
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import NO_VALUE, frame, metadata


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_color_checks_every_channel_and_classifies_each_invalid_pixel_once(dtype):
    pixels = np.array(
        [
            [[0, 0.5, 0.5], [0.5, 0.5, np.nan], [1, np.inf, 0]],
            [[-np.inf, 0, 0], [0.5, 1, 0], [0, 0, 0]],
        ],
        dtype=dtype,
    )
    spatial = frame([3, 2], "camera")
    image = ImageArray(pixels, "rgb", {"spatial": spatial})
    check = builtin_registry().create("inspect.image_validity", parameters={"check_range": True})
    result = check.execute({"image": image})
    assert result is not None and not result.passed, check.error
    assert result.fail_reasons[0]["code"] == "invalid_pixel_fraction_exceeds_limit"
    assert result.outputs["report"] == {
        "selected_count": 6,
        "valid_count": 2,
        "nonfinite_count": 3,
        "out_of_range_count": 1,
        "first_invalid_pixel": [1, 0],
    }
    assert result.outputs["invalid_count"] == 4
    assert result.outputs["invalid_fraction"] == pytest.approx(4 / 6)
    np.testing.assert_array_equal(result.outputs["valid_mask"], [[255, 0, 0], [0, 0, 255]])
    np.testing.assert_array_equal(result.outputs["invalid_mask"], [[0, 255, 255], [255, 255, 0]])
    assert metadata(result.outputs["valid_mask"])["spatial"] == spatial
    assert metadata(result.outputs["invalid_mask"])["spatial"] == spatial
    assert result.drawings["first invalid pixel"].objects[0].points == ((1.0, 0.0),)


def test_range_checks_raw_alpha_and_uses_a_float64_cutoff():
    pixels = np.array([[[0.5, 1, 1, 1], [1, 0.5, 1, 0]]], dtype=np.float32)
    check = builtin_registry().create(
        "inspect.image_validity",
        parameters={"check_range": True, "minimum": 0.5, "maximum": float(np.nextafter(1.0, 2.0))},
    )
    result = check.execute({"image": ImageArray(pixels, "rgba")})
    assert result is not None, check.error
    np.testing.assert_array_equal(result.outputs["valid_mask"], [[255, 0]])
    assert result.outputs["report"]["out_of_range_count"] == 1


def test_intersection_checks_only_selected_integer_pixels():
    image = np.array([[0, 5, 9], [10, 11, 20]], dtype=np.uint16)
    mask = np.array([[255, 255, 0], [255, 0, 255]], dtype=np.uint8)
    region = {
        "model": "image_region_v1",
        "image_size": [3, 2],
        "shape": "rectangle",
        "rectangle": [0, 0, 2, 2],
    }
    check = builtin_registry().create(
        "inspect.image_validity",
        parameters={"check_range": True, "minimum": 5, "maximum": 10, "max_invalid_fraction": 1},
    )
    result = check.execute({"image": image, "mask": mask, "region": region})
    assert result is not None and result.passed, check.error
    assert result.outputs["report"]["selected_count"] == 3
    assert result.outputs["report"]["nonfinite_count"] == 0
    assert result.outputs["report"]["out_of_range_count"] == 2
    np.testing.assert_array_equal(result.outputs["valid_mask"], [[0, 255, 0], [0, 0, 0]])
    np.testing.assert_array_equal(result.outputs["invalid_mask"], [[255, 0, 0], [255, 0, 0]])


def test_empty_selection_fails_with_no_fraction_or_invalid_location():
    check = builtin_registry().create("inspect.image_validity")
    result = check.execute({"image": np.ones((2, 2)), "mask": np.zeros((2, 2), np.uint8)})
    assert result is not None and not result.passed, check.error
    assert result.fail_reasons[0]["code"] == "no_selected_pixels"
    assert result.outputs["invalid_fraction"] is NO_VALUE
    assert result.outputs["report"]["first_invalid_pixel"] is None
    assert not result.outputs["valid_mask"].any()
    assert not result.outputs["invalid_mask"].any()


def test_editor_rejects_an_inverted_range():
    check = builtin_registry().create("inspect.image_validity")
    with pytest.raises(ValidationError, match="Minimum must be less") as error:
        check.validate_parameters(
            {**check.parameters, "check_range": True, "minimum": 1, "maximum": 1}
        )
    assert error.value.fields == ("minimum", "maximum")


@pytest.mark.parametrize("color", ("gray", "rgb", "hsv"))
def test_nonfinite_preview_marks_pixels_without_changing_source(color):
    samples = np.array([[np.nan, 0.5, np.inf]], dtype=np.float32)
    if color != "gray":
        samples = np.stack((np.zeros_like(samples), np.zeros_like(samples), samples), axis=2)
    original = samples.copy()
    rendered = preview_rgb(ImageArray(samples, color))
    valid = [0, 0, 128] if color == "rgb" else [128, 128, 128]
    np.testing.assert_array_equal(rendered, [[[255, 0, 255], valid, [255, 0, 255]]])
    np.testing.assert_array_equal(samples, original)


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_synthetic_depth_graph_measures_only_valid_source_pixels(profile):
    project = (
        Path(__file__).parents[1]
        / "examples/image_processing"
        / "image_validity"
        / "image_validity.json"
    )
    graph = Graph.load(project, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    check = graph.blocks["check"].last_result
    assert check.outputs["invalid_count"] == 100
    assert check.outputs["invalid_fraction"] == 100 / 4096
    assert check.outputs["report"]["first_invalid_pixel"] == [0, 0]
    assert check.outputs["report"]["out_of_range_count"] == 0
    assert ("first invalid pixel" in check.drawings) == (profile == "debug")
    assert graph.blocks["statistics"].last_result.outputs["selected_count"] == 3996
