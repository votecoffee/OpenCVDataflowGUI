"""Check palette indices, custom palettes, invalid samples, and graph wiring."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import DepthArray, ImageArray
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import frame, metadata


def test_fixed_range_clips_before_using_custom_rgb_palette():
    palette = np.zeros((1, 256, 3), np.uint8)
    palette[0, :, 0] = np.arange(256, dtype=np.uint8)
    palette[0, :, 2] = 255 - palette[0, :, 0]
    block = builtin_registry().create(
        "cv.colorize_scalar", parameters={"range_mode": "fixed", "minimum": 0, "maximum": 1}
    )
    result = block.execute(
        {"image": np.array([[-1, 0, 0.5, 1, 2]]), "palette": ImageArray(palette, "rgb")}
    )
    assert result is not None and result.passed, block.error
    expected = np.array([[0, 0, 128, 255, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(result.outputs["levels"], expected)
    output = result.outputs["image"]
    np.testing.assert_array_equal(output, palette[0][expected][..., ::-1])
    assert output.color_format == "bgr"


def test_grayscale_palette_returns_grayscale_output():
    palette = np.arange(255, -1, -1, dtype=np.uint8).reshape(256, 1)
    block = builtin_registry().create("cv.colorize_scalar")
    result = block.execute({"image": np.array([[-200, 200]], np.int16), "palette": palette})
    assert result is not None and result.passed, block.error
    np.testing.assert_array_equal(result.outputs["image"], [[255, 0]])
    assert result.outputs["image"].color_format == "gray"


def test_auto_range_uses_mask_region_intersection_and_ignores_nonfinite_values():
    pixels = np.array([[np.inf, -2, 2, 300], [-999, np.nan, 99, 500]], dtype=np.float64)
    spatial = frame([4, 2], "sensor")
    mask = np.array([[255, 255, 255, 0], [255, 255, 255, 255]], np.uint8)
    region = {
        "model": "image_region_v1",
        "image_size": [4, 2],
        "shape": "rectangle",
        "rectangle": [0, 0, 4, 1],
    }
    block = builtin_registry().create("cv.colorize_scalar")
    result = block.execute(
        {"image": ImageArray(pixels, "gray", {"spatial": spatial}), "mask": mask, "region": region}
    )
    assert result is not None and result.passed, block.error
    outputs = result.outputs
    assert outputs["range"] == {"minimum": -2, "maximum": 2, "valid_count": 2, "units": None}
    np.testing.assert_array_equal(outputs["valid_mask"], [[0, 255, 255, 0], [0, 0, 0, 0]])
    np.testing.assert_array_equal(outputs["levels"], [[0, 0, 255, 0], [0, 0, 0, 0]])
    assert np.count_nonzero(outputs["image"][outputs["valid_mask"] == 0]) == 0
    assert metadata(outputs["image"])["spatial"] == spatial


def test_depth_ignores_nonpositive_values_and_reports_units():
    depth = DepthArray(np.array([[np.nan, 0, -1, 1.5, 3.0]]), "m")
    block = builtin_registry().create("cv.colorize_scalar")
    result = block.execute({"image": depth})
    assert result is not None and result.passed, block.error
    outputs = result.outputs
    assert outputs["range"] == {"minimum": 1.5, "maximum": 3.0, "valid_count": 2, "units": "m"}
    np.testing.assert_array_equal(outputs["valid_mask"], [[0, 0, 0, 255, 255]])
    assert outputs["image"].color_format == "bgr"


def test_wide_float64_range_and_constant_range():
    block = builtin_registry().create("cv.colorize_scalar")
    limit = np.finfo(np.float64).max
    result = block.execute({"image": np.array([[-limit, 0, limit]])})
    assert result is not None and result.passed, block.error
    np.testing.assert_array_equal(result.outputs["levels"], [[0, 128, 255]])
    result = block.execute({"image": np.ones((2, 2), np.float32)})
    assert result is not None and result.passed, block.error
    assert result.outputs["range"]["minimum"] == result.outputs["range"]["maximum"] == 1
    assert not result.outputs["levels"].any()
    assert result.outputs["valid_mask"].all()


def test_all_invalid_pixels_emit_a_black_result_and_fail():
    block = builtin_registry().create("cv.colorize_scalar")
    result = block.execute({"image": np.array([[np.nan, -np.inf]])})
    assert result is not None and not result.passed, block.error
    assert result.fail_reasons[0]["code"] == "no_valid_color_samples"
    assert not result.outputs["image"].any()
    assert result.outputs["range"]["minimum"] is None


def test_invalid_fixed_range_and_palette_are_rejected():
    block = builtin_registry().create("cv.colorize_scalar")
    with pytest.raises(ValidationError, match="minimum must be less"):
        block.validate_parameters({**block.parameters, "range_mode": "fixed", "minimum": 255})
    assert (
        block.execute({"image": np.ones((2, 2)), "palette": np.zeros((16, 16), np.uint8)}) is None
    )
    assert "256-pixel uint8 strip" in block.error


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_example_maps_distance_only_inside_the_circular_part(profile):
    project = (
        Path(__file__).parents[1]
        / "examples/opencv tools"
        / "colorize_scalar"
        / "colorize_scalar.json"
    )
    graph = Graph.load(project, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    outputs = graph.blocks["colors"].last_result.outputs
    mask = graph.blocks["part"].last_result.outputs["mask"] != 0
    assert outputs["range"]["valid_count"] == int(mask.sum())
    assert outputs["range"]["minimum"] == 1
    assert 72 < outputs["range"]["maximum"] < 73
    expected = cv2.applyColorMap(np.asarray(outputs["levels"]), cv2.COLORMAP_VIRIDIS)
    expected[~mask] = 0
    np.testing.assert_array_equal(outputs["image"], expected)
