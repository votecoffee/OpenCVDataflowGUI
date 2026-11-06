"""Verify signed conversion, saturation, selection, and the example graph."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import frame, metadata


def test_scale_and_offset_wires_override_configuration_before_absolute_value():
    source = np.array([[-100, -2, -1, 0, 1, 2, 100]], dtype=np.int16)
    block = builtin_registry().create("cv.scale_absolute", parameters={"scale": 1, "offset": 0})
    result = block.execute({"image": source, "scale": -2, "offset": 3})
    assert result is not None and result.passed, block.error
    output = result.outputs["image"]
    np.testing.assert_array_equal(output, [[203, 7, 5, 3, 1, 1, 197]])
    assert output.dtype == np.uint8


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_large_finite_values_saturate_even_when_scaled_values_overflow(dtype):
    limit = np.finfo(dtype).max
    block = builtin_registry().create("cv.scale_absolute", parameters={"scale": 1e100})
    result = block.execute({"image": np.array([[-limit, -1e10, 0, 1e10, limit]], dtype=dtype)})
    assert result is not None and result.passed, block.error
    np.testing.assert_array_equal(result.outputs["image"], [[255, 255, 0, 255, 255]])


def test_rounds_bounded_magnitudes_at_half_integer_values():
    block = builtin_registry().create("cv.scale_absolute")
    result = block.execute({"image": np.array([[-127.5, 128.5, 254.5, -255.5]])})
    assert result is not None, block.error
    np.testing.assert_array_equal(result.outputs["image"], [[128, 128, 254, 255]])


def test_color_selection_zeroes_pixels_outside_mask_and_region():
    spatial = frame([2, 2], "camera")
    pixels = np.array(
        [[[1, -2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]], dtype=np.int16
    )
    source = ImageArray(pixels, "rgba", {"spatial": spatial})
    mask = np.array([[255, 0], [255, 255]], dtype=np.uint8)
    region = {
        "model": "image_region_v1",
        "image_size": [2, 2],
        "shape": "rectangle",
        "rectangle": [0, 0, 2, 1],
    }
    block = builtin_registry().create("cv.scale_absolute", parameters={"scale": 2})
    result = block.execute({"image": source, "mask": mask, "region": region})
    assert result is not None and result.passed, block.error
    output = result.outputs["image"]
    expected = np.zeros_like(pixels, dtype=np.uint8)
    expected[0, 0] = [2, 4, 6, 8]
    np.testing.assert_array_equal(output, expected)
    assert output.color_format == "rgba"
    assert metadata(output)["spatial"] == spatial


def test_nonfinite_wire_override_fails_without_emitting_an_image():
    block = builtin_registry().create("cv.scale_absolute")
    assert block.execute({"image": np.ones((1, 1), np.uint8), "offset": np.inf}) is None
    assert "finite" in block.error
    assert not block.current_emissions.get("image")


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_example_thresholds_both_gradient_polarities(profile):
    project = (
        Path(__file__).parents[1]
        / "examples/image_processing"
        / "scale_absolute"
        / "scale_absolute.json"
    )
    graph = Graph.load(project, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    dx = graph.blocks["sobel"].last_result.outputs["dx"]
    magnitude = graph.blocks["magnitude"].last_result.outputs["image"]
    expected = np.clip(np.rint(np.abs(dx.astype(float) * 0.75)), 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(magnitude, expected)
    selection = graph.blocks["strong"].last_result.outputs
    mask = selection["mask"] != 0
    assert mask[dx < 0].any() and mask[dx > 0].any()
    np.testing.assert_array_equal(mask, expected >= 120)
    assert selection["count"] == int(mask.sum())
