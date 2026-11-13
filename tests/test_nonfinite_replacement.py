"""Check the replacement block's modes, selections, and typed output."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import DepthArray, ImageArray
from opencv_dataflow.payloads import frame, metadata


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_replaces_nonfinite_channels_individually_and_preserves_encoding(dtype):
    pixels = np.array([[[np.nan, np.inf, -np.inf, 1], [0.25, 0.5, np.nan, 1]]], dtype=dtype)
    spatial = frame([2, 1], "camera")
    image = ImageArray(pixels, "rgba", {"spatial": spatial})
    block = builtin_registry().create(
        "cv.replace_nonfinite",
        parameters={
            "nan_replacement": 0.125,
            "positive_infinity_replacement": 0.75,
            "negative_infinity_replacement": -0.5,
        },
    )
    before = pixels.copy()
    result = block.execute({"image": image})
    assert result is not None and result.passed, block.error
    output = result.outputs["image"]
    np.testing.assert_array_equal(output, [[[0.125, 0.75, -0.5, 1], [0.25, 0.5, 0.125, 1]]])
    np.testing.assert_array_equal(pixels, before)
    assert output.dtype == dtype and output.color_format == "rgba"
    assert metadata(output)["spatial"] == spatial
    np.testing.assert_array_equal(result.outputs["replaced_mask"], [[255, 255]])
    assert metadata(result.outputs["replaced_mask"])["spatial"] == spatial
    assert result.outputs["replaced_pixel_count"] == 2
    assert result.outputs["report"] == {
        "replaced_samples": 4,
        "by_kind": {"nan": 2, "positive_infinity": 1, "negative_infinity": 1},
    }


@pytest.mark.parametrize(
    ("mode", "expected"),
    (("nan_only", [2, np.inf, -np.inf]), ("infinity_only", [np.nan, 3, -4])),
)
def test_modes_leave_other_nonfinite_values_in_place(mode, expected):
    block = builtin_registry().create(
        "cv.replace_nonfinite",
        parameters={
            "mode": mode,
            "nan_replacement": 2,
            "positive_infinity_replacement": 3,
            "negative_infinity_replacement": -4,
        },
    )
    result = block.execute({"image": np.array([[np.nan, np.inf, -np.inf]])})
    assert result is not None, block.error
    np.testing.assert_array_equal(result.outputs["image"], [expected])
    np.testing.assert_array_equal(result.outputs["replaced_mask"] != 0, [np.isfinite(expected)])


def test_selected_color_repair_changes_only_mask_and_region_intersection():
    image = np.array(
        [[[np.nan, np.inf, -np.inf], [np.nan, 0, 1]], [[np.nan, 0, 1], [np.nan, 1, 0]]],
        dtype=np.float64,
    )
    mask = np.array([[255, 0], [255, 255]], dtype=np.uint8)
    region = {
        "model": "image_region_v1",
        "image_size": [2, 2],
        "shape": "rectangle",
        "rectangle": [0, 0, 2, 1],
    }
    block = builtin_registry().create("cv.replace_nonfinite")
    result = block.execute({"image": image, "mask": mask, "region": region})
    assert result is not None, block.error
    expected = image.copy()
    expected[0, 0] = 0
    np.testing.assert_array_equal(result.outputs["image"], expected)
    np.testing.assert_array_equal(result.outputs["replaced_mask"], [[255, 0], [0, 0]])
    assert result.outputs["replaced_pixel_count"] == 1
    assert result.outputs["report"]["replaced_samples"] == 3


def test_replacement_outside_float32_range_fails_with_no_image_event():
    block = builtin_registry().create("cv.replace_nonfinite", parameters={"nan_replacement": 1e40})
    assert block.execute({"image": np.array([[np.nan]], dtype=np.float32)}) is None
    assert "exceeds the float32 range" in block.error
    assert not block.current_emissions.get("image")


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_example_retains_depth_and_revalidates_all_pixels(profile):
    project = (
        Path(__file__).parents[1]
        / "examples/opencv tools"
        / "nonfinite_replacement"
        / "nonfinite_replacement.json"
    )
    graph = Graph.load(project, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    original = graph.blocks["stereo"].last_result.outputs["depth"]
    outputs = graph.blocks["replace"].last_result.outputs
    assert np.isnan(original[:10, :10]).all()
    assert isinstance(outputs["image"], DepthArray)
    assert (outputs["image"].units, outputs["image"].display_max) == ("m", original.display_max)
    assert metadata(outputs["image"])["spatial"] == metadata(original)["spatial"]
    np.testing.assert_array_equal(outputs["image"], np.full((64, 64), 3, dtype=np.float32))
    assert graph.blocks["before"].last_result.outputs["invalid_count"] == 100
    assert outputs["replaced_pixel_count"] == 100
    assert graph.blocks["after"].last_result.outputs["invalid_count"] == 0
