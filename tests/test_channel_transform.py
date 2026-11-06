"""Exercise linear channel transforms through the registered block."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import ValuePacket, frame


def test_selected_rgba_mixing_clips_integer_range_and_preserves_alpha_and_context():
    source = ImageArray(np.array([[[200, 100, 50, 17], [10, 20, 30, 99]]], np.uint8), "rgba")
    spatial = frame([2, 1], "camera")
    block = builtin_registry().create(
        "cv.channel_transform",
        parameters={"matrix": [[1, 1, 0, 10], [0, 1, 0, -105], [0, 0, 1, 0]]},
    )
    result = block.execute(
        {"image": ValuePacket(source, {"spatial": spatial}), "mask": np.array([[1, 0]], np.uint8)}
    )
    assert result is not None, block.error
    np.testing.assert_array_equal(result.outputs["image"], [[[255, 0, 50, 17], [10, 20, 30, 99]]])
    assert image_format(result.outputs["image"]) == "rgba"
    assert result.payloads["image"]["spatial"] == spatial


def test_wired_matrix_generates_signed_mono_and_zeroes_unselected_pixels():
    source = ImageArray(np.array([[[15, 10, 20], [12, 20, 5]]], np.int16), "bgr")
    registry = builtin_registry()
    region = registry.create("region.shape", parameters={"rectangle": [0, 0, 1, 1]})
    selection = region.execute({"image": source})
    assert selection is not None, region.error
    block = registry.create("cv.channel_transform")
    result = block.execute(
        {
            "image": source,
            "matrix": [[1, 0, -1, 2]],
            "region": selection.outputs["region"],
        }
    )
    assert result is not None, block.error
    output = result.outputs["image"]
    assert image_format(output) == "gray" and output.dtype == np.int16
    np.testing.assert_array_equal(output, [[-3, 0]])


def test_channel_transform_reports_invalid_matrix_without_interrupting_other_runs():
    block = builtin_registry().create("cv.channel_transform")
    source = np.array([[0.25, 0.5]], np.float32)
    with pytest.raises(ValueError, match="finite"):
        block.validate_parameters({"matrix": [[float("nan")]]})
    assert block.execute({"image": source, "matrix": [[1, 0, 1]]}) is None
    assert "Each row needs 1 input weight" in block.error
    result = block.execute({"image": source, "matrix": [[2, -0.75]]})
    assert result is not None, block.error
    np.testing.assert_array_equal(result.outputs["image"], [[-0.25, 0.25]])
