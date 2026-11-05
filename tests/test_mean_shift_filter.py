"""Check mean-shift settings, native output, alpha and selected writes."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import ValuePacket, frame


@pytest.mark.parametrize("level", [0, 1, 2])
def test_mean_shift_matches_native_and_keeps_alpha_and_selection(level):
    rng = np.random.default_rng(2)
    image = rng.integers(45, 65, (24, 36, 4), dtype=np.uint8)
    image[:, 18:, :3] += 100
    source = ImageArray(image, "rgba")
    payload = {"spatial": frame([36, 24], "camera")}
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[:, 4:30] = 255
    block = builtin_registry().create(
        "cv.mean_shift_filter",
        parameters={
            "spatial_radius": 4,
            "color_radius": 32,
            "pyramid_level": level,
            "max_iterations": 7,
            "epsilon": 0.5,
        },
    )
    result = block.execute({"image": ValuePacket(source, payload), "mask": mask})
    assert result is not None, block.error
    expected = cv2.pyrMeanShiftFiltering(
        np.ascontiguousarray(image[..., :3]),
        4,
        32,
        maxLevel=level,
        termcrit=(cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 7, 0.5),
    )
    expected[mask == 0] = image[..., :3][mask == 0]
    output = result.outputs["image"]
    np.testing.assert_array_equal(output[..., :3], expected)
    np.testing.assert_array_equal(output[..., 3], image[..., 3])
    assert image_format(output) == "rgba"
    assert result.payloads["image"]["spatial"] == payload["spatial"]


def test_mean_shift_grayscale_keeps_its_type_and_rejects_unsupported_depths():
    source = np.tile(np.array([50, 58, 50, 58], np.uint8), (12, 6))
    block = builtin_registry().create("cv.mean_shift_filter", parameters={"pyramid_level": 0})
    result = block.execute({"image": source})
    assert result is not None, block.error
    output = result.outputs["image"]
    assert output.dtype == np.uint8 and output.ndim == 2
    assert float(output.std()) < float(source.std())
    assert block.execute({"image": source.astype(np.uint16)}) is None
    assert "Unsupported image depth" in block.error
    block.parameters["pyramid_level"] = 9
    assert block.execute({"image": source}) is None
    assert block.error_kind == "invalid"
