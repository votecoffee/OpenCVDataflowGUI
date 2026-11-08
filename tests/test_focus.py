"""Validate focus scores, selected response maps, and the inclusive limit."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock


@pytest.mark.parametrize("method", ["laplacian_variance", "tenengrad"])
def test_focus_matches_native_derivatives_on_mask_and_region(method):
    rng = np.random.default_rng(29)
    image = rng.integers(0, 4096, (30, 40), dtype=np.uint16)
    color = ImageArray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), "rgb")
    ticket = TicketClock().issue("camera")
    packet = ValuePacket(color, {"ticket": ticket, "spatial": frame([40, 30], "fixture")})
    mask = np.zeros(image.shape, np.uint8)
    mask[4:24, 3:27] = 255
    region = {
        "model": "image_region_v1", "image_size": [40, 30],
        "shape": "rectangle", "rectangle": [7, 6, 14, 20],
    }
    selected = np.zeros(image.shape, dtype=bool)
    selected[6:24, 7:21] = True
    native = image.astype(np.float64)
    options = {"ksize": 5, "borderType": cv2.BORDER_REPLICATE}
    if method == "laplacian_variance":
        response = cv2.Laplacian(native, cv2.CV_64F, **options)
        expected = response[selected].var()
        response = np.abs(response)
    else:
        dx = cv2.Sobel(native, cv2.CV_64F, 1, 0, **options)
        dy = cv2.Sobel(native, cv2.CV_64F, 0, 1, **options)
        response = dx**2 + dy**2
        expected = response[selected].mean()
    inputs = {"image": packet, "mask": mask, "region": region}
    block = builtin_registry().create("inspect.focus", parameters={
        "method": method, "aperture": "5", "border": "replicate", "emit_response": True,
    })
    result = block.execute(inputs)
    assert result is not None, block.error
    score = result.outputs["sharpness"]
    assert score == pytest.approx(expected, rel=1e-12)
    assert result.outputs["measurements"]["selected_pixels"] == int(selected.sum())
    assert result.payloads["sharpness"]["ticket"] == ticket
    assert result.payloads["response"]["spatial"]["root_id"] == "fixture"
    assert result.outputs["image"].dtype == np.uint16
    assert result.outputs["image"].color_format == "rgb"
    np.testing.assert_array_equal(result.outputs["response"], np.where(selected, response, 0))
    np.testing.assert_array_equal(result.outputs["image"], color)
    assert "focus measurement" in result.drawings

    block.parameters.update(check_threshold=True, minimum_sharpness=score, emit_response=False)
    accepted = block.execute(inputs)
    assert accepted.passed and accepted.outputs["response"] is NO_VALUE
    block.parameters["minimum_sharpness"] = float(np.nextafter(score, np.inf))
    rejected = block.execute(inputs)
    assert not rejected.passed
    assert rejected.fail_reasons[0]["code"] == "focus_below_minimum"


def test_focus_rejects_empty_selection_and_coordinate_mismatch():
    image = np.zeros((8, 12), dtype=np.float32)
    block = builtin_registry().create("inspect.focus")
    assert block.execute({"image": image, "mask": np.zeros(image.shape, np.uint8)}) is None
    assert "two selected pixels" in block.error
    source = ValuePacket(image, {"spatial": frame([12, 8], "source")})
    other = ValuePacket(np.ones(image.shape, np.uint8), {"spatial": frame([12, 8], "other")})
    assert block.execute({"image": source, "mask": other}) is None
    assert "coordinate frame differs" in block.error
