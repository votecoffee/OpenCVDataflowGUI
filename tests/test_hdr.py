"""Validate HDR response recovery, native display operators and wire contracts."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import FlowGroup, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock


def test_calibrated_merge_recovers_relative_radiance_on_the_reference_grid():
    registry = builtin_registry()
    ramp = np.linspace(0.025, 3.5, 32 * 48, dtype=np.float32).reshape(32, 48)
    reference = np.stack((ramp * 0.3, ramp * 0.8, ramp), axis=-1)
    times = [0.035, 0.11, 0.4, 1.2]
    entries = []
    for index, seconds in enumerate(times):
        pixels = np.rint(np.sqrt(np.minimum(reference * seconds, 1)) * 255).astype(np.uint8)
        entries.append(ValuePacket(ImageArray(pixels, "bgr"), {
            "spatial": frame([48, 32], f"exposure_{index}"),
        }))
    group = FlowGroup(tuple(entries), ("a", "b", "c", "d"))
    ticket = TicketClock().issue("batch")
    batch = ValuePacket(group, {"ticket": ticket})
    calibration = registry.create("cv.calibrate_debevec", parameters={
        "samples": 36, "exposure_times": times,
    })
    fitted = calibration.execute({"exposures": batch})
    assert fitted is not None, calibration.error
    response = fitted.outputs["response"]
    assert response["encoding"] == "bgr" and np.asarray(response["values"]).shape == (256, 3)
    merge = registry.create("cv.merge_debevec", parameters={"reference_index": 1})
    linear = merge.execute({"exposures": batch, "times": times})
    hdr = merge.execute({"exposures": batch, "times": times, "response": response})
    assert linear is not None and hdr is not None, merge.error
    assert hdr.payloads["image"]["ticket"] == ticket
    assert hdr.payloads["image"]["spatial"]["root_id"] == "exposure_1"
    assert hdr.outputs["info"]["response"] == "calibrated"
    image = hdr.outputs["image"]
    assert image.dtype == np.float32 and image.color_format == "bgr" and image.max() > 1

    def relative_error(result):
        ratio = np.asarray(result.outputs["image"]) / reference
        return np.quantile(np.abs(ratio / np.median(ratio, axis=(0, 1)) - 1), 0.95)

    assert relative_error(hdr) < 0.08
    assert relative_error(linear) > 0.2
    rgb_response = {
        **response, "encoding": "rgb", "values": [row[::-1] for row in response["values"]],
    }
    switched = merge.execute({"exposures": batch, "times": times, "response": rgb_response})
    np.testing.assert_array_equal(switched.outputs["image"], image)


def test_bad_times_and_response_fail_before_native_merge(monkeypatch):
    def unexpected():
        pytest.fail("Invalid inputs reached native HDR merge")

    monkeypatch.setattr(cv2, "createMergeDebevec", unexpected)
    pixels = np.arange(128, dtype=np.uint8).reshape(8, 16)
    group = FlowGroup(tuple(ValuePacket(pixels, {}) for _ in range(3)), ("a", "b", "c"))
    merge = builtin_registry().create("cv.merge_debevec")
    assert merge.execute({"exposures": group, "times": [0.1, 0.2]}) is None
    assert "time count" in merge.error
    assert merge.execute({"exposures": group, "times": [0.1, 0.1, 0.1]}) is None
    assert "distinct exposure times" in merge.error
    malformed = {"model": "opencv_debevec_response_v1", "encoding": "gray",
                 "values": [[1.0]] * 255 + [[0.0]]}
    assert merge.execute({
        "exposures": group, "times": [0.1, 0.2, 0.3], "response": malformed,
    }) is None
    assert "positive values" in merge.error


@pytest.mark.parametrize("method", ["linear", "drago", "reinhard", "mantiuk"])
def test_tone_map_preserves_gray_encoding_and_handles_black_samples(method):
    image = np.linspace(0.007, 20, 32 * 48, dtype=np.float32).reshape(32, 48)
    before = image.copy()
    block = builtin_registry().create("cv.tonemap_hdr", parameters={
        "method": method, "gamma": 1.5, "output_depth": "float32", "color_adaptation": 1.0,
    })
    result = block.execute({"image": image})
    assert result is not None, block.error
    output = result.outputs["image"]
    assert result.outputs["info"]["method"] == method
    assert output.dtype == np.float32 and output.color_format == "gray"
    assert np.isfinite(output).all() and 0 <= output.min() < 1e-6
    assert 0.99 <= output.max() <= 1
    np.testing.assert_array_equal(image, before)


def test_linear_tone_map_matches_native_for_rgb():
    pixels = np.linspace(0, 12, 8 * 12 * 3, dtype=np.float32).reshape(8, 12, 3)
    block = builtin_registry().create("cv.tonemap_hdr", parameters={"gamma": 2.2})
    result = block.execute({"image": ImageArray(pixels, "rgb")})
    assert result is not None, block.error
    native = cv2.createTonemap(2.2).process(pixels)
    expected = np.rint(np.clip(native, 0, 1) * 255).astype(np.uint8)
    assert result.outputs["image"].color_format == "rgb"
    np.testing.assert_array_equal(result.outputs["image"], expected)
