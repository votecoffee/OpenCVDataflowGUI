"""Validate HDR response recovery, native display operators and wire contracts."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import FlowGroup, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock


@pytest.mark.parametrize(
    ("method", "settings"), [("debevec", {"samples": 36}), ("robertson", {"max_iterations": 30})]
)
def test_calibrated_merge_recovers_relative_radiance_on_the_reference_grid(method, settings):
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
    calibration = registry.create(f"cv.calibrate_{method}", parameters={
        **settings, "exposure_times": times,
    })
    fitted = calibration.execute({"exposures": batch})
    assert fitted is not None, calibration.error
    response = fitted.outputs["response"]
    assert response["encoding"] == "bgr" and np.asarray(response["values"]).shape == (256, 3)
    merge = registry.create(f"cv.merge_{method}", parameters={"reference_index": 1})
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


def test_robertson_preserves_gray_and_fills_unobserved_levels():
    registry = builtin_registry()
    base = np.tile(np.arange(16, 256, 8, dtype=np.uint8), (8, 1))
    frames = [base // 2, base, (base.astype(np.int16) * 2).clip(0, 255).astype(np.uint8)]
    times = np.float32([0.5, 1.0, 2.0])
    batch = FlowGroup(tuple(ValuePacket(image, {}) for image in frames), ("low", "mid", "high"))
    calibration = registry.create("cv.calibrate_robertson", parameters={"max_iterations": 12})
    fitted = calibration.execute({"exposures": batch, "times": times.tolist()})
    assert fitted is not None, calibration.error
    response = fitted.outputs["response"]
    assert response["encoding"] == "gray"
    observed = np.unique(frames)
    assert fitted.outputs["info"]["observed_levels"] == [len(observed)]
    native = [cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) for image in frames]
    raw = cv2.createCalibrateRobertson(12, 0.01).process(native, times)[:, 0, 0]
    lut = np.asarray(response["values"], dtype=np.float32)[:, 0]
    np.testing.assert_array_equal(lut[observed], raw[observed])
    np.testing.assert_allclose(lut, np.interp(np.arange(256), observed, raw[observed]))
    assert lut[128] == 1

    merge = registry.create("cv.merge_robertson")
    merged = merge.execute({"exposures": batch, "times": times.tolist(), "response": response})
    assert merged is not None, merge.error
    image = merged.outputs["image"]
    assert image.dtype == np.float32 and image.shape == base.shape and image.color_format == "gray"
    expected = cv2.createMergeRobertson().process(
        native, times, np.repeat(lut[:, None, None], 3, axis=2)
    )
    np.testing.assert_array_equal(image, expected[..., 0])

    black_response = {**response, "values": [[0.0], *response["values"][1:]]}
    assert merge.execute({"exposures": batch, "times": times.tolist(), "response": black_response})
    bad_response = {**response, "model": "opencv_debevec_response_v1"}
    result = merge.execute({"exposures": batch, "times": times.tolist(), "response": bad_response})
    assert result is None
    assert "Connect a Robertson camera response" in merge.error


def test_robertson_requires_a_measured_midpoint_before_calling_native(monkeypatch):
    def unexpected(*_args):
        pytest.fail("Missing midpoint reached the native calibrator")

    monkeypatch.setattr(cv2, "createCalibrateRobertson", unexpected)
    dark = np.arange(64, dtype=np.uint8).reshape(8, 8)
    batch = FlowGroup((ValuePacket(dark, {}), ValuePacket(dark + 40, {})), ("a", "b"))
    block = builtin_registry().create("cv.calibrate_robertson")
    assert block.execute({"exposures": batch, "times": [0.5, 1.0]}) is None
    assert "needs intensity 128" in block.error


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
