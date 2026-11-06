"""Verify OpenCV and standard Hann windows through the spectral graph ports."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.payloads import ValuePacket, frame


@pytest.mark.parametrize("window", ["hann", "sqrt_hann"])
def test_hann_window_uses_native_coefficients_and_dft_scale(window):
    image = np.arange(63, dtype=np.uint16).reshape(7, 9) * 901
    spatial = frame([9, 7], "source")
    block = builtin_registry().create(
        "cv.hann_window",
        parameters={
            "window": window,
            "precision": "float64",
        },
    )
    result = block.execute({"image": ValuePacket(image, {"spatial": spatial})})
    assert result is not None, block.error
    expected = cv2.createHanningWindow((9, 7), cv2.CV_64F)
    if window == "hann":
        expected = expected**2
    np.testing.assert_allclose(result.outputs["window"], expected)
    np.testing.assert_allclose(result.outputs["image"], image.astype(np.float64) / 65535 * expected)
    assert result.outputs["coherent_gain"] == pytest.approx(expected.mean())
    for name in ("image", "window"):
        assert result.outputs[name].dtype == np.float64
        assert result.payloads[name]["spatial"] == spatial


def test_row_hann_reduces_off_bin_leakage_and_accepts_single_row():
    registry = builtin_registry()
    x = np.arange(128)
    source = np.sin(2 * np.pi * 6.35 * x / 128)[None, :].astype(np.float32)
    window = registry.create("cv.hann_window", parameters={"axes": "rows"})
    result = window.execute({"image": source})
    assert result is not None, window.error
    assert result.outputs["image"].dtype == np.float32
    np.testing.assert_allclose(result.outputs["window"][0], np.hanning(128), atol=1e-7)
    dft = registry.create("cv.dft", parameters={"mode": "rows"})
    outside = np.abs(np.fft.fftfreq(128) * 128) > 12
    leakage = []
    for image in (source, result.outputs["image"]):
        spectrum = dft.execute({"image": image}).outputs["spectrum"]
        power = np.square(spectrum.values).sum(axis=-1)
        leakage.append(float(power[:, outside].sum() / power.sum()))
    assert leakage[1] < leakage[0] / 20

    window.parameters["axes"] = "2d"
    assert window.execute({"image": source}) is None
    assert "at least three samples" in window.error
