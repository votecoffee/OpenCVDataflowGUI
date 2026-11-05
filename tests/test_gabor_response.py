"""Check the Gabor kernel contract, native signed response and orientation selection."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.payloads import ValuePacket, frame


def test_signed_gabor_exposes_native_kernel_and_retains_selected_frame():
    source = np.arange(80, dtype=np.uint16).reshape(8, 10) * 40
    mask = np.zeros_like(source, dtype=np.uint8)
    mask[:, 4:] = 255
    context = {"spatial": frame([10, 8], "source")}
    block = builtin_registry().create(
        "cv.gabor_response",
        parameters={
            "width": 7,
            "height": 5,
            "sigma": 2,
            "orientation_degrees": 30,
            "wavelength": 4,
            "aspect_ratio": 0.8,
            "phase_degrees": 60,
            "mode": "signed",
            "normalization": "none",
            "output_depth": "float64",
            "border": "replicate",
        },
    )
    result = block.execute({"image": ValuePacket(source, context), "mask": mask})
    assert result is not None, block.error
    kernel = cv2.getGaborKernel((7, 5), 2, np.pi / 6, 4, 0.8, np.pi / 3, cv2.CV_64F)
    expected = cv2.filter2D(source.astype(np.float64), -1, kernel, borderType=cv2.BORDER_REPLICATE)
    expected[mask == 0] = 0
    np.testing.assert_allclose(result.outputs["response"], expected)
    assert result.outputs["response"].dtype == np.float64
    assert set(result.outputs["kernels"]) == {"primary"}
    np.testing.assert_allclose(result.outputs["kernels"]["primary"], kernel)
    assert result.payloads["response"]["spatial"] == context["spatial"]
    assert "spatial" not in result.payloads["kernels"]


def test_gabor_energy_selects_orientation_and_removes_constant_response():
    y, x = np.indices((80, 80))
    stripes = (128 + 60 * np.cos(2 * np.pi * x / 8)).astype(np.float32)
    block = builtin_registry().create("cv.gabor_response")
    matched = block.execute({"image": stripes})
    assert matched is not None, block.error
    assert matched.outputs["response"].dtype == np.float32
    for kernel in matched.outputs["kernels"].values():
        assert np.sum(kernel) == pytest.approx(0, abs=1e-7)
        assert np.linalg.norm(kernel) == pytest.approx(1, abs=1e-7)
    block.parameters["orientation_degrees"] = 90.0
    crossed = block.execute({"image": stripes})
    assert crossed is not None, block.error
    interior = np.s_[16:-16, 16:-16]
    assert (
        matched.outputs["response"][interior].mean()
        > 10 * crossed.outputs["response"][interior].mean()
    )
    flat = block.execute({"image": np.full_like(y, 100, dtype=np.float32)})
    assert flat is not None, block.error
    assert np.max(flat.outputs["response"]) < 1e-4


def test_gabor_rejects_even_size_and_unsafe_float32_conversion():
    block = builtin_registry().create("cv.gabor_response", parameters={"width": 6})
    assert block.execute({"image": np.ones((8, 8), np.uint8)}) is None
    assert block.error_kind == "invalid"
    block.parameters["width"] = 7
    assert block.execute({"image": np.full((8, 8), 1e100)}) is None
    assert "exceeds float32 range" in block.error
