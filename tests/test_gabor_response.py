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


def test_gabor_bank_matches_individual_responses_and_preserves_selection():
    registry = builtin_registry()
    y, x = np.indices((72, 144))
    coordinate = np.where(x < 72, x, y)
    source = (128 + 60 * np.cos(2 * np.pi * coordinate / 8)).astype(np.float32)
    mask = np.full(source.shape, 255, np.uint8)
    mask[:10] = 0
    context = {"spatial": frame([144, 72], "stripes")}
    packet = ValuePacket(source, context)
    bank = registry.create("cv.gabor_bank", parameters={"orientations_degrees": [0, 90]})
    result = bank.execute({"image": packet, "mask": mask})
    assert result is not None, bank.error
    vertical, horizontal = [
        registry.create("cv.gabor_response", parameters={"orientation_degrees": angle})
        .execute({"image": packet})
        .outputs["response"]
        for angle in (0, 90)
    ]
    np.testing.assert_allclose(
        result.outputs["response"], np.where(mask, np.maximum(vertical, horizontal), 0)
    )
    left = np.s_[16:-16, 16:56]
    right = np.s_[16:-16, 88:-16]
    assert np.all(result.outputs["orientation"][left] == 0)
    assert np.all(result.outputs["orientation"][right] == 90)
    assert np.all(result.outputs["valid_mask"][left] == 255)
    assert np.all(result.outputs["valid_mask"][right] == 255)
    expected_confidence = 1 - horizontal[left] / vertical[left]
    np.testing.assert_allclose(result.outputs["confidence"][left], expected_confidence)
    assert all(np.count_nonzero(result.outputs[name][:10]) == 0 for name in result.outputs)
    assert result.payloads["orientation"]["spatial"] == context["spatial"]
    assert result.payloads["orientation"]["angle_units"] == "degrees"


def test_gabor_bank_flags_flat_and_weak_patches_and_validates_angles():
    registry = builtin_registry()
    bank = registry.create("cv.gabor_bank")
    image = np.full((48, 48), 128, np.uint8)
    flat = bank.execute({"image": image})
    assert flat is not None, bank.error
    assert np.count_nonzero(flat.outputs["valid_mask"]) == 0
    assert np.count_nonzero(flat.outputs["confidence"]) == 0

    bank.parameters["minimum_response"] = 1e6
    image[10:30, 10:30] = 32
    weak = bank.execute({"image": image})
    assert weak is not None, bank.error
    assert np.max(weak.outputs["response"]) > 0
    assert not np.any(weak.outputs["valid_mask"])

    for angles in ([0], [0, 0], [0, 180], [0, float("nan")], [False, 90]):
        bank.parameters["orientations_degrees"] = angles
        assert bank.execute({"image": image}) is None
        assert bank.error_kind == "invalid"
        assert "distinct finite orientation angles" in bank.error
