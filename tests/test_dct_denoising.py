"""Qualify border coverage, tiled native parity, and selected writes."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock

TYPE = "cv.dct_denoise"
EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/dct_denoising"


def tool(**parameters):
    return builtin_registry().create(TYPE, parameters=parameters)


def full_native(source, sigma, patch):
    padded = cv2.copyMakeBorder(source, 0, 1, 0, 1, cv2.BORDER_REPLICATE)
    filtered = np.empty_like(padded)
    cv2.xphoto.dctDenoising(padded, filtered, sigma, patch)
    return filtered[:-1, :-1]


def test_tiling_matches_all_native_windows_and_bounds_patch_storage(monkeypatch):
    rng = np.random.default_rng(746)
    source = rng.integers(62, 133, size=(78, 74), dtype=np.uint8)
    patch = 32
    expected = full_native(source, 10, patch)
    native = cv2.xphoto.dctDenoising
    patches = []

    def observe(padded, filtered, sigma, size):
        rows, cols = padded.shape
        patches.append((rows - size) * (cols - size) * size**2 * 4)
        native(padded, filtered, sigma, size)

    monkeypatch.setattr(cv2.xphoto, "dctDenoising", observe)
    block = tool(patch_size=patch)
    result = block.execute({"image": source})
    assert result and result.passed, block.error
    output = result.outputs["image"]
    assert len(patches) == 4 and max(patches) <= 36 * 1024**2
    assert output[-1, -1] > 0
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("dtype, scale", [(np.uint16, 65535), (np.float32, 1), (np.float64, 1)])
def test_selection_copies_native_estimates_only_inside_intersection(dtype, scale):
    rng = np.random.default_rng(612)
    height, width = 38, 43
    bgr = (rng.uniform(0.2, 0.7, (height, width, 3)) * scale).astype(dtype)
    alpha = np.linspace(0, scale, height * width, dtype=dtype).reshape(height, width)
    rgb = ImageArray(np.dstack((bgr[..., ::-1], alpha)), "rgba")
    mask = np.zeros((height, width), np.uint8)
    mask[9:, 3:31] = 4
    context = frame([width, height], "inspection")
    ticket = TicketClock("denoise").issue("camera", "fixed", "part")
    region = {
        "model": "image_region_v1",
        "image_size": [width, height],
        "shape": "rectangle",
        "rectangle": [19, 13, 24, 25],
        "coordinate_context": context,
    }
    block = tool(patch_size=8)
    result = block.execute(
        {
            "image": ValuePacket(rgb, {"spatial": context, "ticket": ticket}),
            "mask": ValuePacket(mask, {"spatial": context}),
            "region": region,
        }
    )
    assert result and result.passed, block.error
    selected = np.zeros_like(mask, dtype=bool)
    selected[13:, 19:31] = True
    reference = full_native(bgr, 10 * scale / 255, 8)[..., ::-1]
    if scale == 1:
        reference = reference.clip(0, 1)
    output = result.outputs["image"]
    assert output.dtype == dtype and image_format(output) == "rgba"
    np.testing.assert_array_equal(output[..., :3][selected], reference[selected])
    np.testing.assert_array_equal(output[~selected], rgb[~selected])
    np.testing.assert_array_equal(output[..., 3], alpha)
    assert result.payloads["image"]["spatial"] == context
    assert result.payloads["image"]["ticket"] == ticket


def test_parameter_and_input_validation_and_bypass():
    for name, value in (("patch_size", 3), ("patch_size", 34), ("noise_sigma", -1)):
        with pytest.raises(ValidationError) as error:
            tool(**{name: value}).validate_parameters()
        assert error.value.fields == (name,)
    image = np.full((8, 8), 120, np.uint8)
    node = tool()
    assert node.execute({"image": image}) is None
    assert "at least the patch size" in node.error
    bypass = tool(noise_sigma=0).execute({"image": image})
    np.testing.assert_array_equal(bypass.outputs["image"], image)
    bypass = node.execute({"image": image, "mask": np.zeros_like(image)})
    np.testing.assert_array_equal(bypass.outputs["image"], image)
    result = tool(patch_size=8).execute({"image": image})
    np.testing.assert_array_equal(result.outputs["image"], image)
    assert node.execute({"image": image.astype(np.float32)}) is None
    assert "in [0, 1]" in node.error
    assert node.execute({"image": image, "mask": image[:2]}) is None
    assert "matching image dimensions" in node.error


@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_project_reduces_noise_and_keeps_edge_contrast(profile):
    graph = Graph.load(EXAMPLE / "dct_denoising.json", builtin_registry())
    graph.set_run_profile(profile)
    graph.set_branch_workers(2)
    try:
        graph.run()
        clean = cv2.imread(str(EXAMPLE / "assets/clean.png")).astype(float)
        noisy = graph.blocks["noisy"].last_result.outputs["image"]
        result = graph.blocks["full"].last_result
        output = result.outputs["image"].astype(float)
        assert np.sqrt(np.mean((noisy - clean) ** 2)) > 12
        assert np.sqrt(np.mean((output - clean) ** 2)) < 4.1
        assert np.std(output[8:32, 8:45, 0]) < 1
        edge_profile = output[8:32, 50:64].mean(axis=(0, 2))
        assert np.argmax(np.diff(edge_profile)) == 5
        contrast = np.mean(output[8:32, 56:59] - output[8:32, 53:56])
        assert contrast == pytest.approx(90.0, abs=1)
        mask = graph.blocks["area"].last_result.outputs["mask"] != 0
        limited = graph.blocks["selected"].last_result.outputs["image"]
        np.testing.assert_array_equal(limited[mask], output[mask])
        np.testing.assert_array_equal(limited[~mask], noisy[~mask])
    finally:
        graph.close()
