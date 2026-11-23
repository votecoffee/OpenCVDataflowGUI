"""Native boundary handling, precision and step-edge qualification."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

TYPE = "cv.anisotropic_diffusion"
EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/anisotropic_diffusion"


def tool(**parameters):
    return builtin_registry().create(TYPE, parameters=parameters)


def scalar_diffusion(signal, alpha, sigma, passes, quantize):
    result = signal.astype(float)
    height, width = signal.shape[:2]
    for _ in range(passes):
        updated = result.copy()
        for y, x in np.ndindex(height, width):
            center = result[y, x]
            delta = []
            for dy, dx in np.ndindex(3, 3):
                if (dy, dx) == (1, 1):
                    continue
                row = min(max(y + dy - 1, 0), height - 1)
                col = min(max(x + dx - 1, 0), width - 1)
                difference = result[row, col] - center
                factor = np.exp(-((np.abs(difference).mean() / sigma) ** 2))
                delta.append(difference * factor)
            updated[y, x] += alpha * np.sum(delta, axis=0)
        result = np.rint(updated) if quantize else updated
    return result


def test_native_passes_rebuild_borders_and_preserve_metadata_and_selection(monkeypatch):
    pixels = np.random.default_rng(429).integers(30, 215, (28, 47, 4), dtype=np.uint8)
    native = cv2.ximgproc.anisotropicDiffusion
    expected = pixels[..., :3].copy()
    for _ in range(5):
        expected = native(expected, 0.1, 0.25, 1)

    calls = []

    def record(image, alpha, sensitivity, iterations):
        calls.append(iterations)
        return native(image, alpha, sensitivity, iterations)

    monkeypatch.setattr(cv2.ximgproc, "anisotropicDiffusion", record)
    spatial = frame([47, 28], "camera")
    ticket = TicketClock("diffusion").issue("camera", "fixed", "part")
    mask = np.zeros(pixels.shape[:2], np.uint8)
    mask[2:19, 4:38] = 255
    region = {
        "model": "image_region_v1",
        "shape": "rectangle",
        "image_size": [47, 28],
        "rectangle": [11, 4, 32, 9],
    }
    block = tool(iterations=5, edge_sensitivity=0.25)
    result = block.execute(
        {
            "image": ValuePacket(
                ImageArray(pixels, "rgba"), {"ticket": ticket, "spatial": spatial}
            ),
            "mask": ValuePacket(mask, {"spatial": spatial}),
            "region": region,
        }
    )
    assert result and result.passed, block.error
    assert calls == [1] * 5
    selected = np.zeros(pixels.shape[:2], bool)
    selected[4:13, 11:38] = True
    actual = result.outputs["image"]
    assert actual.color_format == "rgba" and actual.dtype == np.uint8
    np.testing.assert_array_equal(actual[selected, :3], expected[selected])
    np.testing.assert_array_equal(actual[~selected], pixels[~selected])
    np.testing.assert_array_equal(actual[..., 3], pixels[..., 3])
    assert result.payloads["image"]["ticket"] == ticket
    assert result.payloads["image"]["spatial"] == spatial


def test_native_multiple_passes_preserve_a_flat_field_at_every_border():
    source = np.full((33, 52), 130, np.uint8)
    block = tool(edge_sensitivity=1.0, iterations=10)
    for _ in range(2):
        result = block.execute({"image": source})
        assert result and result.passed, block.error
        np.testing.assert_array_equal(result.outputs["image"], source)


def test_full_range_byte_color_uses_a_safe_explicit_step(monkeypatch):
    source = np.array(
        [[[0, 255, 0], [255, 0, 255], [0, 255, 0]], [[0, 0, 0], [255, 255, 255], [43, 78, 128]]],
        dtype=np.uint8,
    )

    def unexpected(*args):
        raise AssertionError("Full-range data reached the native byte path")

    monkeypatch.setattr(cv2.ximgproc, "anisotropicDiffusion", unexpected)
    block = tool(time_step=0.11, edge_sensitivity=0.8, iterations=3)
    result = block.execute({"image": source})
    assert result and result.passed, block.error
    expected = scalar_diffusion(source, 0.11, 0.8 * 255, 3, quantize=True)
    np.testing.assert_array_equal(result.outputs["image"], expected)


@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_high_depth_retains_subbyte_changes(dtype):
    codes = np.full((9, 8), 10000, np.uint16)
    codes[4, 4] += 29
    signal = codes if dtype == np.uint16 else (codes.astype(np.float32) / 65535)
    block = tool(iterations=2)
    result = block.execute({"image": signal})
    assert result and result.passed, block.error
    expected = scalar_diffusion(signal, 0.1, 0.12 * (65535 if dtype == np.uint16 else 1), 2, False)
    actual = result.outputs["image"]
    assert actual.dtype == dtype
    if dtype == np.uint16:
        assert actual[4, 4] > 10000 and (actual > 10000).sum() > 1
        np.testing.assert_array_equal(actual, np.rint(expected))
    else:
        np.testing.assert_allclose(actual, expected, atol=3e-8, rtol=0)


def test_validation_noop_and_explicit_work_limit(monkeypatch):
    flat = np.full((12, 13, 3), 90, np.uint8)
    invalid = tool(time_step=0.126)
    assert invalid.execute({"image": flat}) is None and "Time step" in invalid.error
    invalid = tool(edge_sensitivity=0)
    assert invalid.execute({"image": flat}) is None and "Edge sensitivity" in invalid.error
    no_op = tool(iterations=0)
    result = no_op.execute({"image": flat})
    assert result and result.passed
    np.testing.assert_array_equal(result.outputs["image"], flat)
    assert result.outputs["image"] is not flat
    tool_ = tool()
    assert tool_.execute({"image": np.ones((9, 9), np.float32) * 2}) is None
    assert "normalized" in tool_.error
    monkeypatch.setattr("opencv_dataflow.anisotropic_diffusion._MAX_EXPLICIT_SAMPLE_STEPS", 1)
    assert tool_.execute({"image": np.ones((9, 9), np.float32)}) is None
    assert "exceeds the limit" in tool_.error


@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_project_measures_noise_edge_and_region(profile):
    graph = Graph.load(EXAMPLE / "anisotropic_diffusion.json", builtin_registry())
    graph.set_run_profile(profile)
    graph.set_branch_workers(3)
    try:
        graph.run()
        node = graph.blocks["refined"]
        assert block_entry(node).path[:2] == ("OpenCV", "Extended Image Processing (ximgproc)")
        truth = cv2.imread(str(EXAMPLE / "assets/clean.png"), cv2.IMREAD_GRAYSCALE).astype(float)
        results = {}
        for name in ("noisy", "early", "refined", "permissive", "selected"):
            block = graph.blocks[name]
            assert block.last_result and block.last_result.passed, block.error
            results[name] = block.last_result.outputs["image"]
        for name, tolerance in (("early", 7.3), ("refined", 1.9)):
            assert np.sqrt(np.mean((results[name].astype(float) - truth) ** 2)) < tolerance

        def edge(image):
            return float((image[:, 81].astype(float) - image[:, 80]).mean())

        assert edge(results["refined"]) == pytest.approx(121, abs=0.5)
        assert edge(results["permissive"]) < 51
        area = np.zeros(truth.shape, bool)
        area[10:77, 19:141] = True
        np.testing.assert_array_equal(results["selected"][area], results["refined"][area])
        np.testing.assert_array_equal(results["selected"][~area], results["noisy"][~area])
    finally:
        graph.close()
