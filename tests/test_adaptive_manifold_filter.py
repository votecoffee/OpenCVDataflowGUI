"""Qualify OpenCV adaptive manifold filtering through the application registry."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray, convert_depth
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

TYPE = "cv.adaptive_manifold_filter"
EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/adaptive_manifold"


def tool(**parameters):
    return builtin_registry().create(TYPE, parameters=parameters)


@pytest.mark.parametrize("adjust", [False, True])
def test_native_outlier_control_mask_and_region_preserve_encoding(adjust):
    rng = np.random.default_rng(4072)
    pixels = rng.integers(4000, 50000, (44, 52, 4), np.uint16)
    guide = rng.integers(35, 220, (44, 52), np.uint8)
    block = tool(sigma_spatial=5, sigma_color=0.15, adjust_outliers=adjust)
    native = cv2.ximgproc.amFilter(
        convert_depth(guide, "float32"),
        convert_depth(pixels[..., :3], "float32"),
        5,
        0.15,
        adjust_outliers=adjust,
    )
    expected = convert_depth(native, "uint16")
    selected = np.zeros(pixels.shape[:2], bool)
    selected[9:29, 15:35] = True
    mask = np.zeros_like(guide)
    mask[9:42, 6:35] = 255
    region = {
        "model": "image_region_v1",
        "image_size": [52, 44],
        "shape": "rectangle",
        "rectangle": [15, 5, 34, 24],
    }
    spatial = frame([52, 44], "camera")
    ticket = TicketClock("manifold").issue("camera", "fixed", "part-1")
    result = block.execute(
        {
            "image": ValuePacket(
                ImageArray(pixels, "rgba"), {"spatial": spatial, "ticket": ticket}
            ),
            "guide": ValuePacket(guide, {"spatial": spatial}),
            "mask": ValuePacket(mask, {"spatial": spatial}),
            "region": region,
        }
    )
    assert result and result.passed, block.error
    output = result.outputs["image"]
    assert output.dtype == np.uint16 and output.color_format == "rgba"
    np.testing.assert_array_equal(output[selected, :3], expected[selected])
    np.testing.assert_array_equal(output[~selected], pixels[~selected])
    np.testing.assert_array_equal(output[..., 3], pixels[..., 3])
    assert result.payloads["image"]["spatial"] == spatial
    assert result.payloads["image"]["ticket"] == ticket


def test_self_guided_float_and_registered_hdr_signal():
    mono = np.linspace(0.15, 0.7, 35 * 38, dtype=np.float32).reshape(35, 38)
    block = tool(sigma_spatial=4, sigma_color=0.2)
    result = block.execute({"image": mono})
    assert result and result.passed, block.error
    expected = cv2.ximgproc.createAMFilter(4, 0.2).filter(mono)
    np.testing.assert_allclose(result.outputs["image"], expected, atol=2e-7, rtol=0)

    hdr = (mono - np.float32(0.4)) * 320
    result = block.execute({"image": hdr, "guide": mono})
    assert result and result.passed, block.error
    expected = cv2.ximgproc.createAMFilter(4, 0.2).filter(hdr, joint=mono)
    np.testing.assert_allclose(result.outputs["image"], expected, atol=1e-4, rtol=0)


def test_frame_alignment_and_parameter_bounds():
    signal = np.full((8, 9), 64, np.uint8)
    signal_packet = ValuePacket(signal, {"spatial": frame([9, 8], "source")})
    guide_packet = ValuePacket(signal, {"spatial": frame([9, 8], "guide")})
    inputs = {"image": signal_packet, "guide": guide_packet}
    strict = tool()
    assert strict.execute(inputs) is None and "share the source pixel frame" in strict.error
    aligned = tool(guide_alignment="pixel_grid", output_depth="float32")
    result = aligned.execute(inputs)
    assert result and result.passed
    np.testing.assert_allclose(result.outputs["image"], 64 / 255, atol=2e-7)
    too_small = tool(sigma_spatial=256)
    assert too_small.execute({"image": signal}) is None and "reduced image axis" in too_small.error
    invalid = tool(sigma_color=1.5)
    assert invalid.execute({"image": signal}) is None and "Sigma color" in invalid.error
    guide = np.full(signal.shape, 2, np.float32)
    assert strict.execute({"image": signal, "guide": guide}) is None and "[0, 1]" in strict.error


def test_native_invalid_values_are_contained_and_followed_by_recovery(monkeypatch):
    image = np.tile(np.linspace(32, 180, 40, dtype=np.uint8), (32, 1))
    block = tool()

    class InvalidNative:
        def filter(self, signal, joint):
            return np.full_like(signal, np.nan)

    with monkeypatch.context() as edit:
        edit.setattr(cv2.ximgproc, "createAMFilter", lambda *args: InvalidNative())
        assert block.execute({"image": image}) is None
        assert "nonfinite samples" in block.error
    result = block.execute({"image": image})
    assert result and result.passed and not block.error
    normalized = convert_depth(image, "float32")
    expected = cv2.ximgproc.createAMFilter(8, 0.2).filter(normalized, joint=normalized)
    np.testing.assert_array_equal(result.outputs["image"], convert_depth(expected, "uint8"))


def test_uniform_color_guide_reduces_to_spatial_filter():
    signal = np.tile(np.linspace(0.2, 0.7, 40, dtype=np.float32), (32, 1))
    guide = np.full((32, 40, 3), (12, 155, 88), np.uint8)
    block = tool(adjust_outliers=True)
    result = block.execute({"image": signal, "guide": guide})
    assert result and result.passed, block.error
    want = cv2.ximgproc.dtFilter(
        convert_depth(guide, "float32"), signal, 8, 0.2, mode=cv2.ximgproc.DTF_RF, numIters=1
    )
    np.testing.assert_array_equal(result.outputs["image"], want)


@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_project_preserves_edge_and_compares_bilateral(profile):
    graph = Graph.load(EXAMPLE / "adaptive_manifold.json", builtin_registry())
    graph.set_run_profile(profile)
    graph.set_branch_workers(2)
    try:
        graph.run()
        block = graph.blocks["manifold"]
        assert block_entry(block).path[:2] == ("OpenCV", "Extended Image Processing (ximgproc)")
        results = {
            key: graph.blocks[key].last_result for key in ("bilateral", "manifold", "selected")
        }
        assert all(result and result.passed for result in results.values())
        images = {key: result.outputs["image"] for key, result in results.items()}
        noisy = cv2.imread(str(EXAMPLE / "assets/noisy.png"), cv2.IMREAD_UNCHANGED)
        truth = cv2.imread(str(EXAMPLE / "assets/clean.png"), cv2.IMREAD_UNCHANGED).astype(float)
        errors = {
            key: np.sqrt(np.mean((image.astype(float) - truth) ** 2))
            for key, image in {**images, "noisy": noisy}.items()
        }
        assert errors["noisy"] > 2100 and errors["bilateral"] < 165
        assert errors["manifold"] < 130
        for name in ("manifold", "bilateral"):
            field = images[name].astype(float)
            edge = (field[14:80, 80] - field[14:80, 79]).mean()
            assert edge == pytest.approx(8200, abs=120)
        selected = np.zeros(truth.shape, dtype=bool)
        selected[8:73, 13:147] = True
        np.testing.assert_array_equal(images["selected"][selected], images["manifold"][selected])
        np.testing.assert_array_equal(images["selected"][~selected], noisy[~selected])
    finally:
        graph.close()
