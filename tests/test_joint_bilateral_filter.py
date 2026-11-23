"""Qualify registered bilateral filtering, typed pixels and edge handling."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock

TYPE = "cv.joint_bilateral_filter"
EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/joint_bilateral_filter"


def tool(**parameters):
    return builtin_registry().create(TYPE, parameters=parameters)


def test_mixed_depth_color_signal_matches_native_float_path_and_keeps_alpha():
    rng = np.random.default_rng(281)
    original = rng.integers(3400, 55000, (23, 29, 4), dtype=np.uint16)
    alpha = original[..., 3].copy()
    source = ImageArray(original, "rgba")
    guide = rng.integers(43, 220, original.shape[:2], dtype=np.uint8)
    mask = np.zeros(original.shape[:2], np.uint8)
    mask[5:21, 6:22] = 255
    region = {
        "model": "image_region_v1",
        "image_size": [29, 23],
        "shape": "rectangle",
        "rectangle": [9, 2, 16, 16],
    }
    block = tool(diameter=0, sigma_color=0.07, sigma_space=3.1, border="reflect_edge")
    result = block.execute({"image": source, "guide": guide, "mask": mask, "region": region})
    assert result and result.passed, block.error
    native = cv2.ximgproc.jointBilateralFilter(
        (guide / 255).astype(np.float32),
        (original[..., :3] / 65535).astype(np.float32),
        0,
        0.07,
        3.1,
        borderType=cv2.BORDER_REFLECT,
    )
    expected = np.rint(np.clip(native.astype(np.float64), 0, 1) * 65535).astype(np.uint16)
    out = result.outputs["image"]
    selected = np.zeros(original.shape[:2], bool)
    selected[5:18, 9:22] = True
    assert out.dtype == np.uint16 and image_format(out) == "rgba"
    np.testing.assert_array_equal(out[..., :3][selected], expected[selected])
    np.testing.assert_array_equal(out[~selected], original[~selected])
    np.testing.assert_array_equal(out[..., 3], alpha)
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("self_guided", [False, True])
def test_constant_border_matches_independent_circular_kernel(self_guided):
    samples = np.arange(1, 16, dtype=np.float32).reshape(3, 5) / 20 + 0.1
    guide = samples if self_guided else np.linspace(0.87, 0.93, 15, dtype=np.float32).reshape(3, 5)
    sigma_color, sigma_space = 0.31, 1.8
    tool_inputs = {"image": samples}
    if not self_guided:
        tool_inputs["guide"] = guide
    block = tool(diameter=5, sigma_color=sigma_color, sigma_space=sigma_space, border="constant")
    result = block.execute(tool_inputs)
    assert result and result.passed, block.error
    g, a = np.pad(guide.astype(float), 2), np.pad(samples.astype(float), 2)
    offsets = [(y, x) for y in range(-2, 3) for x in range(-2, 3) if x * x + y * y <= 4]
    expected = np.zeros_like(samples)
    for iy, ix in np.ndindex(samples.shape):
        y, x = iy + 2, ix + 2
        values, weights = [], []
        for dy, dx in offsets:
            distance = g[y + dy, x + dx] - g[y, x]
            weights.append(
                np.exp(-0.5 * (distance**2 / sigma_color**2 + (dx * dx + dy * dy) / sigma_space**2))
            )
            values.append(a[y + dy, x + dx])
        expected[iy, ix] = np.average(values, weights=weights)
    np.testing.assert_allclose(result.outputs["image"], expected, rtol=0, atol=3e-6)


def test_guide_frame_validation_and_float_output_own_source_ticket():
    sample = np.full((9, 17), 29100, np.uint16)
    guide = np.full((*sample.shape, 3), 224, np.uint8)
    tickets = TicketClock("registered")
    ticket = tickets.issue("measurement", "fixed", "part-8")
    context = frame([17, 9], "source-frame")
    inputs = {
        "image": ValuePacket(sample, {"spatial": context, "ticket": ticket}),
        "guide": ValuePacket(
            guide,
            {
                "spatial": frame([17, 9], "guide-frame"),
                "ticket": tickets.issue("reference", "fixed", "guide"),
            },
        ),
    }
    strict = tool()
    assert strict.execute(inputs) is None and "share the source pixel frame" in strict.error
    pixel_grid = tool(guide_alignment="pixel_grid", output_depth="float32")
    result = pixel_grid.execute(inputs)
    assert result and result.passed, pixel_grid.error
    out = result.outputs["image"]
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, 29100 / 65535, atol=2e-7)
    assert result.payloads["image"]["spatial"] == context
    assert result.payloads["image"]["ticket"] == ticket
    assert pixel_grid.execute({"image": sample, "guide": guide[:, :-1]}) is None
    assert "same dimensions" in pixel_grid.error


def test_invalid_auto_window_and_unrepresentable_native_inputs_are_rejected():
    sample = np.ones((5, 7), np.float32)
    block = tool(diameter=0, sigma_space=85)
    assert block.execute({"image": sample}) is None
    assert "Automatic neighborhood" in block.error
    valid = tool()
    wide = sample.copy()
    wide[:, 0] *= 3e38
    assert valid.execute({"image": wide}) is None and "Source values" in valid.error
    assert valid.execute({"image": sample, "guide": wide}) is None
    assert "Guide values" in valid.error


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_project_transfers_guide_edges_to_16bit_output(profile):
    graph = Graph.load(EXAMPLE / "joint_bilateral_filter.json", builtin_registry())
    graph.set_run_profile(profile)
    graph.set_branch_workers(2)
    try:
        graph.run()
        keys = ("noisy", "self_guided", "guided", "selected")
        results = {key: graph.blocks[key].last_result for key in keys}
        assert all(result and result.passed for result in results.values())
        images = {key: result.outputs["image"] for key, result in results.items()}
        clean = cv2.imread(str(EXAMPLE / "assets/clean.png"), cv2.IMREAD_UNCHANGED).astype(float)
        rms = {
            key: np.sqrt(np.mean((out.astype(float) - clean) ** 2)) for key, out in images.items()
        }
        assert all(image.dtype == np.uint16 for image in images.values())
        assert rms["guided"] < 250 and rms["guided"] < 0.4 * rms["self_guided"]
        guided = images["guided"].astype(float)
        assert np.mean(guided[15:60, 84] - guided[15:60, 83]) == pytest.approx(8000, abs=180)
        stripe = guided[30:88, 109:112].mean() - guided[30:88, 118:124].mean()
        assert stripe == pytest.approx(2900, abs=260)
        selected = np.zeros(clean.shape, bool)
        selected[8:70, 13:157] = True
        np.testing.assert_array_equal(images["selected"][selected], images["guided"][selected])
        np.testing.assert_array_equal(images["selected"][~selected], images["noisy"][~selected])
    finally:
        graph.close()
