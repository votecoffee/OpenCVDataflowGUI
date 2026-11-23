"""Qualify the L0 subproblems, image contracts and texture example."""

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

TYPE = "cv.l0_smoothing"
EXAMPLE = Path(__file__).parents[1] / "examples/custom tools/l0_smoothing"


def tool(**parameters):
    return builtin_registry().create(TYPE, parameters=parameters)


def dense_l0(signal, weight, growth):
    """Independently solve both subproblems with explicit finite differences."""
    rows, cols = signal.shape[:2]
    size = rows * cols
    values = signal.reshape(size, -1).astype(float)
    dx, dy = np.zeros((size, size)), np.zeros((size, size))
    for y, x in np.ndindex(rows, cols):
        index = y * cols + x
        if x < cols - 1:
            dx[index, [index, index + 1]] = [-1, 1]
        if y < rows - 1:
            dy[index, [index, index + cols]] = [-1, 1]
    laplacian = dx.T @ dx + dy.T @ dy
    estimate = values
    penalty = 2 * weight
    while penalty < 100_000:
        horizontal, vertical = dx @ estimate, dy @ estimate
        remove = np.sum(horizontal**2 + vertical**2, axis=1) <= weight / penalty
        horizontal[remove] = 0
        vertical[remove] = 0
        rhs = values + penalty * (dx.T @ horizontal + dy.T @ vertical)
        estimate = np.linalg.solve(np.eye(size) + penalty * laplacian, rhs)
        penalty *= growth
    return np.clip(estimate.reshape(signal.shape), 0, 1)


@pytest.mark.parametrize("shape", [(1, 7), (3, 5), (5, 3, 3)])
def test_dct_solver_matches_an_independent_dense_system(shape):
    rng = np.random.default_rng(95)
    signal = rng.uniform(0.15, 0.82, shape)
    block = tool(lambda_weight=0.007, kappa=4.0)
    result = block.execute({"image": signal})
    assert result and result.passed, block.error
    expected = dense_l0(signal, 0.007, 4.0)
    actual = result.outputs["image"]
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, atol=4e-10, rtol=0)
    np.testing.assert_allclose(actual.mean(axis=(0, 1)), signal.mean(axis=(0, 1)), atol=4e-10)


def test_uint16_region_alpha_and_ticket_are_preserved():
    rng = np.random.default_rng(812)
    pixels = rng.integers(9000, 50000, (4, 5, 4), dtype=np.uint16)
    alpha = pixels[..., 3].copy()
    mask = np.zeros((4, 5), np.uint8)
    mask[:, 1:4] = 255
    region = {
        "model": "image_region_v1",
        "shape": "rectangle",
        "rectangle": [2, 1, 3, 2],
        "image_size": [5, 4],
    }
    context = frame([5, 4], "camera")
    ticket = TicketClock("l0").issue("camera", "fixed", "part")
    block = tool(lambda_weight=0.01, kappa=4)
    result = block.execute(
        {
            "image": ValuePacket(
                ImageArray(pixels, "rgba"), {"spatial": context, "ticket": ticket}
            ),
            "mask": ValuePacket(mask, {"spatial": context}),
            "region": region,
        }
    )
    assert result and result.passed, block.error
    fitted = dense_l0(pixels[..., :3] / 65535, 0.01, 4)
    expected = pixels.copy()
    expected[1:3, 2:4, :3] = np.rint(fitted[1:3, 2:4] * 65535).astype(np.uint16)
    actual = result.outputs["image"]
    assert actual.dtype == np.uint16 and actual.color_format == "rgba"
    np.testing.assert_allclose(actual, expected, atol=1, rtol=0)
    np.testing.assert_array_equal(actual[..., 3], alpha)
    assert result.payloads["image"]["ticket"] == ticket
    assert result.payloads["image"]["spatial"] == context


def test_flat_border_and_zero_weight_are_exact():
    source = np.full((27, 31), 18637, np.uint16)
    result = tool().execute({"image": source})
    assert result and result.passed
    np.testing.assert_array_equal(result.outputs["image"], source)
    arbitrary = np.arange(36, dtype=np.float32).reshape(4, 9)
    result = tool(lambda_weight=0).execute({"image": arbitrary})
    assert result and result.passed and result.outputs["image"] is not arbitrary
    np.testing.assert_array_equal(result.outputs["image"], arbitrary)


def test_bad_schedules_and_workloads_are_rejected(monkeypatch):
    source = np.zeros((13, 15), np.float32)
    block = tool(kappa=1.01)
    assert block.execute({"image": source}) is None and "128 steps" in block.error
    block = tool()
    assert block.execute({"image": source + 1.01}) is None and "within [0, 1]" in block.error
    monkeypatch.setattr("opencv_dataflow.l0_smoothing._MAX_TRANSFORM_SAMPLE_STEPS", 1)
    assert block.execute({"image": source}) is None and "work exceeds" in block.error


@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_graph_reduces_texture_and_measures_step_fidelity(profile):
    graph = Graph.load(EXAMPLE / "l0_smoothing.json", builtin_registry())
    graph.set_run_profile(profile)
    graph.set_branch_workers(3)
    try:
        graph.run()
        node = graph.blocks["moderate"]
        assert block_entry(node).path == ("Custom", "Image processing", "Filters")
        truth = cv2.imread(str(EXAMPLE / "assets/clean.png"), cv2.IMREAD_UNCHANGED)
        output = {}
        for key in ("textured", "gaussian", "moderate", "strong", "selected"):
            block = graph.blocks[key]
            assert block.last_result and block.last_result.passed, block.error
            output[key] = block.last_result.outputs["image"]
            assert output[key].dtype == np.uint16
        moderate, strong, gaussian = (
            output[key].astype(float) for key in ("moderate", "strong", "gaussian")
        )
        assert np.sqrt(np.mean((moderate - truth) ** 2)) < 120
        assert 690 < np.sqrt(np.mean((strong - truth) ** 2)) < 800
        assert (moderate[:, 83] - moderate[:, 82]).mean() == pytest.approx(27900, abs=60)
        assert (gaussian[:, 83] - gaussian[:, 82]).mean() < 5000
        assert moderate[26:61, 121:124].mean() - moderate[26:61, 116:119].mean() > 5500
        assert abs(strong[26:61, 121:124].mean() - strong[26:61, 116:119].mean()) < 25
        selected = np.zeros(truth.shape, bool)
        selected[7:79, 14:150] = True
        np.testing.assert_array_equal(output["selected"][selected], output["moderate"][selected])
        np.testing.assert_array_equal(output["selected"][~selected], output["textured"][~selected])
    finally:
        graph.close()
