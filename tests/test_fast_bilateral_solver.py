"""Known bilateral systems, support semantics and the native build boundary."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.fast_bilateral_solver import _opencv_factory
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

TYPE = "cv.fast_bilateral_solver"
EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/fast_bilateral_solver"


def segment(**parameters):
    return builtin_registry().create(TYPE, parameters=parameters)


def test_bilateral_system_matches_dense_reference():
    signal = np.array([[0, 40, 90, 130, 120, 200], [6, 36, 90, 130, 110, 190]], np.float32)
    weights = np.tile(np.array([1, 0.3, 0, 0, 0.5, 1], np.float32), (2, 1))
    guide = np.zeros((2, 6), np.uint8)
    block = segment(backend="scipy", sigma_spatial=2, lambda_weight=8)
    result = block.execute({"image": signal, "guide": guide, "confidence": weights})
    assert result and result.passed, block.error

    # Explicit splat matrix and 3-bin factorization for the hand-built lattice.
    splat = np.tile(np.repeat(np.eye(3), 2, axis=1), (1, 2))
    blur = np.array([[10, 1, 0], [1, 10, 1], [0, 1, 10]], dtype=float)
    norm = np.ones(3)
    for _ in range(10):
        norm = np.sqrt(norm * splat.sum(axis=1) / (blur @ norm))
    diagonal = np.diag(norm)
    laplacian = np.diag(norm * (blur @ norm)) - diagonal @ blur @ diagonal
    system = 8 * laplacian + np.diag(splat @ weights.ravel())
    expected = splat.T @ np.linalg.solve(system, splat @ (weights * signal).ravel())
    np.testing.assert_allclose(result.outputs["image"], expected.reshape(signal.shape), atol=2e-4)
    assert result.outputs["report"]["relative_residuals"][0] < 1e-5
    assert (result.outputs["supported"] == 255).all()


def test_confidence_mask_output_selection_encoding_and_ticket():
    signal = ImageArray(np.full((8, 10, 4), [100, 200, 300, 70], dtype=np.uint16), "rgba")
    signal[0, 0, :3] = [1000, 2000, 3000]
    signal[..., 3] = np.arange(80, dtype=np.uint16).reshape(8, 10)
    guide = np.full((8, 10, 3), (50, 160, 75), np.uint8)
    confidence = np.zeros((8, 10), np.uint8)
    confidence[0, 0] = 255
    mask = np.zeros_like(confidence)
    mask[3:6, 4:7] = 255
    block = segment(backend="scipy", lambda_weight=0, sigma_spatial=12)
    spatial = frame([10, 8], "camera")
    ticket = TicketClock("bilateral").issue("camera", "fixed", "plate")
    result = block.execute(
        {
            "image": ValuePacket(signal, {"spatial": spatial, "ticket": ticket}),
            "guide": ValuePacket(guide, {"spatial": spatial}),
            "confidence": ValuePacket(confidence, {"spatial": spatial}),
            "mask": ValuePacket(mask, {"spatial": spatial}),
        }
    )
    assert result and result.passed, block.error
    output = result.outputs["image"]
    assert output.dtype == np.uint16 and output.color_format == "rgba"
    selected = mask > 0
    assert np.all(output[selected, :3] == [1000, 2000, 3000])
    np.testing.assert_array_equal(output[~selected], signal[~selected])
    np.testing.assert_array_equal(output[..., 3], signal[..., 3])
    np.testing.assert_array_equal(result.outputs["supported"], mask)
    assert result.payloads["supported"]["spatial"] == spatial
    assert result.payloads["image"]["ticket"] == ticket
    wrong_frame = frame([10, 8], "other-camera")
    assert (
        block.execute(
            {
                "image": ValuePacket(signal, {"spatial": spatial}),
                "guide": ValuePacket(guide, {"spatial": spatial}),
                "confidence": ValuePacket(confidence, {"spatial": wrong_frame}),
            }
        )
        is None
    )
    assert "Confidence must share" in block.error


def test_zero_confidence_and_early_iteration_stop():
    signal = np.tile(np.arange(36, dtype=np.uint8), (18, 1))
    guide = np.zeros_like(signal)
    block = segment(backend="scipy", iterations=1, sigma_spatial=3)
    zero = block.execute({"image": signal, "guide": guide, "confidence": np.zeros_like(signal)})
    assert zero and zero.passed
    np.testing.assert_array_equal(zero.outputs["image"], signal)
    assert not zero.outputs["supported"].any() and zero.outputs["report"]["backend"] == "none"
    result = block.execute({"image": signal, "guide": guide})
    assert result is not None and not result.passed and not block.error
    assert result.outputs["report"]["converged"] is False
    assert result.fail_reasons[0]["code"] == "bilateral_solver_not_converged"
    invalid = block.execute(
        {"image": signal, "guide": guide, "confidence": np.full(signal.shape, 1.01, np.float32)}
    )
    assert invalid is None and "within [0, 1]" in block.error


@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_sparse_signal_preserves_edges_and_unobserved_band(profile):
    graph = Graph.load(EXAMPLE / "sparse_signal.json", builtin_registry())
    graph.set_run_profile(profile)
    try:
        graph.run()
        block = graph.blocks["solver"]
        assert block_entry(block).path[:2] == ("OpenCV", "Extended Image Processing (ximgproc)")
        result = block.last_result
        assert result and result.passed, block.error
        left, middle, unobserved = np.split(result.outputs["image"], 3, axis=1)
        assert (left == 53).all() and (middle == 187).all() and (unobserved == 83).all()
        support = result.outputs["supported"]
        assert (support[:, :96] == 255).all() and not support[:, 96:].any()
        report = result.outputs["report"]
        assert report["backend"] == "scipy" and report["converged"]
        assert report["observed_pixels"] == 150 and report["supported_selected_pixels"] == 9216
    finally:
        graph.close()


def test_native_backend_reports_missing_eigen_or_matches_native_filter():
    guide = np.tile(np.repeat(np.array([12, 32, 38, 45], np.uint8), 8), (24, 1))
    signal = np.tile(np.linspace(0.1, 0.8, 32, dtype=np.float32), (24, 1))
    confidence = np.ones_like(signal)
    block = segment(backend="opencv")
    actual = block.execute({"image": signal, "guide": guide, "confidence": confidence})
    if _opencv_factory() is None:
        assert actual is None and "Eigen-enabled" in block.error
        return
    solver = cv2.ximgproc.createFastBilateralSolverFilter(guide, 8.0, 8.0, 8.0, 128.0, 100, 1e-5)
    expected = solver.filter(signal, confidence)
    assert actual and actual.passed, block.error
    np.testing.assert_allclose(actual.outputs["image"], expected, atol=2e-5)
    assert actual.outputs["report"]["backend"] == "opencv"


def test_distinct_bins_cannot_alias_at_the_native_radix():
    guide = np.full((2, 256), 200, np.uint8)
    guide[0, -1] = guide[1, 0] = 20
    signal = np.zeros_like(guide)
    signal[0, -1] = 180
    weights = np.zeros_like(guide)
    weights[0, -1] = 255
    block = segment(backend="scipy", sigma_spatial=1)
    result = block.execute({"image": signal, "guide": guide, "confidence": weights})
    assert result and result.passed
    assert np.count_nonzero(result.outputs["supported"]) == 1
    np.testing.assert_array_equal(result.outputs["image"], signal)
