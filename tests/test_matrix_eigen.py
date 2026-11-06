"""Verify real spectra, row eigenvectors, and the graph judgment contract."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE


@pytest.mark.parametrize("scale", (1e-200, 1.0, 1e200))
def test_symmetric_eigenpairs_preserve_sign_and_scale(scale):
    matrix = np.array([[3, 4], [4, -3]], dtype=float)
    block = builtin_registry().create(
        "cv.matrix_eigen", parameters={"matrix": [[42]], "method": "symmetric"}
    )
    result = block.execute({"matrix": (matrix * scale).tolist()})
    assert result is not None and result.passed, block.error
    outputs = result.outputs
    values = np.asarray(outputs["eigenvalues"]) / scale
    vectors = np.asarray(outputs["eigenvectors"])
    np.testing.assert_allclose(values, [5, -5])
    np.testing.assert_allclose(vectors @ vectors.T, np.eye(2), atol=1e-14)
    np.testing.assert_allclose((vectors.T * values) @ vectors, matrix)
    assert outputs["basis_rank"] == 2
    assert outputs["spectral_radius"] / scale == pytest.approx(5)
    assert outputs["max_relative_residual"] < 1e-14


def test_general_real_basis_uses_right_eigenvectors_and_signed_spectrum():
    matrix = np.array([[1, 3, 2], [0, -7, 2], [0, 0, 2]], dtype=float)
    block = builtin_registry().create("cv.matrix_eigen")
    result = block.execute({"matrix": matrix.tolist()})
    assert result is not None and result.passed, block.error
    outputs = result.outputs
    values = np.asarray(outputs["eigenvalues"])
    vectors = np.asarray(outputs["eigenvectors"])
    np.testing.assert_allclose(values, [2, 1, -7])
    np.testing.assert_allclose(matrix @ vectors.T, vectors.T * values, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), np.ones(3))
    assert outputs["basis_rank"] == 3
    assert outputs["spectral_radius"] == 7


def test_repeated_general_eigenvalue_can_have_an_incomplete_basis():
    block = builtin_registry().create("cv.matrix_eigen")
    result = block.execute({"matrix": [[2, 1], [0, 2]]})
    assert result is not None and result.passed, block.error
    assert result.outputs["eigenvalues"] == [2, 2]
    assert result.outputs["basis_rank"] == 1
    assert result.outputs["max_relative_residual"] < 1e-14


def test_zero_eigenvalues_have_zero_residuals():
    block = builtin_registry().create("cv.matrix_eigen")
    result = block.execute({"matrix": [[0, 0], [0, 0]]})
    assert result is not None and result.passed, block.error
    assert result.outputs["eigenvalues"] == [0, 0]
    assert result.outputs["spectral_radius"] == 0
    assert result.outputs["basis_rank"] == 2
    assert result.outputs["relative_residuals"] == [0, 0]


def test_complex_spectrum_is_rejected_before_the_real_solver(monkeypatch):
    def unexpected_call(*_args):
        pytest.fail("The native real solver must not receive complex spectra")

    monkeypatch.setattr(cv2, "eigenNonSymmetric", unexpected_call)
    block = builtin_registry().create("cv.matrix_eigen")
    assert block.execute({"matrix": [[1, -2], [2, 1]]}) is None
    assert "complex eigenvalues" in block.error


@pytest.mark.parametrize(
    ("matrix", "message"),
    (([[1, 2]], "square matrix"), ([[1, 2], [0, 1]], "matching entries")),
)
def test_invalid_symmetric_inputs_fail_with_clear_messages(matrix, message):
    block = builtin_registry().create("cv.matrix_eigen", parameters={"method": "symmetric"})
    assert block.execute({"matrix": matrix}) is None
    assert message in block.error


def test_bad_eigenpairs_emit_residual_diagnostics_without_spectrum(monkeypatch):
    monkeypatch.setattr(cv2, "eigen", lambda _matrix: (True, np.zeros((2, 1)), np.eye(2)))
    block = builtin_registry().create("cv.matrix_eigen")
    result = block.execute({"matrix": [[1, 0], [0, 1]]})
    assert result is not None and not result.passed, block.error
    assert result.outputs["max_relative_residual"] == pytest.approx(1 / np.sqrt(2))
    assert result.fail_reasons[0]["code"] == "eigenpair_residual_exceeds_limit"
    for name in ("eigenvalues", "eigenvectors", "basis_rank", "spectral_radius"):
        assert result.outputs[name] is NO_VALUE
        assert block.current_emissions[name] == []
    assert len(block.current_emissions["relative_residuals"]) == 1


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_example_projects_along_gram_matrix_axes(profile):
    project = (
        Path(__file__).parents[1] / "examples/numerical" / "matrix_eigen" / "matrix_eigen.json"
    )
    graph = Graph.load(project, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    axes = graph.blocks["axes"].last_result
    np.testing.assert_allclose(axes.outputs["eigenvalues"], [16, 4])
    assert ("spectrum" in axes.graphics) == (profile == "debug")
    projection = graph.blocks["project"].last_result.outputs["matrix"]
    np.testing.assert_allclose(np.abs(projection), [[np.sqrt(8), 0]], atol=1e-12)
    general = graph.blocks["general"].last_result.outputs
    np.testing.assert_allclose(general["eigenvalues"], [0.8, 0.5])
    assert general["spectral_radius"] == pytest.approx(0.8)
