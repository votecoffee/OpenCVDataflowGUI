"""Exercise the native factor shapes, spectrum, and low-rank graph output."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import NO_VALUE


@pytest.mark.parametrize("mode", ("thin", "full"))
@pytest.mark.parametrize("shape", ((5, 3), (3, 5), (1, 4), (4, 1)))
def test_svd_bases_reconstruct_tall_and_wide_inputs(mode, shape):
    matrix = np.random.default_rng(22).normal(size=shape)
    block = builtin_registry().create("cv.matrix_svd", parameters={"matrix": [[42]], "mode": mode})
    result = block.execute({"matrix": matrix.tolist()})
    assert result is not None, block.error
    values = np.asarray(result.outputs["singular_values"])
    u = np.asarray(result.outputs["left_vectors"])
    vt = np.asarray(result.outputs["right_vectors_t"])
    rows, columns = shape
    size = min(shape)
    assert u.shape == (rows, size if mode == "thin" else rows)
    assert vt.shape == (size if mode == "thin" else columns, columns)
    np.testing.assert_allclose(u.T @ u, np.eye(u.shape[1]), atol=1e-12)
    np.testing.assert_allclose(vt @ vt.T, np.eye(vt.shape[0]), atol=1e-12)
    np.testing.assert_allclose(values, np.linalg.svd(matrix, compute_uv=False))
    np.testing.assert_allclose((u[:, :size] * values) @ vt[:size], matrix, atol=1e-12)
    np.testing.assert_allclose(result.outputs["approximation"], matrix, atol=1e-12)
    assert result.outputs["rank"] == size
    assert result.outputs["retained_energy"] == 1
    assert result.outputs["relative_truncation_error"] == 0


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_svd_example_reuses_low_rank_matrix(profile):
    project = Path(__file__).parents[1] / "examples/numerical" / "matrix_svd" / "matrix_svd.json"
    graph = Graph.load(project, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    fit = graph.blocks["decompose"]
    assert fit.last_result and fit.last_result.passed, fit.error
    outputs = fit.last_result.outputs
    np.testing.assert_allclose(outputs["singular_values"], [5, 2, 0], atol=1e-12)
    assert outputs["rank"] == 2
    assert outputs["retained_energy"] == pytest.approx(25 / 29)
    residual = np.asarray(fit.parameters["matrix"]) - outputs["approximation"]
    expected_error = np.linalg.norm(residual) / np.linalg.norm(fit.parameters["matrix"])
    assert outputs["relative_truncation_error"] == pytest.approx(expected_error)
    assert ("spectrum" in fit.last_result.graphics) == (profile == "debug")
    apply = graph.blocks["apply"]
    assert apply.last_result and apply.last_result.passed, apply.error
    np.testing.assert_allclose(
        apply.last_result.outputs["matrix"], [[10], [0], [0], [0]], atol=1e-12
    )


def test_values_only_emits_spectral_diagnostics_without_vector_events():
    block = builtin_registry().create(
        "cv.matrix_svd", parameters={"mode": "values_only", "components": 1}
    )
    result = block.execute({"matrix": [[4e-200, 0], [0, 1e-200]]})
    assert result is not None, block.error
    np.testing.assert_allclose(result.outputs["singular_values"], [4e-200, 1e-200], atol=0)
    assert result.outputs["rank"] == 2
    assert result.outputs["retained_energy"] == pytest.approx(16 / 17)
    assert result.outputs["relative_truncation_error"] == pytest.approx(1 / np.sqrt(17))
    for name in ("left_vectors", "right_vectors_t", "approximation"):
        assert result.outputs[name] is NO_VALUE
        assert block.current_emissions[name] == []


def test_rank_cutoff_is_independent_of_reconstruction_count_and_handles_zero():
    block = builtin_registry().create("cv.matrix_svd", parameters={"relative_rank_tolerance": 0.1})
    result = block.execute({"matrix": [[100, 0], [0, 5]]})
    assert result is not None, block.error
    assert result.outputs["rank"] == 1
    np.testing.assert_allclose(result.outputs["approximation"], [[100, 0], [0, 5]])
    result = block.execute({"matrix": [[0, 0], [0, 0]]})
    assert result is not None, block.error
    assert result.outputs["rank"] == 0
    assert result.outputs["retained_energy"] == 1
    assert result.outputs["relative_truncation_error"] == 0


@pytest.mark.parametrize(
    ("parameters", "matrix", "error"),
    (
        ({"components": 3}, [[1, 2], [3, 4]], "at most 2 components"),
        ({"mode": "full"}, [[0]] * 1500, "Full bases exceed"),
        ({}, [[1], [2, 3]], "rectangular"),
        ({}, [[1e308, 1e308], [1e308, 1e308]], "float64 range"),
    ),
)
def test_svd_rejects_invalid_input_and_unrepresentable_outputs(parameters, matrix, error):
    block = builtin_registry().create("cv.matrix_svd", parameters=parameters)
    assert block.execute({"matrix": matrix}) is None
    assert error in block.error


def test_parameter_editor_rejects_nonfinite_matrix_values():
    block = builtin_registry().create("cv.matrix_svd")
    with pytest.raises(ValidationError, match="finite") as error:
        block.validate_parameters({**block.parameters, "matrix": [[np.nan]]})
    assert error.value.fields == ("matrix",)
