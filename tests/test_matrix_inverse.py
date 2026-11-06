"""Check native inversion modes, pseudoinverses, and determinant boundaries."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.payloads import NO_VALUE


@pytest.mark.parametrize("method", ("svd", "lu", "cholesky"))
def test_invert_full_rank_square_and_report_determinant(method):
    matrix = np.array([[4, 2], [2, 3]], np.float64)
    block = builtin_registry().create(
        "cv.invert_matrix", parameters={"matrix": [[0]], "method": method}
    )
    result = block.execute({"matrix": matrix.tolist()})
    assert result is not None, block.error
    np.testing.assert_allclose(result.outputs["inverse"], np.linalg.inv(matrix))
    assert result.outputs["rank"] == 2
    assert result.outputs["reciprocal_condition"] == pytest.approx(1 / np.linalg.cond(matrix))
    assert result.outputs["determinant"] == 8.0


@pytest.mark.parametrize(
    ("matrix", "condition"),
    (([[1, 2, 3]], 1.0), ([[1], [2], [3]], 1.0), ([[1, 2], [2, 4], [3, 6]], 0.0)),
)
def test_pseudoinverse_for_rectangular_rank_one_matrices(matrix, condition):
    block = builtin_registry().create("cv.invert_matrix")
    result = block.execute({"matrix": matrix})
    assert result is not None, block.error
    np.testing.assert_allclose(result.outputs["inverse"], np.linalg.pinv(matrix))
    assert result.outputs["rank"] == 1
    assert result.outputs["reciprocal_condition"] == condition
    assert result.outputs["determinant"] is NO_VALUE
    assert block.current_emissions["determinant"] == []


@pytest.mark.parametrize("scale", (1e-12, 1e12))
def test_out_of_range_determinant_keeps_the_valid_inverse(scale):
    block = builtin_registry().create("cv.invert_matrix")
    identity = np.eye(32)
    result = block.execute({"matrix": (identity * scale).tolist()})
    assert result is not None, block.error
    assert result.outputs["determinant"] is NO_VALUE
    np.testing.assert_allclose(result.outputs["inverse"], identity / scale)


def test_tiny_and_zero_square_matrices_keep_scale_and_rank():
    block = builtin_registry().create("cv.invert_matrix")
    result = block.execute({"matrix": [[1e-200]]})
    assert result is not None, block.error
    assert result.outputs == {
        "inverse": [[1e200]],
        "rank": 1,
        "reciprocal_condition": 1,
        "determinant": 1e-200,
    }
    result = block.execute({"matrix": [[0]]})
    assert result is not None, block.error
    assert result.outputs == {
        "inverse": [[0]],
        "rank": 0,
        "reciprocal_condition": 0,
        "determinant": 0,
    }


def test_mixed_scales_recover_determinant_after_native_overflow():
    block = builtin_registry().create("cv.invert_matrix")
    matrix = np.diag([1e12] * 32 + [1e-12] * 32)
    result = block.execute({"matrix": matrix.tolist()})
    assert result is not None, block.error
    assert result.outputs["determinant"] == pytest.approx(1.0)
    assert result.outputs["rank"] == 32


@pytest.mark.parametrize(
    ("method", "matrix", "error"),
    (
        ("lu", [[1, 2, 3]], "square"),
        ("lu", [[1, 2], [2, 4]], "rank deficient"),
        ("cholesky", [[2, 1], [0, 2]], "symmetric"),
        ("cholesky", [[1, 2], [2, 1]], "positive definite"),
    ),
)
def test_direct_methods_reject_invalid_matrices(method, matrix, error):
    block = builtin_registry().create("cv.invert_matrix", parameters={"method": method})
    assert block.execute({"matrix": matrix}) is None
    assert error in block.error
