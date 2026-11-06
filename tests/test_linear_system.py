"""Exercise exact and least-squares systems through the registered block."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry


@pytest.mark.parametrize("method", ("svd", "qr", "lu", "cholesky", "eigen"))
def test_exact_linear_system_with_wired_observations(method):
    block = builtin_registry().create(
        "cv.solve_linear_system",
        parameters={"method": method, "coefficients": [[2, 1], [1, 3]], "observations": [0, 0]},
    )
    result = block.execute({"observations": [4, 7]})
    assert result is not None, block.error
    np.testing.assert_allclose(result.outputs["solution"], [[1], [2]])
    np.testing.assert_allclose(result.outputs["fitted"], [[4], [7]])
    np.testing.assert_allclose(result.outputs["residuals"], [[0], [0]], atol=1e-12)
    assert result.outputs["rank"] == 2


def test_qr_reports_residuals_and_rmse_for_each_rhs():
    block = builtin_registry().create("cv.solve_linear_system", parameters={"method": "qr"})
    result = block.execute(
        {
            "coefficients": [[1, 0], [1, 1], [1, 2], [1, 3]],
            "observations": [[3, 10], [4, 8], [7, 6], [12, 4]],
        }
    )
    assert result is not None, block.error
    np.testing.assert_allclose(result.outputs["solution"], [[2, 10], [3, -2]])
    np.testing.assert_allclose(
        result.outputs["residuals"], [[1, 0], [-1, 0], [-1, 0], [1, 0]], atol=1e-12
    )
    np.testing.assert_allclose(result.outputs["rmse"], [1, 0], atol=1e-12)
    assert result.graphics["residuals"].shape == (300, 640, 3)


def test_svd_solves_underdetermined_rank_deficient_system_with_minimum_norm():
    block = builtin_registry().create("cv.solve_linear_system")
    result = block.execute({"coefficients": [[1, 1, 0], [2, 2, 0]], "observations": [6, 12]})
    assert result is not None, block.error
    np.testing.assert_allclose(result.outputs["solution"], [[3], [3], [0]])
    assert result.outputs["rank"] == 1

    result = block.execute({"coefficients": [[0]], "observations": [[6, 1e-200]]})
    assert result is not None, block.error
    assert result.outputs["rank"] == 0 and result.outputs["solution"] == [[0.0, 0.0]]
    assert result.outputs["rmse"] == [6.0, 1e-200]


@pytest.mark.parametrize(
    ("method", "coefficients", "error"),
    (
        ("qr", [[1, 1], [2, 2]], "rank deficient"),
        ("lu", [[1, 2, 3], [0, 1, 1]], "square"),
        ("eigen", [[1, 2], [0, 1]], "symmetric"),
        ("cholesky", [[1, 2], [2, 1]], "positive definite"),
    ),
)
def test_invalid_decompositions_fail_safely(method, coefficients, error):
    block = builtin_registry().create("cv.solve_linear_system", parameters={"method": method})
    assert block.execute({"coefficients": coefficients, "observations": [1, 2]}) is None
    assert error in block.error
