"""Check root ordering, repeated zero factors, and residual judgments."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry


def test_all_roots_accepts_wired_column_in_constant_first_order():
    block = builtin_registry().create(
        "cv.polynomial_roots",
        parameters={"coefficients": [1], "coefficient_order": "constant_first"},
    )
    result = block.execute({"coefficients": [[-1], [0], [0], [0], [1]]})
    assert result is not None, block.error
    np.testing.assert_allclose(result.outputs["roots"], [[-1, 0], [0, -1], [0, 1], [1, 0]])
    assert result.outputs["real_roots"] == [-1.0, 1.0]
    assert result.passed and result.outputs["max_relative_residual"] < 1e-12


def test_analytic_real_returns_distinct_roots_and_handles_no_real_solutions():
    block = builtin_registry().create("cv.polynomial_roots", parameters={"method": "analytic_real"})
    result = block.execute({"coefficients": [0, 1, 0, -3, 2]})
    assert result is not None, block.error
    np.testing.assert_allclose(result.outputs["real_roots"], [-2, 1])
    result = block.execute({"coefficients": [1, 0, 1]})
    assert result is not None and result.passed
    assert result.outputs["roots"] == []


@pytest.mark.parametrize(("method", "count"), (("all_roots", 3), ("analytic_real", 1)))
def test_zero_factors_produce_exact_roots(method, count):
    block = builtin_registry().create("cv.polynomial_roots", parameters={"method": method})
    result = block.execute({"coefficients": [2, 0, 0, 0]})
    assert result is not None, block.error
    assert result.outputs["roots"] == [[0, 0]] * count
    assert result.outputs["relative_residuals"] == [0.0] * count


def test_underiterated_search_returns_candidates_with_failed_residual_judgment():
    block = builtin_registry().create(
        "cv.polynomial_roots", parameters={"max_iterations": 1, "residual_limit": 1e-8}
    )
    result = block.execute({"coefficients": [1, 0, 0, 1, 1]})
    assert result is not None, block.error
    assert not result.passed and result.outputs["max_relative_residual"] > 1e-8
    assert len(result.outputs["roots"]) == len(result.outputs["relative_residuals"]) == 4


def test_degenerate_polynomials_and_method_limits():
    block = builtin_registry().create("cv.polynomial_roots", parameters={"method": "analytic_real"})
    result = block.execute({"coefficients": [0, 4]})
    assert result is not None and result.outputs["roots"] == []
    assert block.execute({"coefficients": [0, 0]}) is None
    assert "infinitely many roots" in block.error
    assert block.execute({"coefficients": [1, 0, 0, 0, 1]}) is None
    assert "degree 3 or lower" in block.error
