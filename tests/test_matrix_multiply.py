"""Verify native weighted multiplication and its graph table boundaries."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry


def test_weighted_matrix_product_with_all_transpose_flags_and_wired_factors():
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(8, dtype=np.float64).reshape(4, 2)
    addend = np.arange(12, dtype=np.float64).reshape(4, 3)
    block = builtin_registry().create(
        "cv.multiply_matrices",
        parameters={
            "left": left.tolist(),
            "right": [[-1]],
            "product_weight": -2,
            "addend_weight": 0.5,
            "transpose_left": True,
            "transpose_right": True,
            "transpose_addend": True,
        },
    )
    result = block.execute({"right": right.tolist(), "addend": addend.tolist()})
    assert result is not None, block.error
    np.testing.assert_array_equal(
        result.outputs["matrix"], -2 * (left.T @ right.T) + 0.5 * addend.T
    )


def test_matrix_product_can_reuse_a_wide_output():
    block = builtin_registry().create("cv.multiply_matrices", parameters={"left": [[2]]})
    result = block.execute({"right": [[1] * 129]})
    assert result is not None, block.error
    wide = result.outputs["matrix"]
    np.testing.assert_array_equal(wide, [[2] * 129])
    result = block.execute({"left": wide, "right": [[3]] * 129})
    assert result is not None, block.error
    assert result.outputs["matrix"] == [[774]]


@pytest.mark.parametrize(
    ("inputs", "error"),
    (
        ({"left": [[1, 2]], "right": [[1, 2]]}, "Left columns"),
        ({"left": [[1, 2]], "right": [[1], [2]], "addend": [[1, 2]]}, "Addend must match"),
        ({"left": [[1]] * 2_000, "right": [[1] * 1_001]}, "2 million values"),
    ),
)
def test_invalid_matrix_shapes_fail_before_native_multiplication(inputs, error):
    block = builtin_registry().create("cv.multiply_matrices")
    assert block.execute(inputs) is None
    assert error in block.error
