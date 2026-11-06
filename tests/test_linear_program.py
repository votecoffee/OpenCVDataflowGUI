"""Check bounded optima, alternate optima, and failed solution outputs."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.payloads import NO_VALUE


@pytest.mark.parametrize(
    ("direction", "bound", "expected", "value", "slacks"),
    (
        ("maximize", [1, 1, 4], [[2], [2]], 10, [0, 0, 1]),
        ("minimize", [-1, -1, -4], [[1], [3]], 9, [0, 1, 0]),
    ),
)
def test_linear_program_uses_wired_objective_and_reports_original_units(
    direction, bound, expected, value, slacks
):
    block = builtin_registry().create(
        "cv.solve_linear_program",
        parameters={
            "direction": direction,
            "objective": [0],
            "constraints": [bound, [1, 0, 2], [0, 1, 3]],
        },
    )
    result = block.execute({"objective": [[3], [2]]})
    assert result is not None, block.error
    assert result.passed and result.outputs["status"] == "unique"
    np.testing.assert_allclose(result.outputs["solution"], expected)
    np.testing.assert_allclose(result.outputs["slacks"], slacks)
    assert result.outputs["objective_value"] == value


def test_multiple_optima_returns_one_feasible_solution():
    block = builtin_registry().create("cv.solve_linear_program")
    result = block.execute({"objective": [1, 1], "constraints": [[1, 1, 2]]})
    assert result is not None, block.error
    assert result.passed and result.outputs["status"] == "multiple"
    assert result.outputs["objective_value"] == 2
    assert sum(row[0] for row in result.outputs["solution"]) == 2


@pytest.mark.parametrize(
    ("constraints", "status"),
    (([[1, 0], [-1, -1]], "infeasible"), ([[-1, 0]], "unbounded")),
)
def test_failed_linear_program_withholds_allocation_and_retains_status(constraints, status):
    block = builtin_registry().create("cv.solve_linear_program")
    result = block.execute({"objective": [1], "constraints": constraints})
    assert result is not None, block.error
    assert not result.passed and result.outputs["status"] == status
    assert result.fail_reasons[0]["code"] == f"linear_program_{status}"
    for field in ("solution", "objective_value", "slacks"):
        assert result.outputs[field] is NO_VALUE
        assert block.current_emissions[field] == []


def test_constraints_are_validated_before_and_after_solving(monkeypatch):
    block = builtin_registry().create(
        "cv.solve_linear_program", parameters={"constraint_tolerance": 0.125}
    )
    assert block.execute({"objective": [1, 2], "constraints": [[1, 2]]}) is None
    assert "followed by its bound" in block.error

    def inaccurate(target, constraints, *, constr_eps):
        assert constr_eps == 0.125
        return cv2.SOLVELP_SINGLE, np.array([[2.0]])

    monkeypatch.setattr(cv2, "solveLP", inaccurate)
    result = block.execute({"objective": [1], "constraints": [[1, 1]]})
    assert result is not None, block.error
    assert not result.passed and result.outputs["status"] == "numeric_failure"
    assert result.outputs["solution"] is NO_VALUE
