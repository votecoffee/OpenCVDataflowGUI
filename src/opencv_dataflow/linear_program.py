"""Optimize a linear objective with nonnegative variables and upper-bound constraints."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .sample_tables import finite_features

_MAX_VARIABLES = 64
_MAX_CONSTRAINTS = 1_024
_SOLVED = {cv2.SOLVELP_SINGLE: "unique", cv2.SOLVELP_MULTI: "multiple"}
_FAILED = {
    cv2.SOLVELP_UNFEASIBLE: ("infeasible", "No feasible solution"),
    cv2.SOLVELP_UNBOUNDED: ("unbounded", "The objective is unbounded"),
    cv2.SOLVELP_LOST: ("numeric_failure", "Could not verify a finite, feasible solution"),
}


def _table(value, field):
    try:
        table = np.asarray(value)
        if field == "objective":
            if table.ndim == 1 or (table.ndim == 2 and 1 in table.shape):
                table = table.reshape(1, -1)
            if table.ndim != 2 or table.shape[0] != 1:
                raise ValidationError("Use one objective coefficient per variable in a vector")
            return finite_features(table, max_columns=_MAX_VARIABLES)
        constraints = finite_features(table, max_columns=_MAX_VARIABLES + 1)
        if constraints.shape[1] < 2 or len(constraints) > _MAX_CONSTRAINTS:
            raise ValidationError("Use up to 1,024 constraints with a final bound in each row")
        return constraints
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValidationError(f"Invalid {field}: {exc}", field) from exc


def _failed_result(code):
    status, message = _FAILED[code]
    return Result(
        {
            "status": status,
            "solution": NO_VALUE,
            "objective_value": NO_VALUE,
            "slacks": NO_VALUE,
        },
        passed=False,
        message=message,
        fail_reasons=[f"linear_program_{status}"],
    )


class SolveLinearProgram(Block):
    type_id, title = "cv.solve_linear_program", "Solve linear program"
    category, block_type, subtype = "Image processing", "Image processing", "Statistics"
    inputs = (
        Port("objective", "data", required=False, required_when_connected=True),
        Port("constraints", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("status", "text"),
        Port("solution", "data"),
        Port("objective_value", "number"),
        Port("slacks", "data"),
    )
    process_inputs_readonly = True
    defaults = {
        "objective": [],
        "constraints": [],
        "direction": "maximize",
        "constraint_tolerance": 1e-12,
    }
    parameter_choices = {"direction": ("maximize", "minimize")}
    parameter_ranges = {"constraint_tolerance": NumericRange(0)}
    parameter_help = {
        "objective": "One coefficient per nonnegative continuous variable. Wired values win.",
        "constraints": "Rows have one coefficient per variable and a final upper bound.",
        "constraint_tolerance": "Maximum numerical constraint violation in the original units.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for field in ("objective", "constraints"):
            value = p[field]
            if not (isinstance(value, (list, tuple)) and not value):
                _table(value, field)

    def process(self, inputs):
        p = self.parameters
        objective = _table(inputs.get("objective", p["objective"]), "objective")
        constraints = _table(inputs.get("constraints", p["constraints"]), "constraints")
        if constraints.shape[1] != objective.shape[1] + 1:
            raise ValidationError(
                "Each constraint needs one coefficient per variable followed by its bound",
                "constraints",
                "objective",
            )
        tolerance = float(p["constraint_tolerance"])
        target = objective if p["direction"] == "maximize" else -objective
        code, solution = cv2.solveLP(target, constraints, constr_eps=tolerance)
        if code not in _SOLVED:
            return _failed_result(code)

        vector = solution.ravel()
        if not np.isfinite(vector).all() or np.any(vector < -tolerance):
            return _failed_result(cv2.SOLVELP_LOST)
        vector = np.maximum(vector, 0)
        with np.errstate(over="ignore", invalid="ignore"):
            slacks = constraints[:, -1] - constraints[:, :-1] @ vector
            value = float(objective[0] @ vector)
        if not np.isfinite(value) or not np.isfinite(slacks).all() or np.any(slacks < -tolerance):
            return _failed_result(cv2.SOLVELP_LOST)

        status = _SOLVED[code]
        return Result(
            {
                "status": status,
                "solution": vector[:, None].tolist(),
                "objective_value": value,
                "slacks": slacks.tolist(),
            },
            message=f"{status.capitalize()} optimum, objective {value:.4g}",
        )
