"""Verify diagonal and symmetric matrix tools against their graph contracts."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def test_diagonal_matrix_builds_rectangular_scaled_identity_and_accepts_wired_vector():
    block = builtin_registry().create(
        "cv.create_diagonal_matrix",
        parameters={"rows": 2, "columns": 3, "diagonal_value": 0.5},
    )
    assert block.execute({}).outputs["matrix"] == [[0.5, 0, 0], [0, 0.5, 0]]
    block.parameters["diagonal"] = [2, -3]
    assert block.execute({}).outputs["matrix"] == [[2, 0, 0], [0, -3, 0]]
    assert block.execute({"diagonal": [[4], [7]]}).outputs["matrix"] == [[4, 0, 0], [0, 7, 0]]
    assert block.execute({"diagonal": []}) is None
    assert block.error_kind == "invalid" and "needs 2 real values" in block.error


def test_diagonal_matrix_bounds_output_before_allocating():
    block = builtin_registry().create(
        "cv.create_diagonal_matrix", parameters={"rows": 2_000, "columns": 2_000}
    )
    assert block.execute({}) is None
    assert block.error_kind == "invalid" and "2 million values" in block.error


def test_matrix_diagonal_reads_rectangular_input_and_rejects_nonfinite_trace():
    block = builtin_registry().create(
        "cv.matrix_diagonal", parameters={"matrix": [[3, 1, 9], [5, -4, 0]]}
    )
    assert block.execute({}).outputs == {"diagonal": [3, -4], "trace": -1}
    assert block.execute({"matrix": [[1e308, 0], [0, 1e308]]}) is None
    assert block.error_kind == "invalid" and "trace exceeds" in block.error


@pytest.mark.parametrize(("triangle", "value"), [("upper", 2), ("lower", 7)])
def test_complete_symmetry_only_copies_selected_triangle(triangle, value):
    matrix = [[3, 2], [7, 4]]
    block = builtin_registry().create(
        "cv.complete_symmetric_matrix", parameters={"source_triangle": triangle}
    )
    result = block.execute({"matrix": matrix})
    assert result is not None, block.error
    assert result.outputs["matrix"] == [[3, value], [value, 4]]
    assert matrix == [[3, 2], [7, 4]]
    assert block.execute({"matrix": [[1, 2, 3], [4, 5, 6]]}) is None
    assert block.error_kind == "invalid" and "square" in block.error


def test_example_builds_regularized_symmetric_matrix_and_inverts_it():
    path = (
        Path(__file__).parents[1] / "examples/opencv tools/matrix_structure/matrix_structure.json"
    )
    graph = Graph.load(path, builtin_registry())
    try:
        graph.run()
        assert all(block.state == "pass" for block in graph.blocks.values())
        assert graph.blocks["measure"].last_result.outputs == {
            "diagonal": [4.5, 5.5, 6.5],
            "trace": 16.5,
        }
        matrix = np.asarray(graph.blocks["regularize"].last_result.outputs["matrix"])
        inverse = np.asarray(graph.blocks["invert"].last_result.outputs["inverse"])
        np.testing.assert_allclose(matrix @ inverse, np.eye(3), atol=1e-14)
    finally:
        graph.close()
