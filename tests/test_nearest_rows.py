"""Check metric choices, candidate restrictions, and nullable match slots."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry


@pytest.mark.parametrize(
    ("metric", "queries", "references", "expected"),
    (
        ("euclidean", [[0, 0]], [[3, 4], [0, 2]], [2, 5]),
        ("squared_euclidean", [[0, 0]], [[3, 4], [0, 2]], [4, 25]),
        ("manhattan", [[0, 0]], [[3, 4], [0, 2]], [2, 7]),
        ("hamming", [[0]], [[255], [15]], [4, 8]),
        ("hamming2", [[0]], [[255], [15]], [2, 4]),
    ),
)
def test_nearest_rows_metrics(metric, queries, references, expected):
    block = builtin_registry().create(
        "cv.nearest_rows",
        parameters={"queries": queries, "references": [[99]], "metric": metric, "neighbors": 3},
    )
    result = block.execute({"references": references})
    assert result is not None, block.error
    assert result.outputs == {"indices": [[1, 0]], "distances": [expected]}


def test_masked_search_keeps_tie_order_and_null_slots():
    block = builtin_registry().create(
        "cv.nearest_rows", parameters={"neighbors": 2, "candidate_mask": [[0]]}
    )
    result = block.execute(
        {
            "queries": [[0], [2], [3]],
            "references": [[1], [-1], [4]],
            "candidate_mask": [[True, True, True], [0, 0, 1], [0, 0, 0]],
        }
    )
    assert result is not None, block.error
    assert result.outputs == {
        "indices": [[0, 1], [2, None], [None, None]],
        "distances": [[1, 1], [2, None], [None, None]],
    }


def test_crosscheck_keeps_only_mutual_matches():
    block = builtin_registry().create("cv.nearest_rows", parameters={"crosscheck": True})
    inputs = {"queries": [[0], [1], [4]], "references": [[0], [4]]}
    result = block.execute(inputs)
    assert result is not None, block.error
    assert result.outputs == {"indices": [[0], [None], [1]], "distances": [[0], [None], [0]]}

    assert block.execute({**inputs, "candidate_mask": [[1, 1]] * 3}) is None
    assert "Crosscheck requires an empty candidate mask" in block.error
    with pytest.raises(ValueError, match="Crosscheck requires one neighbor"):
        block.validate_parameters({**block.parameters, "neighbors": 2})


def test_invalid_shapes_and_nonbyte_values_report_diagnostics():
    block = builtin_registry().create("cv.nearest_rows")
    assert block.execute({"queries": [[0, 1]], "references": [[0]]}) is None
    assert "feature counts" in block.error
    assert (
        block.execute({"queries": [[0]], "references": [[0], [1]], "candidate_mask": [[1], [0]]})
        is None
    )
    assert "one row per query" in block.error
    block.parameters["metric"] = "hamming"
    assert block.execute({"queries": [[0.5]], "references": [[0]]}) is None
    assert "whole bytes" in block.error
    oversized = np.zeros((1_415, 1), dtype=np.uint8).tolist()
    assert block.execute({"queries": oversized, "references": oversized}) is None
    assert "Search is too large" in block.error
