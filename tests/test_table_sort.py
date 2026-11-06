"""Check record preservation, stable ties, and numeric-table sort indices."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry


@pytest.mark.parametrize(
    ("mode", "order", "key", "expected_table", "expected_indices"),
    (
        ("rows_by_column", "descending", 0, [[3, 1, 1], [3, 0, 2], [2, 4, 1]], [0, 1, 2]),
        ("columns_by_row", "ascending", 2, [[1, 3, 1], [2, 3, 0], [1, 2, 4]], [2, 0, 1]),
        (
            "within_rows",
            "ascending",
            0,
            [[1, 1, 3], [0, 2, 3], [1, 2, 4]],
            [[1, 2, 0], [1, 2, 0], [2, 0, 1]],
        ),
        (
            "within_columns",
            "descending",
            0,
            [[3, 4, 2], [3, 1, 1], [2, 0, 1]],
            [[0, 2, 1], [1, 0, 0], [2, 1, 2]],
        ),
    ),
)
def test_numeric_table_sort_modes(mode, order, key, expected_table, expected_indices):
    block = builtin_registry().create(
        "cv.sort_numeric_table",
        parameters={"table": [[99]], "mode": mode, "order": order, "key_index": key},
    )
    result = block.execute({"table": [[3, 1, 1], [3, 0, 2], [2, 4, 1]]})
    assert result is not None, block.error
    assert result.outputs == {"table": expected_table, "indices": expected_indices}


def test_large_unsigned_keys_keep_precision_and_stable_ties():
    max_id = int(np.iinfo(np.uint64).max)
    block = builtin_registry().create("cv.sort_numeric_table", parameters={"order": "descending"})
    result = block.execute({"table": [[max_id - 1], [max_id], [max_id]]})
    assert result is not None, block.error
    assert result.outputs == {"table": [[max_id], [max_id], [max_id - 1]], "indices": [1, 2, 0]}


def test_invalid_table_and_key_report_diagnostics():
    block = builtin_registry().create("cv.sort_numeric_table", parameters={"key_index": 2})
    assert block.execute({"table": [[0, float("inf")]]}) is None
    assert "JSON compliant" in block.error
    assert block.execute({"table": [[0, 2**53 + 1], [1, 0.5]]}) is None
    assert "preserve precision" in block.error
    assert block.execute({"table": [[0, 1]]}) is None
    assert "Key index is outside the table" in block.error
