"""Check row/column products, reusable offsets, and bounds on Gram matrices."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry


def test_column_gram_with_sample_normalization_matches_covariance():
    samples = np.array([[1, 2], [3, 4], [5, 8]], np.float64)
    block = builtin_registry().create(
        "cv.gram_matrix", parameters={"centering": "column_mean", "normalization": "sample"}
    )
    result = block.execute({"samples": samples.tolist()})
    assert result is not None, block.error
    np.testing.assert_allclose(result.outputs["matrix"], np.cov(samples, rowvar=False))
    np.testing.assert_allclose(result.outputs["offset"], [[3, 14 / 3]])


@pytest.mark.parametrize("precision", ("float32", "float64"))
def test_row_gram_with_wired_offset_and_population_scale(precision):
    samples = np.array([[5, 7, 3], [2, -1, 2]], np.float64)
    block = builtin_registry().create(
        "cv.gram_matrix",
        parameters={
            "axis": "rows",
            "centering": "row_mean",
            "normalization": "population",
            "scale": 1.5,
            "precision": precision,
        },
    )
    result = block.execute({"samples": samples.tolist(), "offset": [[1], [-1]]})
    assert result is not None, block.error
    shifted = samples - [[1], [-1]]
    np.testing.assert_allclose(result.outputs["matrix"], 0.5 * (shifted @ shifted.T))
    assert result.outputs["offset"] == [[1], [-1]]


def test_uncentered_matrix_exposes_reusable_zero_offset():
    registry = builtin_registry()
    direct = registry.create("cv.gram_matrix").execute({"samples": [[1, 2], [3, 4]]})
    assert direct is not None
    assert direct.outputs == {"matrix": [[10, 14], [14, 20]], "offset": [[0]]}
    reused = registry.create("cv.gram_matrix", parameters={"centering": "row_mean"})
    result = reused.execute({"samples": [[1, 2], [3, 4]], "offset": direct.outputs["offset"]})
    assert result is not None, reused.error
    assert result.outputs == direct.outputs


def test_invalid_offset_sample_length_and_output_size_are_rejected():
    block = builtin_registry().create("cv.gram_matrix", parameters={"normalization": "sample"})
    assert block.execute({"samples": [[1, 2]]}) is None
    assert "length two or more" in block.error
    assert block.execute({"samples": [[1, 2], [3, 4]], "offset": [[1, 2, 3]]}) is None
    assert "Offset must match" in block.error
    assert block.execute({"samples": [[1] * 1_415] * 2}) is None
    assert "2 million values" in block.error
