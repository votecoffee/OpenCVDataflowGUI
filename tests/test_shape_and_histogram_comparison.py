"""Check fit residuals and native comparisons of wired histogram data."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry


@pytest.mark.parametrize("shape", ["line", "ellipse", "enclosing_circle"])
def test_native_fit_and_residual(shape):
    block = builtin_registry().create(
        "inspect.fit_shape", parameters={"shape": shape, "check_rmse": True}
    )
    if shape == "line":
        points = [[2 * x, x + 1] for x in range(10)]
    else:
        angle = np.arange(24) * (2 * np.pi / 24)
        points = np.column_stack(
            (
                100 + 9 * np.cos(angle),
                200 + (5 if shape == "ellipse" else 9) * np.sin(angle),
            )
        ).tolist()
    result = block.execute({"points": points})
    assert result is not None, block.error
    assert result.passed
    assert result.outputs["rmse"] < 0.001
    if shape == "line":
        np.testing.assert_allclose(result.outputs["points"], [points[0], points[-1]], atol=1e-5)
    else:
        np.testing.assert_allclose(result.outputs["points"], [[100, 200]], atol=1e-4)
        if shape == "ellipse":
            np.testing.assert_allclose(sorted(result.outputs["fit"]["axis_lengths"]), [10, 18])
        else:
            assert result.outputs["fit"]["radius"] == pytest.approx(9, abs=0.001)


def test_histogram_comparison_normalizes_counts_and_rejects_different_edges():
    registry = builtin_registry()
    hist = registry.create("inspect.histogram")
    compare = registry.create("inspect.compare_histograms", parameters={"check_threshold": True})
    a = hist.execute({"image": np.full((12, 16), 20, np.uint8)}).outputs["histogram"]
    b = hist.execute({"image": np.full((24, 16), 20, np.uint8)}).outputs["histogram"]
    result = compare.execute({"reference": a, "sample": b})
    assert result is not None, compare.error
    assert result.passed and result.outputs["value"] == 0
    c = hist.execute({"image": np.full((12, 16), 200, np.uint8)}).outputs["histogram"]
    different = compare.execute({"reference": a, "sample": c})
    assert different is not None, compare.error
    assert not different.passed and different.outputs["value"] == 1
    c["channels"][0]["edges"][-1] += 1
    assert compare.execute({"reference": a, "sample": c}) is None
    assert "bin edges differ" in compare.error
