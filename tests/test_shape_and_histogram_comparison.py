"""Check fit residuals and native comparisons of wired histogram data."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray


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


def test_backprojection_selects_logical_channel_and_includes_final_edge():
    registry = builtin_registry()
    ref = ImageArray(np.array([[[0, 2, 2], [0, 2, 2], [6, 2, 2]]], np.uint8), "rgb")
    settings = {"bins": 3, "range_mode": "manual", "range_min": 0, "range_max": 6}
    histogram = registry.create("inspect.histogram", parameters=settings)
    model = histogram.execute({"image": ref}).outputs["histogram"]
    image = np.array([[[2, 2, value] for value in (0, 2, 4, 6, 7, 0)]], np.uint8)
    mask = np.array([[255, 255, 255, 255, 255, 0]], np.uint8)
    block = registry.create(
        "inspect.histogram_backproject", parameters={"channel": "R", "normalization": "probability"}
    )
    result = block.execute({"image": image, "histogram": model, "mask": mask})
    assert result is not None, block.error
    assert result.outputs["image"].dtype == np.float32
    np.testing.assert_allclose(result.outputs["image"], [[2 / 3, 0, 1 / 3, 1 / 3, 0, 0]])


def intensity_histogram(counts, edges):
    return {"channels": [{"channel": "Gray", "counts": counts, "edges": edges}]}


def test_earth_movers_distance_uses_bin_centers_and_normalizes_unequal_counts():
    registry = builtin_registry()
    hist = registry.create("inspect.histogram")
    reference = hist.execute({"image": np.full((12, 16), 40, np.uint8)}).outputs["histogram"]
    sample = hist.execute({"image": np.full((24, 16), 47, np.uint8)}).outputs["histogram"]
    block = registry.create(
        "inspect.compare_histograms",
        parameters={"method": "earth_movers_distance", "check_threshold": True, "threshold": 7},
    )
    result = block.execute({"reference": reference, "sample": sample})
    assert result is not None, block.error
    assert result.outputs["value"] == 7 and result.passed
    assert result.outputs["measurements"]["units"] == "sample"
    block.parameters["threshold"] = 6
    assert not block.execute({"reference": reference, "sample": sample}).passed
    block.parameters["normalize"] = False
    assert block.execute({"reference": reference, "sample": sample}) is None
    assert block.error_kind == "invalid" and "equal counts" in block.error
    assert block.execute({"reference": reference, "sample": reference}).outputs["value"] == 0


def test_earth_movers_distance_handles_different_grids_and_nonuniform_bins():
    block = builtin_registry().create(
        "inspect.compare_histograms", parameters={"method": "earth_movers_distance"}
    )
    origin = 1e12
    edges = [origin + x for x in (0, 2, 4, 8, 10)]
    a = intensity_histogram([1, 0, 3, 0], edges)
    b = intensity_histogram([2, 0, 2], [origin + x for x in (2, 4, 8, 10)])
    result = block.execute({"reference": a, "sample": b})
    assert result is not None, block.error
    assert result.outputs["value"] == pytest.approx(2.75)


def test_earth_movers_distance_rejects_unbounded_dense_signatures():
    block = builtin_registry().create(
        "inspect.compare_histograms", parameters={"method": "earth_movers_distance"}
    )
    dense = intensity_histogram([1] * 513, list(range(514)))
    assert block.execute({"reference": dense, "sample": dense}) is None
    assert block.error_kind == "invalid" and "512 occupied bins" in block.error
