"""Validate numeric covariance fitting and correlation-aware distance judgments."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry


@pytest.mark.parametrize("normalization,ddof", [("population", 0), ("sample", 1)])
def test_covariance_fits_mean_and_normalization(normalization, ddof):
    block = builtin_registry().create(
        "cv.feature_covariance", parameters={"normalization": normalization}
    )
    samples = [[1, 6], [3, 8], [5, 4], [7, 10]]
    result = block.execute({"samples": samples})
    assert result is not None, block.error
    assert result.outputs["mean"] == [4, 7]
    np.testing.assert_allclose(
        result.outputs["covariance"], np.cov(samples, rowvar=False, ddof=ddof)
    )
    assert result.outputs["model"]["normalization"] == normalization
    assert result.outputs["sample_count"] == 4


def test_distance_detects_anomaly_across_correlated_features():
    registry = builtin_registry()
    fit = registry.create("cv.feature_covariance", parameters={"normalization": "population"})
    check = registry.create(
        "inspect.mahalanobis_distance", parameters={"check_distance": True, "distance_limit": 3}
    )
    samples = [[-2, -1], [-1, -2], [1, 2], [2, 1]]
    model = fit.execute({"samples": samples}).outputs["model"]
    result = check.execute({"model": model, "samples": [[0, 0], [2, 2], [2, -2]]})
    assert result is not None, check.error
    assert result.outputs["distances"] == pytest.approx([0, 4 / 3, 4])
    assert result.outputs["maximum_distance"] == pytest.approx(4)
    assert result.outputs["outlier_indices"] == [2] and result.outputs["outlier_count"] == 1
    assert not result.passed
    assert result.fail_reasons[0]["code"] == "mahalanobis_distance_out_of_tolerance"
    assert result.graphics["distances"].shape == (300, 640, 3)

    check.parameters["check_distance"] = False
    result = check.execute({"model": model, "samples": [[2, -2]]})
    assert result.passed and result.outputs["outlier_indices"] == [0]


def test_distance_requires_regularization_for_singular_model_and_rejects_indefinite_input():
    registry = builtin_registry()
    fit = registry.create("cv.feature_covariance")
    model = fit.execute({"samples": [[1], [1]]}).outputs["model"]
    check = registry.create("inspect.mahalanobis_distance")
    assert check.execute({"model": model, "samples": [[2]]}) is None
    assert "singular" in check.error
    check.parameters["ridge"] = 0.25
    result = check.execute({"model": model, "samples": [[2]]})
    assert result is not None, check.error
    assert result.outputs["distances"] == [2]

    assert check.execute({"model": model, "samples": [[2, 3]]}) is None
    assert "feature count" in check.error
    assert check.execute({"model": {**model, "covariance": [[-0.1]]}, "samples": [[2]]}) is None
    assert "positive semidefinite" in check.error
    assert fit.execute({"samples": [[1]]}) is None
    assert "at least two rows" in fit.error
