"""Verify fitted PCA models and both directions of native projection."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry


@pytest.mark.parametrize(("selection", "expected_count"), [("variance", 1), ("count", 2)])
def test_pca_retains_variance_and_reconstructs(selection, expected_count):
    registry = builtin_registry()
    fit = registry.create(
        "cv.fit_pca", parameters={"selection": selection, "minimum_variance": 0.9}
    )
    samples = [[6, 20], [14, 20], [10, 19], [10, 21]]
    result = fit.execute({"samples": samples})
    assert result is not None, fit.error
    assert result.outputs["component_count"] == expected_count
    ratios = [16 / 17, 1 / 17][:expected_count]
    assert result.outputs["explained_variance_ratios"] == pytest.approx(ratios)
    assert result.outputs["retained_variance"] == pytest.approx(sum(ratios))
    assert result.graphics["variance"].shape == (300, 640, 3)

    project = registry.create("cv.transform_pca")
    reconstruct = registry.create("cv.transform_pca", parameters={"direction": "reconstruct"})
    model = result.outputs["model"]
    assert model["mean"] == [10, 20]
    scores = project.execute({"model": model, "samples": [[13, 21], [8, 18]]})
    assert scores is not None, project.error
    assert scores.outputs["feature_count"] == expected_count
    expected = [[3, 1], [-2, -2]] if expected_count == 2 else [[3], [-2]]
    np.testing.assert_allclose(scores.outputs["samples"], expected)

    restored = reconstruct.execute({"model": model, "samples": scores.outputs["samples"]})
    assert restored is not None, reconstruct.error
    expected = [[13, 21], [8, 18]] if expected_count == 2 else [[13, 20], [8, 20]]
    np.testing.assert_allclose(restored.outputs["samples"], expected)


def test_pca_checks_effective_rank_and_model_shape():
    registry = builtin_registry()
    fit = registry.create("cv.fit_pca", parameters={"selection": "count"})
    assert fit.execute({"samples": [[0, 0, 0], [1, 2, 3]]}) is None
    assert "fitted rank of 1" in fit.error
    fit.parameters["selection"] = "variance"
    fit.parameters["minimum_variance"] = 1.0
    result = fit.execute({"samples": [[0, 0, 0], [1, 2, 3]]})
    assert result is not None, fit.error
    assert result.outputs["component_count"] == 1
    assert fit.execute({"samples": [[1, 1], [1, 1]]}) is None
    assert "requires variation" in fit.error

    transform = registry.create("cv.transform_pca")
    model = result.outputs["model"]
    assert transform.execute({"model": model, "samples": [[1, 2]]}) is None
    assert "expects 3 features" in transform.error
    bad_basis = {**model, "basis": [[1, 1, 1]]}
    assert transform.execute({"model": bad_basis, "samples": [[1, 2, 3]]}) is None
    assert "orthonormal basis" in transform.error


def test_pca_reconstruction_respects_output_table_limit():
    model = {"model": "principal_components_v1", "mean": [0] * 128, "basis": [[1] + [0] * 127]}
    block = builtin_registry().create("cv.transform_pca", parameters={"direction": "reconstruct"})
    assert block.execute({"model": model, "samples": [[0]] * 16_000}) is None
    assert "output exceeds 2 million values" in block.error
