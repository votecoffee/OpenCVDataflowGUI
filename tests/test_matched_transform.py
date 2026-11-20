"""Qualify metric registration, model records and degenerate correspondence handling."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.matched_transform import EstimateMatchedTransform
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.pointclouds import PointCloud

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/matched_3d_transforms"
ROTATION = Rotation.from_rotvec([0.27, -0.4, 0.34]).as_matrix()
SHIFT = np.array([0.173, -0.104, 0.331])
MODELS = {
    "translation": np.eye(3),
    "rigid": ROTATION,
    "similarity": 1.37 * ROTATION,
    "affine": np.array([[1.2, 0.17, -0.09], [0.14, 0.8, 0.18], [-0.05, 0.06, 1.15]]),
}
_PORT = {"translation": "pose", "rigid": "pose", "similarity": "similarity", "affine": "affine"}


def _linear(record):
    if record["model"] == "pose_3d_v1":
        return np.asarray(record["source_to_target"])[:3, :3]
    if record["model"] == "similarity_3d_v1":
        return record["scale"] * np.asarray(record["rotation"])
    assert record["model"] == "affine_3d_v1"
    return np.asarray(record["linear"])


@pytest.mark.parametrize("model", MODELS)
def test_mixed_unit_noisy_fit_rejects_known_outliers_and_emits_only_correct_type(model):
    rng = np.random.default_rng(510)
    source = rng.uniform(-0.16, 0.16, (41, 3))
    linear = MODELS[model]
    target = source @ linear.T + SHIFT + rng.normal(0, 2e-6, source.shape)
    bad = [0, 4, 7, 12, 14, 21, 25, 32, 36]
    target[bad] += rng.uniform(0.04, 0.2, (len(bad), 3))
    result = EstimateMatchedTransform(
        parameters={"model": model, "output_units": "mm", "minimum_inlier_fraction": 0.75}
    ).execute(
        {
            "source": PointCloud(source * 1000, "mm", "source_cam"),
            "target": PointCloud(target, "m", "target_fiducial"),
        }
    )
    assert result.passed
    for port in ("pose", "similarity", "affine"):
        assert (result.outputs[port] is not NO_VALUE) == (port == _PORT[model])
    record = result.outputs[_PORT[model]]
    assert (record["source_frame"], record["target_frame"], record["units"]) == (
        "source_cam",
        "target_fiducial",
        "mm",
    )
    np.testing.assert_allclose(record["translation"], list(SHIFT * 1000), atol=0.003)
    np.testing.assert_allclose(_linear(record), linear, atol=3e-5)
    assert result.outputs["inlier_pairs"] == [[i, i] for i in range(41) if i not in bad]
    assert result.outputs["report"]["inlier_rmse_m"] < 5e-6


def test_explicit_ordered_pairs_use_original_row_ids_and_validate_uniqueness():
    source = np.array([[0, 0, 0], [0.14, 0, 0], [0, 0.1, 0]])
    correct = source @ ROTATION.T + SHIFT
    target = np.vstack((correct[[2, 0, 1]], [5, 5, 5]))
    a, b = PointCloud(source, "m", "a"), PointCloud(target, "m", "b")
    pairs = [[0, 1], [1, 2], [2, 0]]
    tool = EstimateMatchedTransform()
    result = tool.execute({"source": a, "target": b, "pairs": pairs})
    assert result.passed and result.outputs["inlier_pairs"] == pairs
    np.testing.assert_allclose(_linear(result.outputs["pose"]), ROTATION, atol=2e-7)
    assert tool.execute({"source": a, "target": b, "pairs": [[0, 1], [0, 2], [2, 0]]}) is None
    assert "at most one pair" in tool.error
    assert tool.execute({"source": a, "target": b, "pairs": [[True, 1]]}) is None
    assert "integer row indices" in tool.error


def test_linear_and_planar_degeneracy_withholds_unsupported_transform():
    line = PointCloud([[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]], "mm", "a")
    planar = PointCloud([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], "mm", "b")
    for model, cloud in (("rigid", line), ("similarity", line), ("affine", planar)):
        result = EstimateMatchedTransform(parameters={"model": model}).execute(
            {
                "source": cloud,
                "target": cloud,
            }
        )
        assert not result.passed
        assert result.fail_reasons[0]["code"] == "matched_transform_degenerate_geometry"
        assert result.outputs[_PORT[model]] is NO_VALUE


def test_reflected_cloud_retains_affine_fit_and_withholds_rigid_pose():
    source = np.random.default_rng(714).uniform(-0.2, 0.2, (26, 3))
    reflection = ROTATION * [-1, 1, 1]
    a = PointCloud(source, frame="clean")
    b = PointCloud(source @ reflection.T + SHIFT, frame="mirror")
    inputs = {"source": a, "target": b}
    for model in ("rigid", "similarity"):
        value = EstimateMatchedTransform(
            parameters={
                "model": model,
                "minimum_inlier_fraction": 0.95,
            }
        ).execute(inputs)
        assert not value.passed and value.outputs[_PORT[model]] is NO_VALUE
    affine = EstimateMatchedTransform(parameters={"model": "affine"}).execute(inputs)
    assert affine.passed
    np.testing.assert_allclose(_linear(affine.outputs["affine"]), reflection, atol=1e-7)


def test_translation_handles_large_origins_and_rejects_unitless_inputs():
    source = np.random.default_rng(187).integers(-50, 50, (24, 3)) / 64 + [1e5, 5e4, -1e5]
    translation = np.array([0.125, -0.0625, 0.25])
    target = source + translation
    target[:4] += 0.125
    tool = EstimateMatchedTransform(parameters={"model": "translation", "inlier_threshold_m": 1e-6})
    report = tool.execute({"source": PointCloud(source), "target": PointCloud(target)})
    assert report.passed and report.outputs["report"]["inlier_count"] == 20
    np.testing.assert_array_equal(report.outputs["pose"]["translation"], translation)
    assert (
        tool.execute(
            {
                "source": PointCloud(source, "unitless"),
                "target": PointCloud(target),
            }
        )
        is None
    )
    assert "metric units" in tool.error


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_example_run_qualifies_all_four_models(profile):
    graph = Graph.load(EXAMPLE / "matched_3d_transforms.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        for model, port in _PORT.items():
            outputs = graph.blocks[model].last_result.outputs
            assert outputs["report"]["inlier_count"] == 21
            assert outputs["report"]["max_inlier_error_m"] < 1e-7
            assert outputs[port]["target_frame"] == f"{model}_station"
            np.testing.assert_allclose(outputs[port]["translation"], [170, -110, 310], atol=5e-5)
        assert graph.blocks["similarity"].last_result.outputs["similarity"][
            "scale"
        ] == pytest.approx(1.24)
    finally:
        graph.close()
