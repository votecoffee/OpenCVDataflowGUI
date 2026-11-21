"""Qualify random-pattern metric coordinates, partial views and failure gating."""

import json
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.random_pattern import DetectRandomPatternTarget, GenerateRandomPatternTarget

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/random_pattern_calibration"


def _transform(points, matrix):
    pixels = np.column_stack((points, np.ones(len(points)))) @ np.asarray(matrix).T
    return pixels[:, :2] / pixels[:, 2:3]


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_partial_views_reproject_hidden_points_and_preserve_landmark_ids(profile):
    graph = Graph.load(EXAMPLE / "random_pattern_calibration.json", builtin_registry())
    graph.set_run_profile(profile)
    graph.set_branch_workers(2)
    truth = json.loads((EXAMPLE / "assets/truth.json").read_text(encoding="utf-8"))
    atlas = {}
    try:
        for view in truth["views"]:
            graph.run()
            detector, solver = (graph.blocks[key] for key in ("match", "pose"))
            result, solution = detector.last_result, solver.last_result
            assert result and result.passed, detector.error
            assert solution and solution.passed, solver.error
            data = result.outputs
            record = data["observation"]
            ids = data["feature_ids"]
            ref = np.asarray(record["reference_points"])
            obj = np.asarray(data["object_points"])
            uv = np.asarray(data["image_points"])
            assert len(ids) == len(set(ids)) >= 200 and ids == sorted(ids)
            assert record["target_coverage"] > 0.50 and record["rms_px"] < 0.55
            assert result.payloads["object_points"]["object_point_units"] == "mm"
            assert result.payloads["image_points"]["spatial"]["image_size"] == [640, 480]
            pixels = np.floor(ref + 0.5).astype(int)
            np.testing.assert_array_equal(ids, pixels[:, 1] * 900 + pixels[:, 0])
            np.testing.assert_allclose(obj[:, :2], (ref + 0.5) * (240.0 / 900))
            np.testing.assert_array_equal(obj[:, 2], 0)
            independent = _transform(ref, view["reference_to_image"])
            assert np.quantile(np.linalg.norm(uv - independent, axis=1), 0.95) < 0.9

            t = np.asarray(solution.outputs["pose"]["source_to_target"])
            holdout = np.asarray(view["holdout_xyz_mm"])
            hidden_uv = np.asarray(view["holdout_pixels"])
            xmin, ymin, xmax, ymax = view["occlusion_xyxy"]
            assert len(hidden_uv) >= 8
            assert (([xmin, ymin] < hidden_uv) & (hidden_uv < [xmax, ymax])).all()
            assert np.min(np.linalg.norm(holdout[:, None, :] - obj, axis=2)) > 1
            camera = (holdout @ t[:3, :3].T + t[:3, 3]) @ np.asarray(truth["camera_matrix"]).T
            predicted = camera[:, :2] / camera[:, 2:3]
            assert np.max(np.linalg.norm(predicted - hidden_uv, axis=1)) < 0.15
            np.testing.assert_allclose(t[:3, 3], view["translation_mm"], atol=0.25, rtol=0)
            np.testing.assert_allclose(t[:3, :3], view["rotation"], atol=0.003, rtol=0)
            assert bool(result.drawings) == (profile == "debug")

            current = dict(zip(ids, data["object_points"], strict=True))
            if atlas:
                shared = current.keys() & atlas.keys()
                assert len(shared) >= 100
                assert all(current[key] == atlas[key] for key in shared)
            atlas = current
    finally:
        graph.close()


@pytest.fixture(scope="module")
def target():
    block = GenerateRandomPatternTarget()
    result = block.execute({})
    assert result and result.passed, block.error
    return result.outputs


def test_reproducible_texture_and_print_scale(target):
    again = GenerateRandomPatternTarget().execute({}).outputs
    metric = (
        GenerateRandomPatternTarget(parameters={"width": 0.240, "height": 0.160, "units": "m"})
        .execute({})
        .outputs
    )
    other = GenerateRandomPatternTarget(parameters={"seed": 14}).execute({}).outputs
    np.testing.assert_array_equal(target["image"], again["image"])
    np.testing.assert_array_equal(target["image"], metric["image"])
    np.testing.assert_allclose(target["target"]["print_dpi_xy"], [95.25, 95.25])
    assert target["target"]["reference_sha256"] != other["target"]["reference_sha256"]
    assert again["target"] == target["target"]
    assert metric["target"]["units"] == "m"


@pytest.mark.parametrize("kind", ("blank", "foreign", "absent_mask"))
def test_missing_pattern_returns_diagnostics_and_withholds_geometry(kind, target):
    reference = target["image"]
    inputs = {"reference": reference, "target": target["target"], "image": reference}
    if kind == "blank":
        inputs["image"] = np.full_like(reference, 127)
    elif kind == "foreign":
        inputs["image"] = (
            GenerateRandomPatternTarget(parameters={"seed": 55}).execute({}).outputs["image"]
        )
    else:
        inputs["mask"] = np.zeros(reference.shape, np.uint8)
    match = DetectRandomPatternTarget()
    result = match.execute(inputs)
    assert result is not None and not result.passed, match.error
    assert result.outputs["image"].shape == reference.shape
    assert result.outputs["observation"]["passed"] is False
    assert {reason["code"] for reason in result.fail_reasons} >= {"random_pattern_count"}
    assert all(
        result.outputs[key] is NO_VALUE
        for key in ("feature_ids", "image_points", "object_points", "reference_to_image")
    )


def test_roi_limits_view_points_and_distinguishes_low_target_coverage(target):
    reference = target["image"]
    inputs = {
        "image": reference,
        "reference": reference,
        "target": target["target"],
        "region": {
            "model": "image_region_v1",
            "image_size": [900, 600],
            "shape": "rectangle",
            "rectangle": [190, 130, 320, 260],
        },
    }
    match = DetectRandomPatternTarget()
    result = match.execute(inputs)
    assert result and result.passed, match.error
    points = np.asarray(result.outputs["image_points"])
    assert (([189.5, 129.5] <= points) & (points < [510.5, 390.5])).all()
    assert 0.02 < result.outputs["observation"]["target_coverage"] < 0.20

    match.parameters["minimum_target_coverage"] = 0.25
    failed = match.execute(inputs)
    assert failed is not None and not failed.passed, match.error
    assert {reason["code"] for reason in failed.fail_reasons} == {"random_pattern_coverage"}
    assert failed.outputs["object_points"] is NO_VALUE


@pytest.mark.parametrize(
    "field, value", (("units", "cm"), ("target_size", [0, 1]), ("image_size", [902, 600]))
)
def test_malformed_target_is_a_validation_error(target, field, value):
    match = DetectRandomPatternTarget()
    image = target["image"]
    record = dict(target["target"], **{field: value})
    assert match.execute({"image": image, "reference": image, "target": record}) is None
    assert "Target dimensions" in match.error or "Reference raster" in match.error


def test_altered_reference_and_inconsistent_settings_fail_clearly(target):
    reference = target["image"].copy()
    reference[0, 0] ^= 255
    inputs = {"image": target["image"], "reference": reference, "target": target["target"]}
    match = DetectRandomPatternTarget()
    assert match.execute(inputs) is None
    assert "Reference raster" in match.error
    match = DetectRandomPatternTarget(parameters={"max_view_features": 32, "minimum_inliers": 33})
    assert match.execute(inputs) is None
    assert "view feature budget" in match.error
    invalid = GenerateRandomPatternTarget(parameters={"raster_width": 4096, "height": 240})
    assert invalid.execute({}) is None
    assert "4,194,304 pixels" in invalid.error
