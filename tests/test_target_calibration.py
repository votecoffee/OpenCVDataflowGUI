"""Check released geometry, independent scoring and calibration result contracts."""

import copy
import json
from pathlib import Path
from runpy import run_path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.calibration import camera_parameters
from opencv_dataflow.core import Graph
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.target_calibration import target_observations

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/refine_calibration_target"


@pytest.fixture
def observations():
    return json.loads((EXAMPLE / "assets/observations.json").read_text(encoding="utf-8"))


def calibration_block(**parameters):
    return builtin_registry().create(
        "calibration.refine_target",
        parameters={
            "fixed_point_index": 6,
            "measured_anchor_distance": 121.2,
            "fix_k3": True,
            **parameters,
        },
    )


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_released_geometry_improves_heldout_error_with_measured_scale(profile):
    truth = run_path(str(EXAMPLE / "build_example.py"))
    _, expected = truth["fixture"]()
    graph = Graph.load(EXAMPLE / "refine_calibration_target.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        released, control = (graph.blocks[key].last_result for key in ("release", "nominal"))
        assert released and released.passed
        assert control and not control.passed
        report = released.outputs["report"]
        target = released.outputs["target"]
        assert released.outputs["holdout_rms"] < 0.07 and control.outputs["holdout_rms"] > 0.50
        assert report["fit"]["rms_pixels"] < 0.05
        assert len(report["fit"]["views"]) == 13 and len(report["holdout_views"]) == 3
        assert report["fitted_anchor_distance"] == pytest.approx(121.2, abs=1e-5)
        assert report["nominal_anchor_distance"] == 120.0
        assert report["nominal_scale"] == pytest.approx(1.01)
        xyz = np.array(target["object_points"])
        assert np.linalg.norm(xyz - expected, axis=1).max() < 0.12
        fixed = np.array(target["fixed_axes"])
        sd = np.array(target["coordinate_stddev"], dtype=float)
        assert fixed.sum() == 7 and fixed[[0, 6]].all() and fixed[-1, 2]
        assert np.isnan(sd[fixed]).all() and (sd[~fixed] > 0).all()
        nominal = np.array(target["nominal_points"])
        np.testing.assert_array_equal(xyz[fixed], nominal[fixed])
        assert control.outputs["target"]["coordinate_stddev"] is None
        computed, _ = camera_parameters(released.outputs["calibration"])
        np.testing.assert_allclose(computed, truth["CAMERA"], atol=4.0, rtol=0)
        assert ("Target offsets" in released.graphics) == (profile == "debug")
    finally:
        graph.close()


@pytest.mark.local_examples
def test_holdout_observations_never_enter_the_native_fit_or_leak_into_pose_points(
    observations, monkeypatch
):
    native = cv2.calibrateCameraROExtended
    native_pose = cv2.solvePnP
    fit_pixels, pose_pixels = [], []

    def fit(objects, pixels, *args, **kwargs):
        assert args[1] == 6
        assert all(np.array_equal(objects[0], points) for points in objects)
        assert objects[0][6, 0] == pytest.approx(121.2)
        fit_pixels.append(np.array(pixels))
        return native(objects, pixels, *args, **kwargs)

    def pose(object_points, image_points, *args, **kwargs):
        pose_pixels.append(image_points.copy())
        return native_pose(object_points, image_points, *args, **kwargs)

    monkeypatch.setattr(cv2, "calibrateCameraROExtended", fit)
    monkeypatch.setattr(cv2, "solvePnP", pose)
    block = calibration_block(max_holdout_rms_pixels=5, max_holdout_view_rms_pixels=0.5)
    original = block.execute({"observations": observations})
    assert original and original.passed
    report = original.outputs["report"]
    changed = copy.deepcopy(observations)
    view = next(row for row in changed["views"] if row["role"] == "holdout")
    for row in report["holdout_scored_indices"]:
        view["image_points"][row][0] += 1.4
    altered = block.execute({"observations": changed})
    assert altered and not altered.passed
    previous, updated = (value.outputs["calibration"] for value in (original, altered))
    for field in ("camera_matrix", "distortion", "rms_pixels"):
        np.testing.assert_allclose(updated[field], previous[field], atol=1e-6, rtol=1e-7)
    assert altered.outputs["report"]["worst_holdout_view"] == view["id"]
    assert altered.outputs["report"]["worst_holdout_rms_pixels"] > 1.25
    assert altered.outputs["holdout_rms"] < 1.0
    assert set(report["holdout_scored_indices"]).isdisjoint(report["holdout_pose_indices"])
    np.testing.assert_array_equal(pose_pixels[:3], pose_pixels[3:])
    expected = np.asarray(
        [row["image_points"] for row in observations["views"] if row["role"] == "fit"],
        np.float32,
    )
    np.testing.assert_array_equal(fit_pixels, [expected, expected])


@pytest.mark.local_examples
def test_observation_validation_rejects_duplicate_or_partial_views(observations):
    repeated = copy.deepcopy(observations)
    repeated["views"][-1]["image_points"] = repeated["views"][0]["image_points"]
    with pytest.raises(ValidationError, match="duplicates"):
        target_observations(repeated)
    observations["views"][0]["image_points"].pop()
    with pytest.raises(ValidationError, match="distinct, finite, noncollinear"):
        target_observations(observations)


@pytest.mark.local_examples
def test_invalid_gauge_or_pose_rows_fail_without_entering_native_fit(observations, monkeypatch):
    monkeypatch.setattr(cv2, "calibrateCameraROExtended", lambda *a, **k: pytest.fail("native fit"))
    collinear = calibration_block(fixed_point_index=17)
    assert collinear.execute({"observations": observations}) is None
    assert "noncollinear" in collinear.error
    impossible = calibration_block(pose_point_indices=[0, 1, 2, 3, 4, 35])
    assert impossible.execute({"observations": observations}) is None
    assert "pose points" in impossible.error


@pytest.mark.local_examples
def test_invalid_native_uncertainty_is_contained_and_block_recovers(observations, monkeypatch):
    native = cv2.calibrateCameraROExtended

    def invalid(*args, **kwargs):
        result = list(native(*args, **kwargs))
        result[8] = np.full_like(result[8], np.nan)
        return result

    block = calibration_block()
    with monkeypatch.context() as patch:
        patch.setattr(cv2, "calibrateCameraROExtended", invalid)
        assert block.execute({"observations": observations}) is None
        assert "uncertainty" in block.error
    result = block.execute({"observations": observations})
    assert result and result.passed and not block.error


@pytest.mark.local_examples
def test_measured_scale_is_reported_in_meters_when_requested(observations):
    observations["units"] = "m"
    observations["object_points"] = (np.asarray(observations["object_points"]) / 1000).tolist()
    block = calibration_block(measured_anchor_distance=0.1212)
    result = block.execute({"observations": observations})
    assert result and result.passed, block.error
    report = result.outputs["report"]
    assert report["units"] == result.outputs["calibration"]["units"] == "m"
    assert report["fitted_anchor_distance"] == pytest.approx(0.1212, abs=1e-8)
    assert 0 < report["max_coordinate_stddev"] < 0.0001
    assert report["holdout_rms_pixels"] < 0.07
