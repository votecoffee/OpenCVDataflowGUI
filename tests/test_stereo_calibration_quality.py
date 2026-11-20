"""Verify native stereo fit diagnostics and independently observed holdouts."""

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import require_pinhole, rigid
from opencv_dataflow.payloads import NO_VALUE, ValuePacket

FIT = "3d.stereo_calibrate"
CHECK = "3d.validate_stereo_calibration"
ROOT = Path(__file__).resolve().parents[1] / "examples/opencv tools/stereo_calibration_quality"


def project(xyz, camera, rotation, translation):
    k, d, _ = require_pinhole(camera)
    return cv2.projectPoints(np.asarray(xyz, float), cv2.Rodrigues(rotation)[0], translation, k, d)[
        0
    ].reshape(-1, 2)


def rms(errors):
    return float(np.sqrt(np.mean(np.sum(errors**2, axis=-1))))


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_fit_and_holdout_reproduce_known_stereo_rig(profile):
    registry = builtin_registry()
    graph = Graph.load(ROOT / "stereo_calibration_quality.json", registry)
    try:
        graph.set_run_profile(profile)
        graph.run()
        fit, validate = (graph.blocks[name] for name in ("fit", "validate"))
        for block in (fit, validate):
            assert block.last_result and block.last_result.passed, block.error
        training = graph.recipes.resolve()["settings"]["fit"]
        heldout = graph.recipes.resolve()["settings"]["validation"]
        assets = ROOT / "assets"
        truth = json.loads((assets / "truth.json").read_bytes())
        payload = fit.last_result.outputs
        rig, report = payload["rig"], payload["fit_report"]
        transform = rigid(rig["transform"])
        expected = np.asarray(truth["left_to_right"])
        relative = transform[:3, :3] @ expected[:3, :3].T
        assert np.linalg.norm(transform[:3, 3] - expected[:3, 3]) < 0.20
        assert np.degrees(Rotation.from_matrix(relative).magnitude()) < 0.025
        assert 0.08 < report["rms_pixels"] < 0.10 and len(report["views"]) == 10
        weights = np.array([row["points"] for row in report["views"]])
        assert len(np.unique(weights)) > 1

        # Reproject every fit view using its reported left target pose and the rig.
        for row in report["views"]:
            idx = row["index"]
            l_pose = np.eye(4)
            l_pose[:3, :3] = cv2.Rodrigues(np.asarray(row["target_to_left_rvec"]))[0]
            l_pose[:3, 3] = row["target_to_left_translation"]
            xyz = training["object_views"][idx]
            for side, pose in (("left", l_pose), ("right", transform @ l_pose)):
                xy = np.asarray(training[f"{side}_views"][idx], np.float32)
                predicted = project(xyz, rig[side], pose[:3, :3], pose[:3, 3])
                assert row[f"{side}_rms_pixels"] == pytest.approx(rms(predicted - xy), abs=1e-7)
        for side in ("left", "right"):
            errors = np.asarray([row[f"{side}_rms_pixels"] for row in report["views"]])
            pooled = np.sqrt(np.average(errors**2, weights=weights))
            assert report[f"{side}_rms_pixels"] == pytest.approx(pooled)
        assert report["rms_pixels"] == pytest.approx(
            np.hypot(report["left_rms_pixels"], report["right_rms_pixels"]) / np.sqrt(2)
        )

        checker = registry.create(
            "calibration.charuco", parameters={**truth["target"], "units": "mm"}
        )
        report = validate.last_result.outputs["report"]
        assert report["captures_measured"] == 3 and report["rms_pixels"] < 0.11
        for i, ids in enumerate(truth["heldout_ids"]):
            for side in ("left", "right"):
                data = np.frombuffer((assets / f"{side}-{i}.png").read_bytes(), np.uint8)
                image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
                detection = checker.execute({"image": image, "camera": rig[side]})
                assert detection and detection.passed
                observed = dict(
                    zip(detection.outputs["corner_ids"], detection.outputs["image_points"])
                )
                np.testing.assert_allclose(
                    [observed[key] for key in ids],
                    heldout[f"{side}_views"][i],
                    atol=1e-6,
                )
                assert report["views"][i][f"{side}_rms_pixels"] < 0.17
    finally:
        graph.close()


@pytest.fixture
def example_inputs():
    graph = Graph.load(ROOT / "stereo_calibration_quality.json", builtin_registry())
    try:
        graph.run()
        return tuple(copy.deepcopy(graph.blocks[key].last_inputs) for key in ("fit", "validate"))
    finally:
        graph.close()


@pytest.mark.local_examples
def test_single_camera_outlier_triggers_view_gate_below_joint_limit(example_inputs):
    values, _ = example_inputs
    values["right_views"][4][2][0] += 10
    values["right_views"][4][2][1] -= 8
    fit = builtin_registry().create(FIT, parameters={"max_view_rms": 0.6, "max_rms": 2.0})
    result = fit.execute(values)
    assert result and not result.passed, fit.error
    report = result.outputs["fit_report"]
    assert report["rms_pixels"] < 1
    assert report["worst_view_index"] == 4 and report["worst_view_camera"] == "right"
    assert report["views"][4]["right_rms_pixels"] > 2
    assert report["views"][4]["left_rms_pixels"] < 0.6
    assert [v["code"] for v in result.fail_reasons] == ["stereo_calibration_view_rms"]


@pytest.mark.local_examples
def test_refining_intrinsics_retains_legacy_solver_result(example_inputs):
    values, _ = example_inputs
    fit = builtin_registry().create(FIT, parameters={"fix_intrinsics": False})
    result = fit.execute(values)
    assert result and result.passed, fit.error
    objects, left, right = (
        [np.asarray(x, np.float32) for x in values[k]]
        for k in ("object_views", "left_views", "right_views")
    )
    k1, d1, size = require_pinhole(values["left_camera"])
    k2, d2, _ = require_pinhole(values["right_camera"])
    standard = cv2.stereoCalibrate(
        objects,
        left,
        right,
        k1,
        d1,
        k2,
        d2,
        size,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS,
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-6),
    )
    assert result.outputs["rms"] == pytest.approx(standard[0], abs=1e-9)
    recovered = rigid(result.outputs["rig"]["transform"])
    np.testing.assert_allclose(recovered[:3, :3], standard[5], atol=1e-8)
    np.testing.assert_allclose(recovered[:3, 3], standard[6].ravel(), atol=1e-8)


@pytest.mark.local_examples
def test_holdout_detects_bad_baseline_despite_good_single_camera_pose_fits(example_inputs):
    _, values = example_inputs
    values["rig"]["transform"]["source_to_target"][0][3] += 2.5
    block = builtin_registry().create(CHECK)
    result = block.execute(values)
    assert result and not result.passed
    report = result.outputs["report"]
    assert report["rms_pixels"] > 3
    for view in report["views"]:
        for side in ("left", "right"):
            assert view[f"{side}_anchor_rms_pixels"] < 0.1
            assert view[f"{side}_rms_pixels"] > 3
    assert {v["code"] for v in result.fail_reasons} == {
        "stereo_validation_cross_camera_rms",
        "stereo_validation_rms",
    }


@pytest.mark.local_examples
def test_many_invalid_poses_keep_detailed_report_and_bounded_ticket_reasons(example_inputs):
    _, values = example_inputs
    for key in ("object_views", "left_views", "right_views"):
        values[key] = [values[key][0]] * 40
    values["left_views"] = [[[120, 110]] * 24] * 40
    block = builtin_registry().create(CHECK)
    result = block.execute(values)
    assert result and not result.passed
    report = result.outputs["report"]
    assert report["captures_measured"] == 0 and len(report["views"]) == 40
    assert len(report["failures"]) == 40 and len(result.fail_reasons) <= 32
    assert result.outputs["rms"] is NO_VALUE


@pytest.mark.local_examples
def test_validates_units_and_grid_before_running_holdout_pose(example_inputs):
    _, values = example_inputs
    check = builtin_registry().create(CHECK)
    tagged = ValuePacket(values["object_views"], {"object_point_units": "m"})
    assert check.execute({**values, "object_views": tagged}) is None
    assert "units must match" in check.error
    values["right_views"][0][1][0] = -0.5
    assert check.execute(values) is None
    assert "outside its camera grid" in check.error
