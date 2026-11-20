"""Exercise native rig calibration, held-out predictions and frame conventions."""

import copy
import json
import runpy
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import camera_record, pose_record
from opencv_dataflow.multicamera_calibration import CalibrateMulticameraRig, ValidateMulticameraRig
from opencv_dataflow.payloads import NO_VALUE

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/multicamera_rig"
KNOWN_TRANSLATIONS = np.array([[0, 0, 0], [-0.17, 0.013, 0.024], [0.19, 0.041, -0.018]])
KNOWN_ROTATIONS = np.array([[0, 0, 0], [0.018, 0.064, -0.014], [-0.036, -0.055, 0.021]])
EXPECTED_PAIRS = [["camera_a", "camera_b"], ["camera_b", "camera_c"]]


def _assets():
    return tuple(
        json.loads((EXAMPLE / "assets" / f"{name}.json").read_text(encoding="utf-8"))
        for name in ("train", "holdout")
    )


def _assert_known_rig(rig, scale=1, max_translation=0.001):
    assert rig["anchor_frame"] == "camera_a"
    records = rig["anchor_to_camera"]
    assert [p["target_frame"] for p in records] == ["camera_a", "camera_b", "camera_c"]
    assert all(p["source_frame"] == "camera_a" and p["units"] == rig["units"] for p in records)
    recovered = np.array([p["source_to_target"] for p in records])
    np.testing.assert_array_equal(recovered[0], np.eye(4))
    expected = np.repeat(np.eye(4)[None], 3, axis=0)
    expected[:, :3, :3] = [cv2.Rodrigues(rv)[0] for rv in KNOWN_ROTATIONS]
    expected[:, :3, 3] = KNOWN_TRANSLATIONS * scale
    np.testing.assert_allclose(recovered[:, :3, :3], expected[:, :3, :3], atol=0.0015)
    np.testing.assert_allclose(
        recovered[:, :3, 3], expected[:, :3, 3], atol=max_translation * scale, rtol=0
    )
    measured_bc = recovered[2] @ np.linalg.inv(recovered[1])
    expected_bc = expected[2] @ np.linalg.inv(expected[1])
    np.testing.assert_allclose(measured_bc[:3, :3], expected_bc[:3, :3], atol=0.0025)
    np.testing.assert_allclose(
        measured_bc[:3, 3], expected_bc[:3, 3], atol=2 * max_translation * scale, rtol=0
    )
    np.testing.assert_allclose(
        np.linalg.inv(recovered[2]) @ measured_bc @ recovered[1], np.eye(4), atol=1e-10
    )


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_chain_recovers_invisible_end_to_end_pair_and_cross_validates(profile):
    graph = Graph.load(EXAMPLE / "multicamera_rig.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        fitter, validator = graph.blocks["fit"], graph.blocks["holdout_check"]
        fit, valid = fitter.last_result, validator.last_result
        assert fit and fit.passed, fitter.error
        assert valid and valid.passed, validator.error
        rig, fit_report = fit.outputs["rig"], fit.outputs["report"]
        _assert_known_rig(rig, max_translation=0.00035)
        for i, model in enumerate(rig["cameras"]):
            assert model["camera_matrix"][0][0] == pytest.approx(676 + 22 * i, abs=0.5)
            assert model["camera_matrix"][1][1] == pytest.approx(681 + 27 * i, abs=0.5)
        assert fit_report["initialization_pairs"] == EXPECTED_PAIRS
        assert fit_report["shared_captures"] == [[10, 10, 0], [10, 20, 10], [0, 10, 10]]
        assert fit_report["per_camera_capture_rms_px"][0][10:] == [None] * 10
        assert fit_report["rms_px"] == pytest.approx(0.027829, abs=0.0001)
        holdout = valid.outputs["report"]
        assert sorted(set(rig["training_capture_ids"]) & set(holdout["capture_ids"])) == []
        assert len(holdout["cross_projections"]) == 12
        assert all(row["fit_camera"] != row["test_camera"] for row in holdout["cross_projections"])
        assert holdout["rms_px"] == pytest.approx(0.034694, abs=0.001)
        if profile == "debug":
            graphic = fit.graphics["reprojection_error"]
            assert graphic.shape == (72, 240, 3)
            assert graphic[0, :120].any() and not graphic[0, 120:].any()
        else:
            assert not fit.graphics and not valid.graphics
    finally:
        graph.close()


@pytest.mark.local_examples
@pytest.mark.parametrize(
    ("lens", "mode", "initialization", "pinhole_distortion"),
    [
        ("pinhole", "fit", "register_cameras", "rational"),
        ("pinhole", "fixed", "stereo_calibrate", "standard"),
        ("fisheye", "fit", "register_cameras", "standard"),
        ("fisheye", "fixed", "stereo_calibrate", "standard"),
    ],
)
def test_lens_and_intrinsics_options_in_millimeters(lens, mode, initialization, pinhole_distortion):
    builder = runpy.run_path(str(EXAMPLE / "build_example.py"))
    records = [
        builder["observations"](label, count, seed, lens)
        for label, count, seed in (("train", 20, 85147), ("held", 6, 15198))
    ]
    for record in records:
        record["units"] = "mm"
        record["object_points"] = (np.array(record["object_points"]) * 1000).tolist()
    train, held = records
    inputs = {"observations": train}
    if mode == "fixed":
        inputs["camera_models"] = [
            camera_record(k, [0] * (4 if lens == "fisheye" else 5), (1280, 960), "mm", frame, lens)
            for k, frame in zip(builder["optics"](), train["camera_frames"], strict=True)
        ]
    block = CalibrateMulticameraRig(
        parameters={
            "lens": lens,
            "intrinsics": mode,
            "initialization": initialization,
            "pinhole_distortion": pinhole_distortion,
        }
    )
    fit = block.execute(inputs)
    assert fit and fit.passed, block.error
    rig = fit.outputs["rig"]
    _assert_known_rig(rig, scale=1000)
    assert rig["lens"] == lens and rig["units"] == "mm"
    if mode == "fixed":
        assert rig["cameras"] == inputs["camera_models"]
    elif pinhole_distortion == "rational":
        assert all(len(item["distortion"]) >= 8 for item in rig["cameras"])
    evaluator = ValidateMulticameraRig()
    checked = evaluator.execute({"rig": rig, "observations": held})
    assert checked and checked.passed, evaluator.error
    assert checked.outputs["report"]["rms_px"] < 0.15


@pytest.mark.local_examples
def test_invalid_observations_are_rejected_before_native_fit(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid visibility or geometry must stop before the native fit")

    monkeypatch.setattr(cv2, "calibrateMultiviewExtended", unexpected)
    train, _ = _assets()
    disconnected = copy.deepcopy(train)
    disconnected["visibility"][1][10:] = [0] * 10
    malformed = copy.deepcopy(train)
    malformed["image_points"][0][0][4] = [-1, -1]
    duplicate_ids = copy.deepcopy(train)
    duplicate_ids["capture_ids"][1] = duplicate_ids["capture_ids"][0]
    block = CalibrateMulticameraRig()
    for record, message in (
        (disconnected, "connected shared captures"),
        (malformed, "full ordered"),
        (duplicate_ids, "distinct names"),
    ):
        assert block.execute({"observations": record}) is None
        assert message in block.error


@pytest.mark.local_examples
def test_quality_gate_retains_report_and_withholds_failed_rig():
    train, _ = _assets()
    block = CalibrateMulticameraRig(parameters={"max_view_rms_px": 0.01, "max_rms_px": 0.01})
    fit = block.execute({"observations": train})
    assert fit is not None and not fit.passed and not block.error
    assert fit.outputs["rig"] is NO_VALUE
    assert fit.outputs["report"]["rms_px"] > 0.01
    assert {r["code"] for r in fit.fail_reasons} == {
        "multicamera_fit_rms",
        "multicamera_fit_view_rms",
    }


@pytest.mark.local_examples
def test_holdout_rejects_reused_observation_ids_and_independently_detects_pose_error():
    train, held = _assets()
    fit = CalibrateMulticameraRig().execute({"observations": train})
    assert fit and fit.passed
    block = ValidateMulticameraRig()
    assert block.execute({"rig": fit.outputs["rig"], "observations": train}) is None
    assert "disjoint" in block.error
    rig = copy.deepcopy(fit.outputs["rig"])
    t = np.array(rig["anchor_to_camera"][2]["source_to_target"])
    t[0, 3] += 0.03
    rig["anchor_to_camera"][2] = pose_record(t, "camera_a", "camera_c", "m")
    check = block.execute({"rig": rig, "observations": held})
    assert check is not None and not check.passed and not block.error
    report = check.outputs["report"]
    assert report["rms_px"] > 5.0
    assert {r["code"] for r in check.fail_reasons} == {
        "multicamera_holdout_rms",
        "multicamera_holdout_view_rms",
    }
    missing = copy.deepcopy(held)
    missing["visibility"][2] = [0] * len(missing["capture_ids"])
    check = block.execute({"rig": fit.outputs["rig"], "observations": missing})
    assert check and not check.passed
    assert {r["code"] for r in check.fail_reasons} == {"multicamera_holdout_camera_unverified"}
