"""Verify capture geometry and compare repetitions against a held-out target."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.calibration_capture import AuditCalibrationCapture
from opencv_dataflow.core import Graph

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/calibration_capture_audit"


@pytest.fixture(scope="module")
def audited():
    graph = Graph.load(EXAMPLE / "calibration_capture_audit.json", builtin_registry())
    try:
        graph.run()
        expected = ("fit", "repeat", "holdout", "repeat_holdout", "audit")
        output = {}
        for key in expected:
            block = graph.blocks[key]
            assert block.last_result and block.last_result.passed, (key, block.error)
            output[key] = block.last_result
        return output
    finally:
        graph.close()


@pytest.mark.local_examples
def test_repeat_capture_audit_has_diversity_and_held_out_accuracy(audited):
    truth = json.loads((EXAMPLE / "assets/truth.json").read_text(encoding="utf-8"))
    result = audited["audit"]
    report = result.outputs["report"]
    assert result.passed and report["passed"] is True
    assert set(result.graphics) == {"Primary capture coverage", "Repeat capture coverage"}
    assert all(image.shape == (6 * 52, 8 * 52, 3) for image in result.graphics.values())
    for set_name, key in (("primary", "fit"), ("repeat", "repeat")):
        coverage = report[set_name]
        assert coverage["occupied_cell_fraction"] == coverage["occupied_edge_fraction"] == 1.0
        assert coverage["views"] == 10 and coverage["outside_image_points"] == 0
        assert 0.6 < coverage["depth_span_fraction"] < 0.75
        assert min(coverage["tilt_span_xy_degrees"]) > 30
        assert np.max(coverage["views_per_cell"]) <= 10
        camera = audited[key].outputs["calibration"]["camera_matrix"]
        np.testing.assert_allclose(camera, truth["camera_matrix"], atol=1.7, rtol=0)
        assert audited[key].outputs["rms"] < 0.09
    assert report["comparison"]["reused_views"] == 0
    assert report["comparison"]["samples"] == 63
    assert 0.75 < report["comparison"]["max_drift_pixels"] < 1.9
    assert max(map(abs, report["comparison"]["focal_shift_xy_percent"])) < 0.05
    measured = audited["holdout"].outputs["report"]["points"]
    centers = np.array([point["observed"] for point in measured])
    expected = np.array(truth["holdout"]["observed_truth_px"])
    assert np.max(np.min(np.linalg.norm(centers[:, None] - expected, axis=2), axis=1)) < 0.22
    assert report["holdout"]["rms_pixels"] < 0.09
    assert report["repeat_holdout"]["rms_pixels"] < 0.09
    assert report["holdout"]["observation_sha256"] == report["repeat_holdout"]["observation_sha256"]


@pytest.mark.local_examples
def test_weak_capture_fails_audit_despite_low_rms_in_lean_mode():
    graph = Graph.load(EXAMPLE / "weak_capture.json", builtin_registry())
    graph.set_run_profile("lean")
    try:
        graph.run()
        fitted, audit = graph.blocks["fit"], graph.blocks["audit"]
        assert fitted.last_result and fitted.last_result.passed, fitted.error
        assert fitted.last_result.outputs["rms"] < 0.09
        assert audit.last_result and not audit.last_result.passed, audit.error
        report = audit.last_result.outputs["report"]["primary"]
        assert report["occupied_cell_fraction"] == 6 / 48
        assert report["occupied_edge_fraction"] == 0
        assert not audit.last_result.graphics
        assert {reason["code"] for reason in audit.last_result.fail_reasons} == {
            f"calibration_capture_primary_{name}"
            for name in ("image_coverage", "edge_coverage", "tilt_span", "depth_span")
        }
    finally:
        graph.close()


@pytest.mark.local_examples
def test_repeat_projection_uses_same_rays_and_rejects_reused_views(audited):
    primary = audited["fit"].outputs["fit_report"]
    repeat = copy.deepcopy(audited["repeat"].outputs["fit_report"])
    repeat["capture_optics"] = copy.deepcopy(primary["capture_optics"])
    repeat["capture_optics"]["camera_matrix"][0][2] += 3.0
    block = AuditCalibrationCapture()
    result = block.execute({"fit_report": primary, "repeat_fit_report": repeat})
    assert result and not result.passed, block.error
    assert result.outputs["report"]["comparison"]["max_drift_pixels"] == pytest.approx(3.0)
    assert [reason["code"] for reason in result.fail_reasons] == [
        "calibration_capture_repeat_drift"
    ]

    result = block.execute({"fit_report": primary, "repeat_fit_report": primary})
    assert result and not result.passed
    comparison = result.outputs["report"]["comparison"]
    assert comparison["max_drift_pixels"] == pytest.approx(0.0)
    assert comparison["reused_views"] == 10
    assert [reason["code"] for reason in result.fail_reasons] == [
        "calibration_capture_repeat_reused"
    ]


@pytest.mark.local_examples
def test_withheld_observations_are_bound_to_the_fit_and_reject_reuse(audited):
    fit = audited["fit"].outputs["fit_report"]
    holdout = copy.deepcopy(audited["holdout"].outputs["report"])
    block = AuditCalibrationCapture()
    holdout["observation_sha256"] = fit["views"][3]["observation_sha256"]
    result = block.execute({"fit_report": fit, "holdout": holdout})
    assert result and not result.passed
    assert result.fail_reasons[0]["code"] == "calibration_capture_primary_holdout_reused"

    holdout["optics_sha256"] = audited["repeat_holdout"].outputs["report"]["optics_sha256"]
    assert block.execute({"fit_report": fit, "holdout": holdout}) is None
    assert "connected fit optics" in block.error


@pytest.mark.local_examples
def test_audit_checks_measured_holdouts_and_stable_image_dimensions(audited):
    fit = audited["fit"].outputs["fit_report"]
    repeat = copy.deepcopy(audited["repeat"].outputs["fit_report"])
    block = AuditCalibrationCapture()
    repeat["capture_optics"]["image_size"][0] += 1
    assert block.execute({"fit_report": fit, "repeat_fit_report": repeat}) is None
    assert "same image dimensions" in block.error

    holdout = copy.deepcopy(audited["holdout"].outputs["report"])
    holdout["rms_pixels"] = 0.6
    result = block.execute({"fit_report": fit, "holdout": holdout})
    assert result and not result.passed
    assert result.fail_reasons[0]["code"] == "calibration_capture_primary_holdout_rms"

    holdout["status"] = "calibration_validation_grid_not_found"
    result = block.execute({"fit_report": fit, "holdout": holdout})
    assert result and not result.passed
    assert result.outputs["report"]["holdout"]["rms_pixels"] is None
    assert result.fail_reasons[0]["code"] == "calibration_capture_primary_holdout_unmeasured"


@pytest.mark.local_examples
def test_validation_pair_uses_the_same_held_out_view(audited):
    repeat_holdout = copy.deepcopy(audited["repeat_holdout"].outputs["report"])
    repeat_holdout["observation_sha256"] = "7" * 64
    block = AuditCalibrationCapture()
    result = block.execute(
        {
            "fit_report": audited["fit"].outputs["fit_report"],
            "repeat_fit_report": audited["repeat"].outputs["fit_report"],
            "holdout": audited["holdout"].outputs["report"],
            "repeat_holdout": repeat_holdout,
        }
    )
    assert result and not result.passed
    assert result.fail_reasons[0]["code"] == "calibration_capture_holdout_mismatch"


@pytest.mark.local_examples
def test_using_a_training_photo_as_holdout_fails_the_saved_graph():
    graph = Graph.load(EXAMPLE / "calibration_capture_audit.json", builtin_registry())
    graph.blocks["independent"].parameters["path"] = "assets/primary-01.png"
    try:
        graph.run()
        audit = graph.blocks["audit"]
        assert audit.last_result and not audit.last_result.passed, audit.error
        assert {reason["code"] for reason in audit.last_result.fail_reasons} == {
            "calibration_capture_primary_holdout_reused",
            "calibration_capture_repeat_holdout_reused",
        }
    finally:
        graph.close()


@pytest.mark.local_examples
def test_audit_reports_upstream_quality_failure(audited):
    fit = copy.deepcopy(audited["fit"].outputs["fit_report"])
    fit["passed"] = False
    holdout = copy.deepcopy(audited["holdout"].outputs["report"])
    holdout["passed"] = False
    result = AuditCalibrationCapture().execute({"fit_report": fit, "holdout": holdout})
    assert result and not result.passed
    assert {reason["code"] for reason in result.fail_reasons} == {
        "calibration_capture_primary_fit_failed",
        "calibration_capture_primary_holdout_validation_failed",
    }
