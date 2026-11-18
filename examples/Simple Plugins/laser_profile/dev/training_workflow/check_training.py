"""Actual inspector calibration/learn/independent-preview/Stage/Apply/Save As checks."""

# Select the pinned package before importing host lazy plugin providers.
# ruff: noqa: E402, I001

import argparse
import copy
import importlib.util
import json
import shutil
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pinned_laser_training", ROOT / "plugin.py")
plugin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin)

import numpy as np
from opencv_dataflow.app import MainWindow
from opencv_dataflow.assistant_api import ProjectSession, make_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.pointclouds import PointCloud
from PySide6.QtWidgets import QApplication, QDialog

from opencv_simple_laser_profile import load_recipe
from opencv_simple_laser_profile.cli import save_image
from opencv_simple_laser_profile.qt_calibration import CalibrationReviewDialog

CASES = ("calibrated_step", "calibrated_unavailable", "real_wli1")


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )


def stable_result(result):
    return json.loads(
        json.dumps(
            dict(
                passed=bool(result.passed),
                score=result.score,
                message=result.message,
                fail_reasons=result.fail_reasons,
                outputs={
                    name: value
                    for name, value in result.outputs.items()
                    if value is not NO_VALUE and not isinstance(value, (np.ndarray, PointCloud))
                },
            ),
            allow_nan=False,
        )
    )


def agreement(expected, actual):
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        return max((agreement(expected[k], actual[k]) for k in expected), default=0.0)
    if isinstance(expected, list):
        assert len(expected) == len(actual)
        return max((agreement(a, b) for a, b in zip(expected, actual)), default=0.0)
    if isinstance(expected, float) and isinstance(actual, float):
        error = abs(expected - actual)
        assert error < 1e-10, (expected, actual)
        return error
    assert expected == actual, (expected, actual)
    return 0.0


def finish(app, dialog):
    deadline = time.monotonic() + 60
    while dialog.pending:
        app.processEvents()
        dialog.poll()
        assert time.monotonic() < deadline, dialog.status.text()
        time.sleep(0.002)


def verify_case(root, case, registry, app):
    folder, metric = root / "projects", case.startswith("calibrated_")
    project = folder / f"{case}.json"
    graph = Graph.load(project, registry)
    window = MainWindow()
    window.registry = registry
    warnings = []
    window.warn = warnings.append
    window.replace_graph(graph, project)
    window.show()
    block = graph.blocks["laser"]
    before = copy.deepcopy(block.parameters)
    original_bytes = (folder / before["recipe_path"]).read_bytes()
    expected = json.loads((ROOT / "expected.json").read_bytes())[case]
    inspector = window.inspector
    inspector.select(block)
    evidence = root / "evidence" / case
    evidence.mkdir(parents=True)
    record = dict(case=case, desktop_input_used=False, window_capture_used=False)
    try:
        graph.run()
        assert block.last_result is not None, block.error
        assert stable_result(block.last_result) == expected
        inspector.open_plugin_dialog()
        app.processEvents()
        dialog = inspector._plugin_dialog
        finish(app, dialog)
        original = dialog.draft.candidate
        assert original is not None and dialog.stage_button.isEnabled()
        np.testing.assert_array_equal(dialog.draft.reference, original.reference)
        assert bool(dialog.draft.camera) == metric
        dialog.review_calibration()
        review = dialog.children[-1]
        assert review.metadata.isReadOnly()
        assert review.probe_button.isEnabled() == metric
        if metric:
            review.probe()
            assert (
                "XYZ" in review.probe_result.text()
                and "Unavailable" not in review.probe_result.text()
            )
        review.reject()
        # Cancel preserves the applied recipe, reference and full-precision calibration.
        dialog.reject()
        assert block.parameters == before
        inspector.open_plugin_dialog()
        app.processEvents()
        dialog = inspector._plugin_dialog
        finish(app, dialog)
        if metric:
            observations = json.loads((ROOT / "observations.json").read_bytes())
            current = dialog.draft.candidate
            with patch.object(CalibrationReviewDialog, "exec", return_value=QDialog.Rejected):
                dialog.fit_calibration(observations)
                finish(app, dialog)
            assert dialog.draft.candidate is current
            with patch.object(CalibrationReviewDialog, "exec", return_value=QDialog.Accepted):
                dialog.fit_calibration(observations)
                finish(app, dialog)
            assert dialog.draft.candidate is None and not dialog.stage_button.isEnabled()
            fit = dialog.draft.metadata["calibration_evidence"]["calibration_fit"]
            assert len(fit["training_ids"]) == 2 and len(fit["validation"]) == 2
            assert all(v["available_points"] == v["supplied_points"] for v in fit["validation"])
            write_json(
                evidence / "calibration_fit.json", dialog.draft.metadata["calibration_evidence"]
            )
            record["calibration_fit"] = fit
        else:
            assert "PIXEL ONLY" in dialog.setup.text() and dialog.draft.laser is None
        # Temporarily change the ROI, then deliberately restore its exact original mask.
        ref = original.reference
        dialog.set_region((0, 0, ref.shape[1] - 1, ref.shape[0]))
        assert dialog.draft.candidate is None and not dialog.stage_button.isEnabled()
        dialog.draft.set_mask(original.mask)
        dialog.invalidate()
        dialog.start_learning()
        finish(app, dialog)
        assert dialog.stage_button.isEnabled(), dialog.status.text()
        assert dialog.draft.candidate.settings == original.settings
        np.testing.assert_array_equal(dialog.draft.reference, original.reference)
        if original.mask is not None:
            np.testing.assert_array_equal(dialog.draft.mask, original.mask)
        reference_health = dialog.reference_health.text()
        reference_report = copy.deepcopy(dialog.draft.result["report"])
        dialog.preview_connected()
        finish(app, dialog)
        assert dialog.reference_health.text() == reference_health
        assert dialog.draft.result["report"] == reference_report
        assert dialog.last_result["report"]["valid_samples"] == expected["outputs"]["count"]
        save_image(evidence / "preview_overlay.png", dialog.last_result["overlay"])
        save_image(evidence / "preview_profile.png", dialog.last_result["chart"])
        if metric:
            dialog.open_cloud()
            native = dialog.children[-1]
            assert native.cloud is not None
            native.flyaround.setChecked(True)
            yaw = native.yaw.value()
            native.advance_orbit()
            assert native.yaw.value() != yaw
            native.flyaround.setChecked(False)
            write_cloud(evidence / "preview_profile.ply", native.cloud, binary=True)
        else:
            assert dialog.last_result["report"]["valid_metric_samples"] == 0
        record.update(
            setup=dialog.setup.text(),
            reference_health=reference_health,
            last_preview_health=dialog.health.text(),
            guidance=dialog.guidance.text(),
            reference_preview_separate=True,
            original_settings_exact=True,
            original_reference_and_mask_exact=True,
            calibration_review_readonly=True,
            calibration_fit_cancel_preserves_draft=True,
            native_profile_cloud=metric,
        )
        assert block.parameters == before
        dialog.stage()
        assert block.parameters == before
        inspector.apply()
        assert not warnings, warnings
        staged = folder / f"{case}_staged.json"
        graph.save(staged)
        inspector.open_plugin_dialog()
        app.processEvents()
        reopened = inspector._plugin_dialog
        finish(app, reopened)
        assert reopened.stage_button.isEnabled()
        assert bool(reopened.draft.camera) == metric
        assert reopened.draft.candidate.settings == original.settings
        if metric:
            assert reopened.draft.metadata["calibration_evidence"]["calibration_fit"] == fit
        reopened.reject()
        applied = copy.deepcopy(block.parameters)
        graph.run()
        assert block.last_result is not None, block.error
        roundoff = agreement(expected, stable_result(block.last_result))
        write_json(evidence / "runtime_report.json", block.last_result.outputs["report"])
        inspector.open_plugin_review_dialog()
        app.processEvents()
        completed = inspector._plugin_review_dialog
        assert completed.snapshot is not None
        inspector.close_plugin_review_dialog()
        target = root / "relocated" / case / "project.json"
        target.parent.mkdir(parents=True)
        relative = block.parameters["recipe_path"]
        raw = (folder / relative).read_bytes()
        with ProjectSession.open(staged, registry) as session:
            assert session.inspect()["validation"]["valid"]
            session.save(target)
        with ProjectSession.open(target, registry) as copied:
            assert (target.parent / relative).read_bytes() == raw
            saved = load_recipe(target.parent / relative)
            np.testing.assert_array_equal(saved.reference, original.reference)
            copied.run()
            assert stable_result(copied.graph.blocks["laser"].last_result) == stable_result(
                block.last_result
            )
        assert (folder / before["recipe_path"]).read_bytes() == original_bytes
        assert block.parameters == applied and not warnings, warnings
        record.update(
            stage_preserves_active_until_apply=True,
            cancel_preserves_active=True,
            exact_recipe_save_as=True,
            original_recipe_preserved=True,
            maximum_staged_numeric_difference=roundoff,
            runtime_count=block.last_result.outputs["count"],
            runtime_passed=block.last_result.passed,
            metric=metric,
        )
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
    return record


def verify(root, output, cases=CASES):
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(root / "../../training", output / "projects")
    registry = make_registry(plugins=[str(ROOT / "plugin.py")])
    app = QApplication.instance() or QApplication([])
    records = [verify_case(output, case, registry, app) for case in cases]
    write_json(output / "validation_training.json", records)
    print(
        json.dumps(
            dict(
                native_workflows=len(records),
                counts=[r["runtime_count"] for r in records],
                passed=[r["runtime_passed"] for r in records],
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(ROOT, args.output.resolve())
