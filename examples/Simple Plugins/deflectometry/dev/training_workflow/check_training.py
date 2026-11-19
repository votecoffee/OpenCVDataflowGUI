"""Replay Deflectometry teaching through the actual offscreen inspector."""

import argparse
import copy
import json
import os
import runpy
import shutil
import time
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from opencv_dataflow.app import MainWindow
from opencv_dataflow.assistant_api import ProjectSession, make_registry
from opencv_dataflow.core import Graph
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

CASES = ("analytic_clean", "connected_change", "sofast_self_review")


def finish(app, dialog):
    deadline = time.monotonic() + 60
    while dialog.pending:
        app.processEvents()
        dialog.poll()
        assert time.monotonic() < deadline, dialog.status.text()
        time.sleep(0.003)


def check_case(app, registry, work, case, expected, arrays, baseline):
    project = work / "projects" / (case + ".json")
    graph = Graph.load(project, registry)
    from opencv_simple_deflectometry.recipe import load_recipe

    window = MainWindow()
    window.registry = registry
    warnings = []
    window.warn = warnings.append
    window.replace_graph(graph, project)
    window.show()
    block = graph.blocks["deflect"]
    before = copy.deepcopy(block.parameters)
    old_path = project.parent / before["recipe_path"]
    old_bytes = old_path.read_bytes()
    inspector = window.inspector
    inspector.select(block)

    def assert_result(result):
        assert result is not None, block.error
        assert baseline["stable_result"](result) == expected
        assert baseline["array_hashes"](result) == arrays

    def open_trainer():
        inspector.open_plugin_dialog()
        app.processEvents()
        dialog = inspector._plugin_dialog
        finish(app, dialog)
        assert dialog.stage_button.isEnabled(), dialog.status.text()
        return dialog

    try:
        graph.run()
        assert_result(block.last_result)
        dialog = open_trainer()
        dialog.fields["angles_y"].setText("0, 0, 0")
        assert "Check Y shifts" in dialog.setup_status.text()
        assert not dialog.audit_button.isEnabled() and not dialog.stage_button.isEnabled()
        dialog.reject()
        assert block.parameters == before

        dialog = open_trainer()
        reference = dialog.reference
        capture = reference.reference
        if case == "sofast_self_review":
            assert "exposure unknown" in dialog.setup_status.text()
            assert "Uncalibrated code values" in dialog.setup_status.text()
        with patch("opencv_simple_deflectometry.qt_training.execute", side_effect=AssertionError):
            dialog.start_audit()
            finish(app, dialog)
        audit = dialog.audit_result
        assert audit is not None, dialog.status.text()
        assert audit["active_pixels"] == int(capture.mask.sum())
        assert all(
            0 <= audit["complete_both"] <= audit[key] for key in ("complete_x", "complete_y")
        )
        dialog.frame_list.setCurrentRow(len(audit["rows"]) - 1)
        assert (
            "Y step" in dialog.selected_pattern.text()
            and "nonfinite" in dialog.selected_pattern.text()
        )
        dialog.source_flags.setChecked(True)
        assert not dialog.editor.image_item.pixmap().isNull()
        rows, cols = capture.mask.shape
        dialog.set_region((0, 0, cols - 1, rows))
        assert dialog.audit_result is dialog.candidate is dialog.last_result is None
        dialog.set_mask(capture.mask)
        dialog.preview_reference()
        finish(app, dialog)
        assert dialog.stage_button.isEnabled(), dialog.status.text()
        assert dialog.candidate.reference.sequence == capture.sequence
        health = dialog.reference_health.text()
        reference_report = json.loads(json.dumps(dialog.reference_result["report"]))

        if before["source"] == "connected capture":
            dialog.preview_connected()
        else:
            manifest = project.parent / before["capture_manifest"]
            with patch.object(dialog, "choose", return_value=manifest):
                dialog.preview_file()
        finish(app, dialog)
        assert dialog.reference_health.text() == health
        report = dialog.last_result["report"]
        target_report = expected["outputs"]["report"]
        for key in ("comparable_pixels", "changed_pixels", "passed", "metric_height"):
            assert report[key] == target_report[key]
        assert not report["metric_height"]
        evidence = work / "evidence" / case
        evidence.mkdir(parents=True)
        baseline["write_json"](evidence / "source-audit.json", audit)
        for name in ("overlay", "quality"):
            okay, png = cv2.imencode(".png", dialog.last_result["views"][name][:, :, ::-1])
            assert okay
            (evidence / (name + ".png")).write_bytes(png.tobytes())
        last_row = dialog.frame_list.currentRow()
        dialog.stage()
        assert block.parameters == before and old_path.read_bytes() == old_bytes
        inspector.apply()
        assert not warnings, warnings
        staged = project.with_name(case + "_trained.json")
        graph.save(staged)

        reopened = open_trainer()
        assert reopened.reference.notes["reference_fit"] == reference_report
        assert reopened.frame_list.currentRow() == last_row
        assert reopened.reference.reference.sequence == capture.sequence
        np.testing.assert_array_equal(reopened.reference.reference.mask, capture.mask)
        reopened.reject()
        graph.run()
        assert_result(block.last_result)
        baseline["native_review"](app, window, block)

        copied = work / "relocated" / case / "project.json"
        copied.parent.mkdir(parents=True)
        with ProjectSession.open(staged, registry) as session:
            session.save(copied)
        relative = block.parameters["recipe_path"]
        assert (copied.parent / relative).read_bytes() == (project.parent / relative).read_bytes()
        saved = load_recipe(copied.parent / relative, sha256=block.parameters["sha256"])
        assert saved.reference.sequence == capture.sequence
        np.testing.assert_array_equal(saved.reference.x_images, capture.x_images)
        np.testing.assert_array_equal(saved.reference.y_images, capture.y_images)
        np.testing.assert_array_equal(saved.reference.mask, capture.mask)
        with ProjectSession.open(copied, registry) as session:
            session.run()
            assert_result(session.graph.blocks["deflect"].last_result)
        return {
            "case": case,
            "comparable_pixels": report["comparable_pixels"],
            "changed_pixels": report["changed_pixels"],
            "passed": report["passed"],
            "native_restore_audit_preview_stage_apply_cancel_and_save_as": True,
            "report_and_arrays_exact": True,
        }
    finally:
        window.dirty = False
        window.close()
        window.deleteLater()
        app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--projects", type=Path, help="Defaults to BASELINE/projects")
    parser.add_argument("--plugin", type=Path, help="Optional trusted dev4 loader")
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory")
    parser.add_argument(
        "--verify-previous", action="store_true", help="Replay the unsplit baseline"
    )
    args = parser.parse_args()
    root, output = args.baseline.resolve(), args.output.resolve()
    if args.verify_previous and not (root / "projects").is_dir():
        parser.error("--verify-previous requires an original baseline with its projects/ folder")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(
        args.projects or root / "projects",
        output / "projects",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    plugin = args.plugin.resolve(strict=True) if args.plugin else None
    registry = make_registry(plugins=[str(plugin)] if plugin else [])
    baseline = runpy.run_path(str(root / "examples.py"))
    app = QApplication.instance() or QApplication([])
    expected = json.loads((root / "expected.json").read_bytes())
    arrays = {
        row["case"]: row["arrays"]
        for row in json.loads((root / "verification.json").read_bytes())["cases"]
    }
    result = [
        check_case(app, registry, output, case, expected[case], arrays[case], baseline)
        for case in CASES
    ]
    baseline["write_json"](output / "training-checks.json", result)
    print(json.dumps(result))
    if args.verify_previous:
        from opencv_dataflow.plugins import load_plugin

        baseline["verify"].__globals__["load_plugin"] = (
            (lambda _, reg: load_plugin(plugin, reg)) if plugin else (lambda *_: None)
        )
        baseline["verify"](root, output / "unchanged-runtime", native=True)


if __name__ == "__main__":
    main()
