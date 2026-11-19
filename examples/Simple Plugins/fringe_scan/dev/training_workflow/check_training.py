"""Verify the actual inspector with a pinned FringeScan plugin and existing captures."""

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
from opencv_dataflow.payloads import NO_VALUE
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

CASES = ("analytic_clean", "connected_missing", "warsaw_crown_phase")


def finish(app, dialog):
    deadline = time.monotonic() + 60
    while dialog.pending:
        app.processEvents()
        dialog.poll()
        assert time.monotonic() < deadline, dialog.status.text()
        time.sleep(0.003)


def check_case(app, registry, work, case, expected, arrays, baseline):
    source = work / "projects" / (case + ".json")
    graph = Graph.load(source, registry)
    # Loading the graph selects the host's pinned package when no loader was specified.
    from opencv_simple_fringe_scan.execution import calibrated_record
    from opencv_simple_fringe_scan.recipe import load_recipe

    window = MainWindow()
    window.registry = registry
    warnings = []
    window.warn = warnings.append
    window.replace_graph(graph, source)
    window.show()
    block = graph.blocks["fringe"]
    before = copy.deepcopy(block.parameters)
    old_path = source.parent / before["recipe_path"]
    old_bytes = old_path.read_bytes()
    inspector = window.inspector
    inspector.select(block)
    stable_result, array_hashes = baseline["stable_result"], baseline["array_hashes"]
    try:
        graph.run()
        assert block.last_result is not None, block.error
        assert stable_result(block.last_result) == expected
        assert array_hashes(block.last_result) == arrays

        inspector.open_plugin_dialog()
        app.processEvents()
        dialog = inspector._plugin_dialog
        finish(app, dialog)
        assert dialog.stage_button.isEnabled(), dialog.status.text()
        dialog.fields["exposure"].setText("0")
        assert "positive exposure" in dialog.setup_status.text()
        assert not dialog.preview_button.isEnabled() and not dialog.stage_button.isEnabled()
        dialog.reject()
        assert block.parameters == before

        inspector.open_plugin_dialog()
        app.processEvents()
        dialog = inspector._plugin_dialog
        finish(app, dialog)
        reference = dialog.reference
        definition = reference.definition
        assert dialog.sequence_table.rowCount() == len(definition["frames"])
        assert dialog.sequence_table.item(0, 0).text() == definition["frames"][0]["id"]
        assert calibrated_record(
            dialog.candidate.definition.get("calibration")
        ) == calibrated_record(definition.get("calibration"))
        with patch("opencv_simple_fringe_scan.qt_training.execute", side_effect=AssertionError):
            dialog.start_audit()
            finish(app, dialog)
            assert dialog.audit_result is not None, dialog.status.text()
            dialog.sequence_table.setCurrentCell(len(definition["frames"]) - 1, 0)
            assert dialog.source_frame.currentIndex() == len(definition["frames"]) - 1
            dialog.source_flags.setChecked(True)
            assert not dialog.editor.image_item.pixmap().isNull()
        height, width = reference.reference.shape[1:]
        dialog.set_region((0, 0, width - 1, height))
        assert dialog.audit_result is dialog.candidate is dialog.last_result is None
        dialog.set_mask(reference.mask)
        dialog.preview_reference()
        finish(app, dialog)
        assert dialog.stage_button.isEnabled(), dialog.status.text()
        np.testing.assert_array_equal(dialog.candidate.reference, reference.reference)
        np.testing.assert_array_equal(dialog.candidate.mask, reference.mask)
        health = dialog.reference_health.text()
        report = copy.deepcopy(dialog.reference_result["report"])
        if before["source"] == "connected capture":
            dialog.preview_connected()
        else:
            # Exercise the file-preview path without a desktop file chooser.
            with patch.object(
                dialog, "choose", return_value=source.parent / before["capture_manifest"]
            ):
                dialog.preview_file()
        finish(app, dialog)
        assert dialog.reference_health.text() == health
        assert dialog.last_result["report"]["valid_pixels"] == expected["outputs"]["count"]
        assert dialog.reference_result["report"] == report

        evidence = work / "evidence" / case
        evidence.mkdir(parents=True)
        for name in ("overview", "decode_quality", "quality"):
            if name in dialog.last_result["views"]:
                rgb = dialog.last_result["views"][name]
                okay, png = cv2.imencode(".png", rgb[:, :, ::-1])
                assert okay
                (evidence / (name + ".png")).write_bytes(png.tobytes())
        metric = block.last_result.outputs["cloud"] is not NO_VALUE
        assert dialog.cloud_button.isEnabled() == metric
        if metric:
            dialog.open_cloud()
            viewer = dialog.cloud_windows[-1]
            assert viewer.flyaround.isChecked()
            yaw = viewer.yaw.value()
            viewer.advance_orbit()
            assert viewer.yaw.value() != yaw
        dialog.stage()
        assert block.parameters == before and old_path.read_bytes() == old_bytes
        inspector.apply()
        assert not warnings, warnings
        staged = source.with_name(case + "_trained.json")
        graph.save(staged)

        inspector.open_plugin_dialog()
        app.processEvents()
        reopened = inspector._plugin_dialog
        finish(app, reopened)
        assert reopened.stage_button.isEnabled(), reopened.status.text()
        assert reopened.reference.definition["reference_fit"] == json.loads(json.dumps(report))
        np.testing.assert_array_equal(reopened.reference.reference, reference.reference)
        np.testing.assert_array_equal(reopened.reference.mask, reference.mask)
        reopened.reject()

        graph.run()
        assert block.last_result is not None, block.error
        assert stable_result(block.last_result) == expected
        assert array_hashes(block.last_result) == arrays
        baseline["native_review"](app, window, block)
        copied = work / "relocated" / case / "project.json"
        copied.parent.mkdir(parents=True)
        with ProjectSession.open(staged, registry) as session:
            session.save(copied)
        relative = block.parameters["recipe_path"]
        assert (copied.parent / relative).read_bytes() == (source.parent / relative).read_bytes()
        saved = load_recipe(copied.parent / relative, sha256=block.parameters["sha256"])
        assert saved.definition["frames"] == definition["frames"]
        np.testing.assert_array_equal(saved.reference, reference.reference)
        with ProjectSession.open(copied, registry) as session:
            session.run()
            result = session.graph.blocks["fringe"].last_result
            assert result is not None and stable_result(result) == expected
            assert array_hashes(result) == arrays
        return {
            "case": case,
            "valid_pixels": expected["outputs"]["count"],
            "passed": expected["passed"],
            "metric_cloud": metric,
            "native_flyaround": metric,
            "stage_apply_cancel_and_exact_recipe_save_as": True,
            "report_maps_graphics_and_cloud_exact": True,
        }
    finally:
        window.dirty = False
        window.close()
        window.deleteLater()
        app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Original completed bundle")
    parser.add_argument("--projects", type=Path, help="Defaults to BASELINE/projects")
    parser.add_argument("--plugin", type=Path, help="Optional trusted dev4 loader")
    parser.add_argument("--output", type=Path, required=True, help="New verification directory")
    parser.add_argument(
        "--verify-previous", action="store_true", help="Also replay the original unsplit bundle"
    )
    args = parser.parse_args()
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    root, output = args.baseline.resolve(), args.output.resolve()
    if args.verify_previous and not (root / "projects").is_dir():
        parser.error("--verify-previous requires an original baseline with its projects/ folder")
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
