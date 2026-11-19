"""Check GDRNPP Setup against two retained T-LESS projects in the native inspector."""

import argparse
import copy
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

from opencv_dataflow.app import MainWindow
from opencv_dataflow.assistant_api import ProjectSession, make_registry
from opencv_dataflow.cloud_viewer import CloudViewer
from opencv_dataflow.core import Graph

CASES = ("accepted_parts", "rejected_part")


def finish(app, dialog):
    deadline = time.monotonic() + 180
    while dialog.pending:
        app.processEvents()
        dialog._poll()
        assert time.monotonic() < deadline, dialog.status.text()
        time.sleep(0.01)


def check_case(app, registry, examples, output, case):
    source = examples / case
    project = output / case / "run_project.json"
    project.parent.mkdir()
    prior = json.loads((source / "validation.json").read_bytes())
    for name in (project.name, "input.gdrnpp"):
        shutil.copy2(source / name, project.parent / name)
    old_path = project.parent / "input.gdrnpp"
    old_bytes = old_path.read_bytes()
    graph = Graph.load(project, registry)
    block = graph.blocks["locate"]
    applied = copy.deepcopy(block.parameters)
    warnings = []
    window = MainWindow()
    window.warn = warnings.append
    window.registry = registry
    window.replace_graph(graph, project)
    window.show()
    inspector = window.inspector
    inspector.select(block)

    def open_setup():
        inspector.open_plugin_dialog()
        app.processEvents()
        dialog = inspector._plugin_dialog
        assert dialog is not None, warnings
        finish(app, dialog)
        assert dialog.summary is not None, dialog.status.text()
        assert dialog.stage_button.isEnabled()
        assert dialog.boxes.rowCount() == prior["proposals"]
        return dialog

    try:
        dialog = open_setup()
        assert dialog.summary["reference_capture_id"] == prior["capture_id"]
        assert "tless_object6" in dialog.reference_info.text()
        assert "Selected box depth:" in dialog.box_info.text()
        assert dialog.depth_count > 0
        original_bounds = dialog._draft()["detections"]
        original_depth_count = dialog.depth_count
        total_pixels = dialog.depth_pixels.size
        right = dialog.boxes.item(0, 3)
        right.setText("99999")
        assert "Instance row 1" in dialog.box_info.text()
        assert not dialog.stage_button.isEnabled() and not dialog.preview_button.isEnabled()
        dialog.reject()
        assert block.parameters == applied and old_path.read_bytes() == old_bytes

        dialog = open_setup()
        preflight = {}
        checked = dialog._checkpoint_checked

        def checked_and_recorded(metadata, arrays):
            checked(metadata, arrays)
            preflight.update(metadata["prepared"])

        assert dialog.checkpoint_button.isEnabled(), dialog.checkpoint_status.text()
        with patch.object(dialog, "_checkpoint_checked", checked_and_recorded):
            dialog.checkpoint_button.click()
            finish(app, dialog)
        assert dialog.checkpoint_prepared, dialog.status.text()
        assert not preflight["inspection_performed"]
        assert preflight["worker"]["inference_request_count"] == 0
        assert preflight["recipe_archive_sha256"] == prior["archive_sha256"]
        assert dialog.preview_report is None
        dialog.preview_button.click()
        finish(app, dialog)
        report = dialog.preview_report
        assert report is not None, dialog.status.text()
        assert report["worker"]["model_cache_hit"]
        assert report["worker"]["inference_request_count"] == 1
        assert len(report["candidates"]) == prior["proposals"]
        assert report["accepted_candidates"] == prior["accepted"]
        assert dialog.candidates.currentIndex() + 1 == prior["selected_proposal"]
        assert dialog.preview_arrays["overlay"].shape == dialog.reference["rgb"].shape
        dialog.result_cloud_button.click()
        clouds = dialog.findChildren(CloudViewer)
        assert len(clouds) == 1 and clouds[0].flyaround.isChecked()
        cloud = clouds[0].cloud
        assert cloud.frame == dialog.summary["camera_frame"] and cloud.units == "m"
        np.testing.assert_array_equal(cloud.colors, dialog.preview_arrays["colors"])
        point_count = int(len(cloud.points))
        clouds[0].close()

        independent = None
        if case == "rejected_part":
            request = examples / "live_review/capture.json"
            capture_id = json.loads(request.read_bytes())["capture_id"]
            assert capture_id != prior["capture_id"]
            reference_status = dialog.reference_info.text()
            with patch(
                "opencv_simple_gdrnpp.qt_setup.QFileDialog.getOpenFileName",
                return_value=(str(request), ""),
            ):
                dialog.test_button.click()
            finish(app, dialog)
            assert dialog.result_info.text().startswith("Capture file.")
            assert dialog.reference_info.text() == reference_status
            report = dialog.preview_report
            assert report["worker"]["inference_request_count"] == 2
            assert len(report["candidates"]) == 5
            independent = dict(
                capture_id=capture_id,
                proposals=len(report["candidates"]),
                accepted=report["accepted_candidates"],
            )

        new_name = original_name = dialog.name.text()
        new_name += " workflow check"
        dialog.name.setText(new_name)
        assert dialog.preview_report is None and not dialog.checkpoint_prepared
        dialog.stage_button.click()
        finish(app, dialog)
        assert inspector._plugin_dialog is None and not warnings, warnings
        draft = inspector.read_parameters(validate=False)
        assert draft["recipe_path"] != applied["recipe_path"]
        assert block.parameters == applied and old_path.read_bytes() == old_bytes
        inspector.apply()
        assert block.parameters == draft and not warnings, warnings
        reopened = open_setup()
        assert reopened.name.text() == new_name != original_name
        assert reopened._draft()["detections"] == original_bounds
        assert reopened.depth_count == original_depth_count
        reopened.reject()

        graph.save(project)
        relocated = output / "relocated" / case / "project.json"
        relocated.parent.mkdir(parents=True)
        with ProjectSession.open(project, registry) as session:
            session.save(relocated)
        original_asset = project.parent / draft["recipe_path"]
        with ProjectSession.open(relocated, registry) as session:
            relative = session.graph.blocks["locate"].parameters["recipe_path"]
        relocated_bytes = (relocated.parent / relative).read_bytes()
        assert relocated_bytes == original_asset.read_bytes()
        assert hashlib.sha256(relocated_bytes).hexdigest() == draft["sha256"]
        return dict(
            case=case,
            retained_capture_id=prior["capture_id"],
            registered_depth_pixels=original_depth_count,
            reference_pixels=total_pixels,
            proposals=prior["proposals"],
            accepted=prior["accepted"],
            selected_proposal=prior["selected_proposal"],
            flyaround_points=point_count,
            checkpoint_inference_requests=0,
            warm_preview_inference_requests=1,
            independent_capture=independent,
            stage_apply_cancel_and_exact_save_as=True,
        )
    finally:
        window.dirty = False
        window.close()
        window.deleteLater()
        app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("examples", type=Path, help="Path to the host's gdrnpp examples")
    parser.add_argument("output", type=Path, help="New scratch directory for copied projects")
    args = parser.parse_args()
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    registry = make_registry()
    app = QApplication.instance() or QApplication([])
    results = []
    for case in CASES:
        result = check_case(app, registry, args.examples.resolve(strict=True), output, case)
        print(json.dumps(result), flush=True)
        results.append(result)
    (output / "qualification.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
