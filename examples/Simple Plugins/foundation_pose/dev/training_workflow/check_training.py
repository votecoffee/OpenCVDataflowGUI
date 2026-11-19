"""Replay FoundationPose setup with the bundled plugin and isolated ONNX worker."""

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
from opencv_dataflow.app import MainWindow
from opencv_dataflow.assistant_api import ProjectSession, make_registry
from opencv_dataflow.core import Graph
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

CASES = ("held_out_frame13", "wrong_object")


def finish(app, dialog):
    deadline = time.monotonic() + 180
    while dialog.pending:
        app.processEvents()
        dialog.poll()
        assert time.monotonic() < deadline, dialog.status.text()
        time.sleep(0.01)


def check_case(app, registry, examples, output, case):
    root = examples / case
    project = output / case / "run_project.json"
    project.parent.mkdir()
    for asset in ("run_project.json", "trained.sfp", "capture.npz", "coarse.json"):
        shutil.copy2(root / asset, project.parent / asset)
    old_asset = project.with_name("trained.sfp")
    old_bytes = old_asset.read_bytes()
    expected = json.loads((root / "validation.json").read_bytes())
    graph = Graph.load(project, registry)
    block = graph.blocks["pose"]
    assert hashlib.sha256(old_bytes).hexdigest() == expected["recipe_sha256"]
    applied = copy.deepcopy(block.parameters)

    window = MainWindow()
    warnings = []
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
        assert dialog.stage_button.isEnabled(), dialog.status.text()
        assert dialog.filmstrip.count() == 4
        return dialog

    try:
        graph.run()
        assert block.state == "fail", block.error
        runtime = block.last_result.outputs
        assert runtime["accepted_count"] == expected["accepted"]
        assert len(runtime["report"]["candidates"]) == expected["candidates"]

        dialog = open_setup()
        assert "XYZ span" in dialog.identity.text()
        assert dialog.reference_view.cloud.frame == dialog.reference.frame
        reference_mask = dialog.reference.mask.copy()
        dialog.filmstrip.setCurrentRow(3)
        assert not dialog.filmstrip.currentItem().icon().isNull()
        dialog.editor.replace_mask(np.zeros_like(reference_mask))
        assert not dialog.stage_button.isEnabled()
        dialog.editor.undo()
        assert dialog.stage_button.isEnabled()
        dialog.fields["depth_multiplier"].setText("0.01")
        assert "at least 12 depths" in dialog.health.text()
        assert not dialog.stage_button.isEnabled()
        dialog.reject()
        assert block.parameters == applied and old_asset.read_bytes() == old_bytes

        dialog = open_setup()
        reference_status = dialog.health.text()
        assert dialog.backend_button.isEnabled(), dialog.backend_status.text()
        dialog.backend_button.click()
        finish(app, dialog)
        assert dialog.checked_backend is not None, dialog.status.text()
        assert dialog.last_result is None
        assert "both hashes matched" in dialog.backend_status.text()
        if case == "held_out_frame13":
            dialog.preview_wire_button.click()
            preview_source = "Connected:"
        else:
            with patch.object(dialog, "choose", return_value=project.with_name("coarse.json")):
                dialog.import_coarse()
            with patch.object(dialog, "choose", return_value=project.with_name("capture.npz")):
                dialog.preview_file_button.click()
            preview_source = "Independent:"
        finish(app, dialog)
        assert dialog.last_result is not None, dialog.status.text()
        meta, arrays = dialog.last_result
        report = meta["report"]
        assert dialog.status.text().startswith(preview_source)
        assert dialog.health.text() == reference_status
        assert report["worker"]["registration_requests"] == 1
        assert report["execution"]["initialization"] == expected["initialization"]
        assert report["candidates"] == runtime["report"]["candidates"]
        assert report["accepted_candidates"] == expected["accepted"]
        np.testing.assert_array_equal(arrays["overlay_rgb"], runtime["overlay"])
        np.testing.assert_array_equal(arrays["review_xyz"], runtime["cloud"].points)
        np.testing.assert_array_equal(arrays["review_rgb"], runtime["cloud"].colors)
        assert dialog.result_view.flyaround.isChecked()
        assert dialog.result_view.cloud.frame == report["target_frame"]

        selected = int(np.count_nonzero(reference_mask))
        measured = int(dialog.target_measured)
        dialog.fields["minimum_coverage"].setText("0.701")
        assert dialog.last_result is None
        dialog.stage_button.click()
        assert inspector._plugin_dialog is None and not warnings, warnings
        staged = inspector.read_parameters(validate=False)
        assert staged["recipe_path"] != applied["recipe_path"]
        assert block.parameters == applied and old_asset.read_bytes() == old_bytes
        assert inspector.apply()
        reopened = open_setup()
        assert reopened.settings().minimum_coverage == 0.701
        np.testing.assert_array_equal(reopened.reference.mask, reference_mask)
        assert reopened.health.text() == reference_status
        reopened.reject()
        graph.save(project)

        destination = output / "relocated" / case / "project.json"
        destination.parent.mkdir(parents=True)
        with ProjectSession.open(project, registry) as session:
            session.save(destination)
        with ProjectSession.open(destination, registry) as session:
            relative = session.graph.blocks["pose"].parameters["recipe_path"]
            retained = (destination.parent / relative).read_bytes()
            assert retained == (project.parent / staged["recipe_path"]).read_bytes()
            assert hashlib.sha256(retained).hexdigest() == staged["sha256"]
        return dict(
            case=case,
            preview=preview_source.rstrip(":").lower(),
            selected_reference_pixels=selected,
            measured_reference_pixels=measured,
            candidates=len(report["candidates"]),
            accepted=report["accepted_candidates"],
            registrations_after_check_and_first_preview=report["worker"]["registration_requests"],
            exact_graph_candidates_overlay_and_cloud=True,
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
    parser.add_argument("examples", type=Path, help="Host foundation_pose examples")
    parser.add_argument("output", type=Path, help="Fresh scratch directory")
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
