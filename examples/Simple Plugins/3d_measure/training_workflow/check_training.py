"""Native retained-setup/edit/train/Stage/Apply/reopen/Save As qualification."""

# Select the explicit wheel before importing host lazy plugin providers.
# ruff: noqa: E402, I001

import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pinned_measurement_training", ROOT / "plugin.py")
plugin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin)

import cv2
import numpy as np
from opencv_dataflow.app import MainWindow
from opencv_dataflow.assistant_api import ProjectSession, make_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.pointclouds import PointCloud, render_cloud
from PySide6.QtWidgets import QApplication, QDialog

from opencv_simple_3d_measure import load_recipe
from opencv_simple_3d_measure.qt_bounds import RegionBoundsDialog

CASES = ("real13", "real24", "missing")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def stable_result(result):
    return json.loads(json.dumps(dict(passed=result.passed, outputs={
        name: value for name, value in result.outputs.items()
        if name in ("measurements", "planes", "lines", "report", "value", "count",
                    "residual_report") and value is not NO_VALUE}), allow_nan=False))


def finish(app, dialog):
    deadline = time.monotonic() + 60
    while dialog._future is not None:
        app.processEvents()
        dialog.poll_training()
        assert time.monotonic() < deadline, dialog.status.text()
        time.sleep(.002)


def verify_case(root, case, registry, app):
    folder = root / "projects"
    project = folder / f"{case}.json"
    graph = Graph.load(project, registry)
    window = MainWindow()
    window.registry = registry
    warnings = []
    window.warn = warnings.append
    window.replace_graph(graph, project)
    window.show()
    block = graph.blocks["measure"]
    before = copy.deepcopy(block.parameters)
    original_bytes = (folder / before["recipe_path"]).read_bytes()
    expected = json.loads((ROOT / "expected.json").read_bytes())[case]
    inspector = window.inspector
    inspector.select(block)
    record = {"case": case, "desktop_input_used": False, "window_capture_used": False}
    artifacts = root / "evidence" / case
    artifacts.mkdir(parents=True)
    try:
        graph.run()
        assert block.last_result is not None, block.error
        assert stable_result(block.last_result) == expected
        inspector.open_plugin_dialog()
        app.processEvents()
        dialog = inspector._plugin_dialog
        assert dialog is not None and dialog.session.trained
        original = dialog.session.recipe
        assert len(dialog.editor.source.cloud.points) == len(original.reference)
        np.testing.assert_array_equal(dialog.session.reference, original.reference)
        np.testing.assert_array_equal(dialog.editor.source.cloud.points,
                                      original.reference.astype(np.float32))
        assert dialog.session.units == original.units and dialog.session.frame == original.frame
        assert dialog.session.regions == original.regions
        assert dialog.session.measurements == original.measurements
        assert dialog.session.settings == original.settings
        assert len(dialog.editor.boxes) == len(original.regions)
        # Definition selection links to the datum without mutating the model.
        dialog.table.selectRow(0)
        app.processEvents()
        first = original.measurements[0]
        assert dialog._current_plane().name == first.plane
        assert dialog.table.item(0, 4).text() == first.plane
        record["definition_summary"] = dialog.measurement_summary.text()
        region = dialog._current_plane()
        # Reject/invalid input never changes a region. Accept is one reversible edit.
        with patch.object(RegionBoundsDialog, "exec", return_value=QDialog.Rejected):
            dialog._edit_bounds()
        assert dialog.session.trained
        def edit(editor):
            editor.minimum[0].setText(repr(region.minimum[0] +
                                         (region.maximum[0] - region.minimum[0]) * .05))
            editor._accept()
            return editor.result()
        with patch.object(RegionBoundsDialog, "exec", edit):
            dialog._edit_bounds()
        assert not dialog.session.trained and not dialog.buttons["Stage recipe"].isEnabled()
        dialog._history(False)
        assert dialog.session.regions == original.regions and not dialog.session.trained
        dialog._history(True)
        assert dialog.session.regions != original.regions
        dialog._history(False)
        assert dialog.session.regions == original.regions
        dialog.start_training()
        finish(app, dialog)
        assert dialog.session.trained, dialog.status.text()
        trained = dialog.session.recipe
        np.testing.assert_array_equal(trained.reference, original.reference)
        assert trained.measurements == original.measurements
        assert trained.settings == original.settings
        for a, b in zip(trained.regions, original.regions):
            assert (a.name, a.minimum, a.maximum, a.tolerance) == (
                b.name, b.minimum, b.maximum, b.tolerance)
        reference = PointCloud(dialog.session.reference, original.units, original.frame,
                               colors=dialog.editor._colors)
        write_cloud(artifacts / "reference_fit.ply", reference, binary=True)
        record.update(reference_rows=len(original.reference), units=original.units,
                      measurement_frame=original.frame, setup_guidance=dialog.guidance.text(),
                      numeric_edit_undo_redo=True, restored_settings_and_regions=True,
                      reference_xyz_exact=True)
        # Stage remains separate from Apply. Cancel from a freshly reopened popup
        # leaves the active recipe and numerical execution unchanged.
        assert block.parameters == before
        dialog._stage()
        assert block.parameters == before
        inspector.apply()
        assert not warnings, warnings
        assert block.parameters["recipe_path"] != before["recipe_path"]
        staged = folder / f"{case}_staged.json"
        graph.save(staged)
        inspector.open_plugin_dialog()
        reopened = inspector._plugin_dialog
        assert reopened.session.trained
        np.testing.assert_array_equal(reopened.session.reference, original.reference)
        assert reopened.session.measurements == original.measurements
        reopened._flip()
        assert not reopened.session.trained
        applied = copy.deepcopy(block.parameters)
        reopened.reject()
        assert block.parameters == applied
        graph.run()
        assert block.last_result is not None, block.error
        # Fixed recipe unchanged execution is exact above. Retraining may change
        # floating roundoff in fitted axes; do not compare artifact identity.
        for old, new in zip(expected["outputs"]["measurements"],
                            block.last_result.outputs["measurements"]):
            assert (old["name"], old["status"], old["units"], old["reason"]) == (
                new["name"], new["status"], new["units"], new["reason"])
            if old["value"] is None:
                assert new["value"] is None
            else:
                assert abs(new["value"] - old["value"]) < 1e-9
        assert block.last_result.passed == expected["passed"]
        inspector.open_plugin_review_dialog()
        app.processEvents()
        review = inspector._plugin_review_dialog
        assert review.snapshot is not None and review.cloud is not None
        review.flyaround.setChecked(True)
        yaw = review.yaw.value()
        review.advance_orbit()
        assert review.yaw.value() != yaw
        review.flyaround.setChecked(False)
        write_cloud(artifacts / "runtime_fit.ply", review.cloud, binary=True)
        assert cv2.imwrite(str(artifacts / "runtime_fit.png"), render_cloud(
            review.cloud, 960, 640, yaw=25, pitch=-20, stable_scale=True))
        measurement_values = block.last_result.outputs["measurements"]
        write_json(artifacts / "measurements.json", measurement_values)
        inspector.close_plugin_review_dialog()
        target = root / "relocated" / case / "project.json"
        target.parent.mkdir(parents=True)
        relative = block.parameters["recipe_path"]
        staged_bytes = (folder / relative).read_bytes()
        with ProjectSession.open(staged, registry) as session:
            assert session.inspect()["validation"]["valid"]
            session.save(target)
        with ProjectSession.open(target, registry) as session:
            assert session.inspect()["validation"]["valid"]
            assert (target.parent / relative).read_bytes() == staged_bytes
            saved = load_recipe(target.parent / relative)
            np.testing.assert_array_equal(saved.reference, original.reference)
            assert saved.measurements == original.measurements
            session.run()
            assert stable_result(session.graph.blocks["measure"].last_result) == stable_result(
                block.last_result)
        assert (folder / before["recipe_path"]).read_bytes() == original_bytes
        record.update(stage_preserves_active_until_apply=True, cancel_preserves_active=True,
                      exact_recipe_save_as=True, independent_native_flyaround=True,
                      unchanged_recipe_runtime_exact=True, original_recipe_preserved=True,
                      retrained_values_match_within_1e_9_declared_units=True,
                      passed=block.last_result.passed,
                      unavailable=sum(m["value"] is None for m in measurement_values),
                      measurements=measurement_values)
        assert not warnings, warnings
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
    return record


def verify(root, output, cases=CASES):
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(root / "projects", output / "projects")
    registry = make_registry(plugins=[str(ROOT / "plugin.py")])
    app = QApplication.instance() or QApplication([])
    records = [verify_case(output, case, registry, app) for case in cases]
    write_json(output / "validation_training.json", records)
    print(json.dumps({"native_workflows": len(records), "passed": [r["passed"] for r in records],
                      "unavailable": [r["unavailable"] for r in records]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(ROOT, args.output.resolve())
