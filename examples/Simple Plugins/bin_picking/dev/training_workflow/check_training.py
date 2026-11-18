"""Actual native inspector setup/edit/preview/Stage/Apply/reopen/Save As checks."""

# Select the explicit wheel before the host's lazy plugin imports.
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
spec = importlib.util.spec_from_file_location("pinned_bin_training", ROOT / "plugin.py")
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
from PySide6.QtWidgets import QApplication

from opencv_simple_bin_picking.jaw_teaching import reference_capture
from opencv_simple_bin_picking.qt_setup import BinBoundsDialog

CASES = ("suction_good", "suction_real13", "jaw_clear", "jaw_real13")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def stable_result(result):
    return json.loads(json.dumps(dict(
        passed=bool(result.passed), score=result.score, message=result.message,
        fail_reasons=result.fail_reasons, outputs={name: value
            for name, value in result.outputs.items()
            if value is not NO_VALUE and not isinstance(value, PointCloud)}), allow_nan=False))


def staged_decisions(record):
    record = copy.deepcopy(record)
    # Retaining explicit pose/depth reference notes creates a new jaw archive.
    # Preserve/check that checksum separately; compare every decision field.
    record["outputs"]["report"].pop("recipe_sha256", None)
    return record


def decision_agreement(expected, actual):
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        return max((decision_agreement(expected[k], actual[k]) for k in expected), default=0.)
    if isinstance(expected, list):
        assert len(expected) == len(actual)
        return max((decision_agreement(a, b) for a, b in zip(expected, actual)), default=0.)
    if isinstance(expected, float) and isinstance(actual, float):
        error = abs(expected - actual)
        assert error <= 1e-12, (expected, actual)
        return error
    assert expected == actual, (expected, actual)
    return 0.


def graphics_hashes(result):
    return {name: {field: hashlib.sha256(getattr(cloud, field).tobytes()).hexdigest()
                   for field in ("points", "colors")}
            for name, cloud in result.outputs.items() if isinstance(cloud, PointCloud)}


def finish(app, dialog):
    deadline = time.monotonic() + 90
    while dialog._future is not None:
        app.processEvents()
        dialog.poll()
        assert time.monotonic() < deadline, dialog.status.text()
        time.sleep(.002)


def ready_reference(dialog, jaw, app):
    if jaw:
        # Explicit graph input capture, including declared poses. No hidden pose inference.
        if reference_capture(dialog.session.recipe) is None:
            dialog.capture_reference()
        dialog.preview_reference()
    finish(app, dialog)
    assert (dialog.session.ready if jaw else dialog.session.trained), dialog.status.text()


def verify_case(root, case, registry, app):
    folder = root / "projects"
    project = folder / f"{case}.json"
    jaw = case.startswith("jaw_")
    graph = Graph.load(project, registry)
    window = MainWindow()
    window.registry = registry
    warnings = []
    window.warn = warnings.append
    window.replace_graph(graph, project)
    window.show()
    block = next(b for b in graph.blocks.values()
                 if b.type_id == ("3d.simple_bin_picking_jaw" if jaw else "3d.simple_bin_picking"))
    before = copy.deepcopy(block.parameters)
    original_bytes = (folder / before["recipe_path"]).read_bytes()
    expected = json.loads((ROOT / "expected.json").read_bytes())[case]
    inspector = window.inspector
    inspector.select(block)
    record = dict(case=case, desktop_input_used=False, window_capture_used=False,
                  robot_execution=False)
    evidence = root / "evidence" / case
    evidence.mkdir(parents=True)
    try:
        graph.run()
        assert block.last_result is not None, block.error
        assert stable_result(block.last_result) == expected
        original_graphics = graphics_hashes(block.last_result)
        inspector.open_plugin_dialog()
        app.processEvents()
        dialog = inspector._plugin_dialog
        assert dialog is not None
        original = dialog.session.recipe
        base = original.bin if jaw else original
        assert base.frame in dialog.setup.text() and base.camera.frame in dialog.setup.text()
        assert f"{base.units}/stored sample" in dialog.setup.text()
        ready_reference(dialog, jaw, app)
        assert dialog.buttons["Stage recipe"].isEnabled()
        cached = dialog.session.result if jaw else dialog.result
        field = "minimum_opening" if jaw else "cup_radius"
        old_value = dialog.fields[field].value()
        invalid = (original.settings.maximum_opening * 1000 if jaw
                   else original.settings.body_radius + 1)
        dialog.fields[field].setValue(invalid)
        assert "Invalid draft" in dialog.guidance.text()
        assert "stale" in dialog.guidance.text()
        assert not dialog.buttons["Preview reference"].isEnabled()
        assert not dialog.buttons["Stage recipe"].isEnabled()
        assert (dialog.session.result if jaw else dialog.result) is cached
        record["invalid_settings_reason"] = dialog.guidance.text()
        dialog.fields[field].setValue(old_value)
        dialog.reject()
        assert block.parameters == before
        # Reopen the actual applied recipe before reversible geometric edits.
        inspector.open_plugin_dialog()
        app.processEvents()
        dialog = inspector._plugin_dialog
        if jaw:
            if reference_capture(dialog.session.recipe) is None:
                dialog.capture_reference()
            retained = dialog.session.recipe
            dialog.fields["opening_margin"].setValue(retained.settings.opening_margin * 1000 + .1)
            dialog.sync()
            assert dialog.session.recipe.settings != retained.settings
            dialog.undo()
            assert dialog.session.recipe.settings == retained.settings
            assert dialog.session.recipe.grasps == retained.grasps
            np.testing.assert_array_equal(dialog.session.recipe.grasp_from_tcp,
                                          retained.grasp_from_tcp)
            dialog.preview_reference()
        else:
            finish(app, dialog)
            retained = dialog.session.recipe
            def edit(editor):
                editor.minimum[0].setText(repr(retained.minimum[0] +
                    (retained.maximum[0] - retained.minimum[0]) * .02))
                editor._accept()
                return editor.result()
            with patch.object(BinBoundsDialog, "exec", edit):
                dialog.edit_bounds()
            assert dialog.session.recipe.minimum != retained.minimum
            assert not dialog.session.trained
            dialog.undo()
            assert dialog.session.recipe.minimum == retained.minimum
            dialog.preview_reference()
        finish(app, dialog)
        assert dialog.buttons["Stage recipe"].isEnabled(), dialog.status.text()
        trained = dialog.session.recipe
        trained_bin = trained.bin if jaw else trained
        assert trained.settings == original.settings
        assert trained_bin.minimum == base.minimum and trained_bin.maximum == base.maximum
        np.testing.assert_array_equal(trained_bin.bin_to_camera, base.bin_to_camera)
        if jaw:
            np.testing.assert_array_equal(trained.part.points, original.part.points)
            np.testing.assert_array_equal(trained.part.normals, original.part.normals)
            assert trained.grasps == original.grasps
            assert reference_capture(trained) is not None
        else:
            np.testing.assert_array_equal(trained.reference_depth, original.reference_depth)
        assert dialog.viewer.cloud is not None
        with (patch("opencv_simple_bin_picking.planning.Picker.plan",
                    side_effect=AssertionError("Review planned suction")),
              patch("opencv_simple_bin_picking.jaw.JawPlanner.plan",
                    side_effect=AssertionError("Review planned jaw"))):
            layer = next(iter(dialog.layers.values()))
            layer.setChecked(False)
            layer.setChecked(True)
            dialog.viewer.flyaround.setChecked(True)
            yaw = dialog.viewer.yaw.value()
            dialog.viewer.advance_orbit()
            assert dialog.viewer.yaw.value() != yaw
            dialog.viewer.flyaround.setChecked(False)
        write_cloud(evidence / "training_preview.ply", dialog.viewer.cloud, binary=True)
        assert cv2.imwrite(str(evidence / "training_preview.png"), render_cloud(
            dialog.viewer.cloud, 960, 640, yaw=155, pitch=-20, stable_scale=True))
        record.update(setup_summary=dialog.setup.text(), ready_guidance=dialog.guidance.text(),
                      preview_health=dialog.health.text(), edit_undo_restores_definitions=True,
                      settings_exact=True, calibration_exact=True, native_training_flyaround=True)
        assert block.parameters == before
        dialog.stage()
        assert block.parameters == before
        inspector.apply()
        assert not warnings, warnings
        assert block.parameters["recipe_path"] != before["recipe_path"]
        staged = folder / f"{case}_staged.json"
        graph.save(staged)
        inspector.open_plugin_dialog()
        app.processEvents()
        reopened = inspector._plugin_dialog
        finish(app, reopened)
        assert reopened.session.recipe.settings == original.settings
        if jaw:
            assert reference_capture(reopened.session.recipe) is not None
        reopened.reject()
        applied = copy.deepcopy(block.parameters)
        graph.run()
        assert block.last_result is not None, block.error
        actual = stable_result(block.last_result)
        if "recipe_sha256" in actual["outputs"]["report"]:
            assert actual["outputs"]["report"]["recipe_sha256"] == block.parameters["sha256"]
        roundoff = decision_agreement(staged_decisions(expected), staged_decisions(actual))
        assert graphics_hashes(block.last_result) == original_graphics
        inspector.open_plugin_review_dialog()
        app.processEvents()
        review = inspector._plugin_review_dialog
        assert review.snapshot is not None and review.cloud is not None
        write_cloud(evidence / "runtime_review.ply", review.cloud, binary=True)
        write_json(evidence / "runtime_report.json", block.last_result.outputs["report"])
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
            copied.run()
            assert stable_result(next(b for b in copied.graph.blocks.values()
                if b.type_id == block.type_id).last_result) == actual
        assert (folder / before["recipe_path"]).read_bytes() == original_bytes
        assert block.parameters == applied
        record.update(original_recipe_preserved=True, cancel_preserves_active=True,
                      stage_preserves_active_until_apply=True, exact_recipe_save_as=True,
                      retained_reference_reopens=True, original_recipe_runtime_exact=True,
                      staged_decisions_agree_within_1e_12=True,
                      maximum_staged_numeric_difference=roundoff,
                      runtime_clouds_exact=True, eligible_picks=block.last_result.outputs["count"])
        assert not warnings, warnings
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
    records = [verify_case(output, name, registry, app) for name in cases]
    write_json(output / "validation_training.json", records)
    print(json.dumps(dict(native_workflows=len(records),
                          eligible_picks=[r["eligible_picks"] for r in records])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(ROOT, args.output.resolve())
