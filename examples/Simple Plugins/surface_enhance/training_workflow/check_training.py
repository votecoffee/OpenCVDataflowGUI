"""Exercise actual inspector source audit, training, independent preview and persistence.

Uses Qt offscreen without desktop input or window capture. Output must be new.
"""

# Select the pinned wheel before host lazy plugin providers.
# ruff: noqa: E402, I001
import argparse
import copy
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("pinned_surface_training", ROOT / "plugin.py")
plugin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin)

import cv2
import numpy as np
from opencv_dataflow.app import MainWindow
from opencv_dataflow.assistant_api import ProjectSession, make_registry
from opencv_dataflow.core import Graph
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication
from result_checks import array_hashes, native_review, stable_result

from opencv_simple_surface_enhance.execution import capture_from_recipe
from opencv_simple_surface_enhance.recipe import import_capture, load_recipe

CASES = ("analytic_clean", "connected_material", "missing")


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )


def finish(app, dialog):
    deadline = time.monotonic() + 60
    while dialog.pending:
        app.processEvents()
        dialog.poll()
        assert time.monotonic() < deadline, dialog.status.text()
        time.sleep(0.002)


def save_rgb(path, image):
    okay, encoded = cv2.imencode(".png", image[:, :, ::-1])
    assert okay
    path.write_bytes(encoded.tobytes())


def verify_case(output, case, registry, app):
    folder = output / "projects"
    project = folder / f"{case}.json"
    graph = Graph.load(project, registry)
    window = MainWindow()
    window.registry = registry
    warnings = []
    window.warn = warnings.append
    window.replace_graph(graph, project)
    window.show()
    block = graph.blocks["surface"]
    before = copy.deepcopy(block.parameters)
    original_bytes = (folder / before["recipe_path"]).read_bytes()
    expected = json.loads((ROOT / "expected.json").read_bytes())[case]
    arrays = json.loads((ROOT / "expected_arrays.json").read_bytes())[case]
    inspector = window.inspector
    inspector.select(block)
    evidence = output / "evidence" / case
    evidence.mkdir(parents=True)
    record = dict(case=case, desktop_input_used=False, window_capture_used=False)
    try:
        graph.run()
        assert block.last_result is not None, block.error
        assert stable_result(block.last_result) == expected
        assert array_hashes(block.last_result) == arrays
        inspector.open_plugin_dialog()
        app.processEvents()
        dialog = inspector._plugin_dialog
        finish(app, dialog)
        original = dialog.reference
        assert dialog.stage_button.isEnabled(), dialog.status.text()
        assert dialog.build_candidate().settings == original.settings
        np.testing.assert_array_equal(
            dialog.build_candidate().rig.directions, original.rig.directions
        )
        np.testing.assert_array_equal(
            dialog.build_candidate().rig.exposure_s, original.rig.exposure_s
        )
        dialog.black.setText("1")
        assert "Invalid draft" in dialog.setup.text()
        assert not any(
            button.isEnabled()
            for button in (dialog.audit_button, dialog.preview_button, dialog.stage_button)
        )
        assert dialog.audit_result is dialog.reference_result is dialog.last_result is None
        dialog.reject()
        assert block.parameters == before
        inspector.open_plugin_dialog()
        app.processEvents()
        dialog = inspector._plugin_dialog
        finish(app, dialog)
        # A source audit uses the preparation contract but never estimates or relights.
        with patch(
            "opencv_simple_surface_enhance.qt_training.execute",
            side_effect=AssertionError("Fit during audit"),
        ):
            dialog.start_audit()
            finish(app, dialog)
            assert dialog.audit_table.rowCount() == len(original.rig.light_ids)
            dialog.audit_table.setCurrentCell(3, 0)
            assert dialog.source_light.currentIndex() == dialog.lights.currentRow() == 3
            dialog.lights.setCurrentCell(2, 4)
            assert dialog.source_light.currentIndex() == dialog.audit_table.currentRow() == 2
            assert dialog.lights.currentColumn() == 4
            dialog.source_flags.setChecked(True)
        write_json(evidence / "source_audit.json", dialog.audit_result)
        # Direct view rendering is a diagnostic image, not a desktop screenshot.
        source = dialog.editor.image_item.pixmap().toImage()
        assert source.save(str(evidence / "source_flags.png"))
        zoom = dialog.editor.transform().m11()
        dialog.editor.zoom(1.25)
        assert dialog.editor.transform().m11() > zoom
        dialog.set_region((0, 0, original.reference.shape[2] - 1, original.reference.shape[1]))
        assert dialog.audit_result is dialog.candidate is dialog.last_result is None
        assert not dialog.stage_button.isEnabled()
        # Explicitly restore the exact mask; no implicit Undo or fitted-input substitution.
        dialog.set_mask(original.mask)
        dialog.preview_reference()
        finish(app, dialog)
        assert dialog.stage_button.isEnabled(), dialog.status.text()
        np.testing.assert_array_equal(dialog.candidate.reference, original.reference)
        np.testing.assert_array_equal(dialog.candidate.mask, original.mask)
        np.testing.assert_array_equal(dialog.candidate.rig.directions, original.rig.directions)
        np.testing.assert_array_equal(dialog.candidate.rig.exposure_s, original.rig.exposure_s)
        assert dialog.candidate.settings == original.settings
        reference_health = dialog.reference_health.text()
        reference_report = copy.deepcopy(dialog.reference_result["report"])
        if before["source"] == "connected capture":
            dialog.preview_connected()
        else:
            manifest = folder / before["capture_manifest"]
            dialog.preview_capture(
                capture_from_recipe(import_capture(manifest)), title="Independent manifest replay"
            )
        finish(app, dialog)
        assert dialog.reference_health.text() == reference_health
        assert dialog.reference_result["report"] == reference_report
        # The block adds its separate coverage decision after the estimator returns.
        expected_fit = {
            key: value for key, value in expected["outputs"]["report"].items() if key != "decision"
        }
        assert json.loads(json.dumps(dialog.last_result["report"], allow_nan=False)) == expected_fit
        for name in ("overview", "quality", "residual", "normals"):
            save_rgb(evidence / f"preview_{name}.png", dialog.last_result["views"][name])
        record.update(
            setup=dialog.setup.text(),
            reference_health=reference_health,
            last_preview_health=dialog.health.text(),
            guidance=dialog.guidance.text(),
            reference_preview_separate=True,
            source_audit_without_fit=True,
            linked_light_selection=True,
            invalid_draft_disables_actions=True,
            edited_draft_discards_prior_evidence=True,
            original_settings_exact=True,
            original_reference_and_mask_exact=True,
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
        np.testing.assert_array_equal(reopened.reference.reference, original.reference)
        np.testing.assert_array_equal(reopened.reference.rig.directions, original.rig.directions)
        assert reopened.reference.metadata["reference_fit"] == json.loads(
            json.dumps(reference_report, allow_nan=False)
        )
        reopened.reject()
        applied = copy.deepcopy(block.parameters)
        graph.run()
        assert block.last_result is not None, block.error
        assert stable_result(block.last_result) == expected
        assert array_hashes(block.last_result) == arrays
        write_json(evidence / "runtime_report.json", block.last_result.outputs["report"])
        native_review(app, window, block)
        target = output / "relocated" / case / "project.json"
        target.parent.mkdir(parents=True)
        relative = block.parameters["recipe_path"]
        raw = (folder / relative).read_bytes()
        with ProjectSession.open(staged, registry) as session:
            assert session.inspect()["validation"]["valid"]
            session.save(target)
        with ProjectSession.open(target, registry) as copied:
            assert (target.parent / relative).read_bytes() == raw
            np.testing.assert_array_equal(
                load_recipe(target.parent / relative).reference, original.reference
            )
            copied.run()
            assert stable_result(copied.graph.blocks["surface"].last_result) == expected
            assert array_hashes(copied.graph.blocks["surface"].last_result) == arrays
        assert (folder / before["recipe_path"]).read_bytes() == original_bytes
        assert block.parameters == applied and not warnings, warnings
        record.update(
            stage_preserves_active_until_apply=True,
            cancel_preserves_active=True,
            exact_recipe_save_as=True,
            original_recipe_preserved=True,
            completed_review_without_estimation=True,
            reports_and_raw_maps_and_graphics_exact=True,
            runtime_count=block.last_result.outputs["valid_pixels"],
            runtime_passed=bool(block.last_result.passed),
        )
    finally:
        window.dirty = False
        window.close()
        window.deleteLater()
        app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    return record


def verify(root, output, cases=CASES):
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(root / "projects", output / "projects")
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
