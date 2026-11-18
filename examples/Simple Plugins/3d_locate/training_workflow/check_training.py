"""Native fly-around crop/undo/train/test/Stage/Apply/reopen/Save As checks."""

# Choose the pinned package before importing the host's lazy tool providers.
# ruff: noqa: E402

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
spec = importlib.util.spec_from_file_location("pinned_3d_training", ROOT / "plugin.py")
plugin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin)

import cv2
import numpy as np
from opencv_dataflow.app import MainWindow
from opencv_dataflow.assistant_api import ProjectSession, make_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import render_cloud
from PySide6.QtWidgets import QApplication

from opencv_simple_3d_locate import load_recipe
from opencv_simple_3d_locate.training_setup import crop_selection


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )


def finish(app, dialog):
    deadline = time.monotonic() + 60
    while dialog.future is not None:
        app.processEvents()
        assert time.monotonic() < deadline, dialog.status.text()
        time.sleep(0.002)


def without_timing(value):
    if isinstance(value, dict):
        return {k: without_timing(v) for k, v in value.items() if k != "seconds"}
    if isinstance(value, list):
        return [without_timing(v) for v in value]
    return value


def verify_case(root, name, registry, app):
    folder = root / name
    project = folder / "project.json"
    originals = {
        p.relative_to(folder).as_posix(): digest(p)
        for p in (folder / "assets").rglob("*")
        if p.is_file()
    }
    graph = Graph.load(project, registry)
    window = MainWindow()
    window.registry = registry
    warnings = []
    window.warn = warnings.append
    window.replace_graph(graph, project)
    window.show()
    block = graph.blocks["locate"]
    before = copy.deepcopy(block.parameters)
    inspector = window.inspector
    inspector.select(block)
    record = {"case": name, "desktop_input_used": False, "window_capture_used": False}
    try:
        inspector.open_plugin_dialog()
        app.processEvents()
        dialog = inspector._plugin_dialog
        original = dialog.recipe
        assert original is not None and dialog.views.currentIndex() == 0
        np.testing.assert_array_equal(dialog.native.cloud.points, original.samples[:, :3])
        dialog.native.flyaround.setChecked(True)
        assert dialog.native.orbit_timer.isActive()
        yaw = dialog.native.yaw.value()
        dialog.native.advance_orbit()
        assert dialog.native.yaw.value() != yaw
        dialog.native.flyaround.setChecked(False)
        dialog.reject()
        assert block.parameters == before
        inspector.open_plugin_dialog()
        dialog = inspector._plugin_dialog
        original = dialog.recipe
        cad = dialog.cad
        with patch(
            "opencv_simple_3d_locate.training_setup.Locator",
            side_effect=AssertionError("Compiled during crop"),
        ):
            dialog._fill_bounds()
            low = dialog.reference.points.min(0)
            high = dialog.reference.points.max(0)
            high[0] = low[0] + (high[0] - low[0]) * 0.75
            dialog.crop_min.setText(repr(low.tolist()))
            dialog.crop_max.setText(repr(high.tolist()))
            dialog._preview_crop()
            selected = crop_selection(dialog.reference, low, high)
            count = int(selected.sum())
            assert 30 <= count < len(dialog.reference.points)
            write_cloud(folder / "crop_preview.ply", dialog.native.cloud, binary=True)
            assert cv2.imwrite(
                str(folder / "crop_preview.png"),
                render_cloud(dialog.native.cloud, 960, 640, yaw=25, pitch=-20, stable_scale=True),
            )
            record["crop_preview"] = {
                "source_points": len(selected),
                "kept_points": count,
                "low": low.tolist(),
                "high": high.tolist(),
                "legend": (
                    "Cyan kept source; gray excluded source; "
                    "orange derived bounds, not measured points"
                ),
            }
            assert dialog.recipe is original
            dialog._crop()
            assert dialog.recipe is None and dialog.cad is None
            assert len(dialog.reference.points) == count
            dialog._undo_crop()
            assert dialog.cad is cad and not dialog._crop_history
            np.testing.assert_array_equal(dialog.reference.points, original.reference[:, :3])
            np.testing.assert_array_equal(dialog.reference.normals, original.reference[:, 3:])
        # Cancelled work never stages or replaces the applied recipe.
        dialog.train_button.click()
        dialog.cancel_button.click()
        finish(app, dialog)
        assert dialog.recipe is None and block.parameters == before
        dialog.train_button.click()
        finish(app, dialog)
        assert dialog.recipe is not None, dialog.status.text()
        assert dialog.recipe.settings == original.settings
        np.testing.assert_array_equal(dialog.recipe.reference, original.reference)
        np.testing.assert_array_equal(dialog.recipe.samples, original.samples)
        np.testing.assert_array_equal(dialog.recipe.cad.mesh.triangles, original.cad.mesh.triangles)
        record["unchanged_training_arrays_exact"] = True
        record["preparation_health"] = dialog.health_label.text()
        # Capture the actual completed search/camera/depth together; test truth is
        # never supplied as a pose seed or used to tune a recipe.
        graph.run()
        assert block.last_result is not None, block.error
        baseline = copy.deepcopy(block.last_result.outputs["report"])
        expected_count = block.last_result.outputs["count"]
        assert expected_count == (1 if name == "real_positive" else 0)
        dialog.capture_preview_button.click()
        assert dialog.processed_capture is not None
        record["setup"] = dialog.setup_label.text()
        dialog.preview_button.click()
        finish(app, dialog)
        assert dialog.result is not None, dialog.status.text()
        assert len(dialog.result.matches) == expected_count
        assert dialog.result.report()["matches"] == baseline["matches"]
        selection = "all accepted instances"
        if not dialog.result.matches:
            row = dialog.result_selection.findData(("rejected", 0))
            if row >= 0:
                dialog.result_selection.setCurrentIndex(row)
                selection = "rejected candidate, not a detection"
        assert dialog.native.cloud is not None and dialog.native.cloud.colors is not None
        write_cloud(folder / "reviewed_fit.ply", dialog.native.cloud, binary=True)
        assert cv2.imwrite(
            str(folder / "reviewed_fit.png"),
            render_cloud(dialog.native.cloud, 960, 640, yaw=25, pitch=-20, stable_scale=True),
        )
        record.update(
            accepted_count=expected_count,
            selected_evidence=selection,
            test_report=without_timing(dialog.result.report()),
            native_3d_preview=True,
            crop_does_not_prepare_or_locate=True,
            restore_cancel=True,
            crop_undo_restores_cad=True,
            cancelled_job_not_staged=True,
        )
        assert block.parameters == before
        dialog._stage()
        assert block.parameters == before
        inspector.apply()
        assert not warnings, warnings
        assert block.parameters["recipe_path"] != before["recipe_path"]
        staged = folder / "staged_project.json"
        graph.save(staged)
        inspector.open_plugin_dialog()
        reopened = inspector._plugin_dialog
        np.testing.assert_array_equal(reopened.recipe.reference, original.reference)
        np.testing.assert_array_equal(reopened.recipe.samples, original.samples)
        assert reopened.camera is None and reopened.test_scene is None
        assert reopened.views.currentIndex() == 0 and reopened.native.cloud is not None
        reopened.reject()
        target = root / ".local" / name / "relocated/project.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        relative = block.parameters["recipe_path"]
        expected = digest(folder / relative)
        with ProjectSession.open(staged, registry) as session:
            assert session.inspect()["validation"]["valid"]
            session.save(target)
        with ProjectSession.open(target, registry) as relocated:
            assert relocated.inspect()["validation"]["valid"]
            assert digest(target.parent / relative) == expected
            saved = load_recipe(target.parent / relative)
            np.testing.assert_array_equal(saved.reference, original.reference)
            np.testing.assert_array_equal(saved.cad.mesh.triangles, original.cad.mesh.triangles)
        for relative, expected in originals.items():
            assert digest(folder / relative) == expected, relative
        record.update(
            stage_preserves_applied_until_apply=True,
            apply_reopen=True,
            exact_recipe_save_as=True,
            original_assets_preserved=True,
        )
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
    return record


def verify(root, output):
    output.mkdir(parents=True, exist_ok=False)
    for name in ("real_positive", "real_no_match"):
        shutil.copytree(
            root / name,
            output / name,
            ignore=shutil.ignore_patterns(
                ".local", "__pycache__", "staged_project.json", "crop_preview.*", "reviewed_fit.*"
            ),
        )
    registry = make_registry(plugins=[str(ROOT / "plugin.py")])
    app = QApplication.instance() or QApplication([])
    records = [
        verify_case(output, name, registry, app) for name in ("real_positive", "real_no_match")
    ]
    write_json(output / "validation_training.json", records)
    print(
        json.dumps(
            {
                "native_workflows": 2,
                "accepted_counts": [r["accepted_count"] for r in records],
                "trained_arrays_exact": True,
                "exact_recipe_save_as": True,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(ROOT, args.output.resolve())
