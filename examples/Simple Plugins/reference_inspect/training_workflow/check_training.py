"""Native training/check/cancel/Stage/Apply/reopen/Save As; no desktop input or package installation."""

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
for filename in ("pattern_plugin.py", "plugin.py"):
    spec = importlib.util.spec_from_file_location(
        filename.replace(".", "_"), ROOT / filename
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

import cv2
import numpy as np
from opencv_dataflow.app import MainWindow
from opencv_dataflow.assistant_api import ProjectSession, make_registry
from opencv_dataflow.core import Graph
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog

from opencv_simple_reference_inspect import ReferenceModel, load_model
from opencv_simple_reference_inspect.graphics import backdrop
from opencv_simple_reference_inspect.qt_masks import MaskEditor
from opencv_simple_reference_inspect.qt_review import ReferenceReviewDialog


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def finish(app, dialog):
    deadline = time.monotonic() + 45
    while dialog.future is not None:
        app.processEvents()
        assert time.monotonic() < deadline, dialog.status.text()
        time.sleep(0.002)


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
    block = graph.blocks["inspection"]
    before = copy.deepcopy(block.parameters)
    inspector = window.inspector
    inspector.select(block)
    record = {"case": name, "desktop_input_used": False, "window_capture_used": False}
    try:
        with patch.object(
            ReferenceModel,
            "inspect",
            side_effect=AssertionError("Inspected during restore"),
        ):
            inspector.open_plugin_dialog()
            app.processEvents()
            dialog = inspector._plugin_dialog
            original_model = dialog.model
            assert original_model is not None and dialog.train_button.isEnabled()
            assert len(dialog.samples) == len(original_model.samples) == 8
            dialog.reject()
        assert block.parameters == before
        inspector.open_plugin_dialog()
        dialog = inspector._plugin_dialog
        original_model = dialog.model
        reference = dialog.samples[0]
        original_masks = dialog.inspection_mask.copy(), dialog.normalization_mask.copy()

        def mask_undo(editor):
            saved = editor.mask
            editor._commit(np.zeros_like(saved))
            assert not editor.ok.isEnabled()
            editor.undo()
            np.testing.assert_array_equal(editor.mask, saved)
            return QDialog.DialogCode.Accepted

        with patch.object(MaskEditor, "exec", mask_undo):
            dialog._edit_mask("inspection_mask")
            dialog._edit_mask("normalization_mask")
        assert dialog.model is original_model

        # An explicit integer-translation control checks the import convention.
        # It is removed before fitting, so the retained training set is unchanged.
        matrix = np.array([[1.0, 0, 4], [0, 1.0, 3]])
        source = cv2.warpAffine(reference, matrix, reference.shape[::-1])
        source_path, pose_path = (
            folder / "alignment_control.tif",
            folder / "alignment_control.json",
        )
        ok, encoded = cv2.imencode(".tif", source)
        assert ok
        source_path.write_bytes(encoded.tobytes())
        write_json(
            pose_path,
            {"reference_to_image": matrix.tolist(), "generated_control": True},
        )
        with patch.object(
            QFileDialog,
            "getOpenFileName",
            side_effect=[(str(source_path), ""), (str(pose_path), "")],
        ):
            dialog._import_aligned()
        assert len(dialog.samples) == 9 and dialog.model is None
        support = dialog.sample_support[-1]
        np.testing.assert_array_equal(dialog.samples[-1][support], reference[support])
        record["imported_alignment"] = copy.deepcopy(dialog.sample_sources[-1])
        dialog.inspection_mask = np.ones(reference.shape, bool)
        dialog._invalidate()
        assert not dialog.train_button.isEnabled()
        assert any(dialog._unsupported_counts)
        record["unsupported_mask_guidance"] = dialog.setup_label.text()
        overlay = backdrop(dialog.samples[-1])
        overlay[~support] = [45, 95, 170]
        contours = cv2.findContours(
            original_masks[0].astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )[0]
        cv2.drawContours(overlay, contours, -1, (245, 220, 40), 2)
        assert cv2.imwrite(str(folder / "alignment_support.png"), overlay)
        dialog.inspection_mask, dialog.normalization_mask = original_masks
        dialog._remove()
        assert len(dialog.samples) == 8 and dialog.train_button.isEnabled()
        np.testing.assert_array_equal(np.stack(dialog.samples), original_model.samples)
        record["controls_removed_before_training"] = True
        dialog.train_button.click()
        dialog.cancel_button.click()
        finish(app, dialog)
        assert dialog.model is None and block.parameters == before
        record["cancelled_training_not_staged"] = True
        dialog.train_button.click()
        finish(app, dialog)
        assert dialog.model is not None, dialog.status.text()
        for field in (
            "samples",
            "mean",
            "deviation",
            "tolerance",
            "sample_support",
            "inspection_mask",
            "normalization_mask",
        ):
            np.testing.assert_array_equal(
                getattr(dialog.model, field), getattr(original_model, field)
            )
        for field in ("sample_count", "samples", "warnings", "mean_standard_deviation"):
            assert dialog.model.health[field] == original_model.health[field]
        record["trained_arrays_and_original_health_exact"] = True
        record["setup_guidance"] = dialog.setup_label.text()
        record["health"] = dialog.health_label.text()
        # Independent held-out input remains outside the good-image filmstrip.
        if name == "generated":
            from opencv_simple_reference_inspect.cli import read_image

            dialog.test_image = read_image(folder / "assets/changed.png")
            dialog._preview()
        else:
            graph.run()  # First held-out normal: existing false alarm stays visible.
            graph.run()  # First held-out anomaly, with the original fixture pose.
            assert block.last_result is not None, block.error
            dialog._preview_wired()
            assert (
                dialog.inspection[0].report == block.last_result.outputs["inspection"]
            )
        assert len(dialog.samples) == 8
        captured = dialog.inspection

        def review(view):
            view.show()
            app.processEvents()
            view._actual_size()
            for label in (
                "Signed residual",
                "Deviation / tolerance",
                "Inspected coverage",
            ):
                view.mode.setCurrentText(label)
            view.review.save(folder / "review")
            view.close()

        with (
            patch.object(ReferenceReviewDialog, "exec", review),
            patch.object(
                ReferenceModel,
                "inspect",
                side_effect=AssertionError("Inspected during review"),
            ),
        ):
            dialog._review()
        assert dialog.inspection is captured
        record["independent_inspection"] = captured[0].report
        assert block.parameters == before
        dialog._stage()
        assert block.parameters == before
        inspector.apply()
        assert not warnings, warnings
        assert block.parameters["recipe_path"] != before["recipe_path"]
        staged = folder / "staged_project.json"
        graph.save(staged)
        with patch.object(
            ReferenceModel,
            "inspect",
            side_effect=AssertionError("Inspected during reopen"),
        ):
            inspector.open_plugin_dialog()
            reopened = inspector._plugin_dialog
            np.testing.assert_array_equal(
                reopened.model.samples, original_model.samples
            )
            np.testing.assert_array_equal(reopened.inspection_mask, original_masks[0])
            np.testing.assert_array_equal(
                reopened.normalization_mask, original_masks[1]
            )
            assert len(reopened.sample_sources) == 8
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
            saved = load_model(target.parent / relative)
            np.testing.assert_array_equal(saved.samples, original_model.samples)
            np.testing.assert_array_equal(
                saved.sample_support, original_model.sample_support
            )
            assert len(saved.health["training_sources"]) == 8
        for relative, original_hash in originals.items():
            assert digest(folder / relative) == original_hash, relative
        record.update(
            native_mask_undo=True,
            restore_cancel=True,
            stage_before_apply_preserves_applied=True,
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
    for name in ("generated", "visa_posed"):
        shutil.copytree(
            root / name,
            output / name,
            ignore=shutil.ignore_patterns(
                ".local",
                "__pycache__",
                "review",
                "staged_project.json",
                "alignment_support.png",
            ),
        )
    registry = make_registry(
        plugins=[str(ROOT / "pattern_plugin.py"), str(ROOT / "plugin.py")]
    )
    app = QApplication.instance() or QApplication([])
    records = [
        verify_case(output, name, registry, app) for name in ("generated", "visa_posed")
    ]
    write_json(output / "validation_training.json", records)
    print(
        json.dumps(
            {
                "native_workflows": 2,
                "original_assets_preserved": True,
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
