"""Exercise source retention, native annotation review, Stage/Apply and relocation."""

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from opencv_dataflow.app import MainWindow
from opencv_dataflow.assistant_api import ProjectSession
from opencv_dataflow.core import Graph
from opencv_dataflow.plugins import load_plugin
from PySide6.QtWidgets import QApplication


def write(path, value):
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def check(root, output):
    from opencv_defect_segmentation.graphics import draw_heatmap
    from opencv_defect_segmentation.training_sources import annotation_preview, load_sources

    app = QApplication.instance() or QApplication([])
    output.mkdir(parents=True, exist_ok=False)
    receipts = []
    for name in ("pcb", "tote"):
        folder = output / name
        shutil.copytree(root / name, folder, ignore=shutil.ignore_patterns(".local", "__pycache__"))
        window = MainWindow()
        warnings = []
        window.warn = warnings.append
        load_plugin(root / "plugin.py", window.registry)
        graph = Graph.load(folder / "project.json", window.registry)
        block = graph.blocks["segment"]
        original = block.parameters.copy()
        original_model = (folder / original["model_path"]).read_bytes()
        window.replace_graph(graph, folder / "project.json")
        window.scene.nodes[block.id].setSelected(True)
        inspector = window.inspector

        def wait(condition):
            deadline = time.monotonic() + 120
            while not condition():
                app.processEvents()
                if time.monotonic() > deadline:
                    raise TimeoutError("Training review did not finish")
                time.sleep(0.005)
            app.processEvents()

        try:
            inspector.open_plugin_dialog()
            dialog = inspector._plugin_dialog
            assert dialog is not None, warnings
            dialog._set_path("training", folder / "training.json")
            dialog._set_path("validation", folder / "validation.json")
            dialog.train.click()
            wait(lambda: dialog._future is None)
            candidate = dialog._candidate
            assert candidate is not None and candidate.sources is not None, dialog.summary.text()
            _, region = dialog._read_region()
            review_dir = folder / "training_review"
            review_dir.mkdir(exist_ok=True)
            cv2.imwrite(
                str(review_dir / "annotations.png"),
                annotation_preview(candidate.sources, region)[..., ::-1],
            )
            cv2.imwrite(str(review_dir / "validation.png"), candidate.preview[..., ::-1])
            dialog.validation_labels.click()
            editor = dialog._editor
            assert editor is not None
            assert not editor.preview.isChecked()
            with patch.object(
                candidate.model, "segment", wraps=candidate.model.segment
            ) as inference:
                editor.preview.setChecked(True)
                cached = editor._predicted
                editor.heatmap.setChecked(True)
                editor.threshold.setValue(min(1.0, original["threshold"] + 0.05))
                editor.minimum_area.setValue(original["min_area_px"] + 1)
                assert inference.call_count == 1
                score = np.where(cached.inspected, cached.score, 0.0).astype(np.float32)
                heatmap = draw_heatmap(
                    editor.view.pixels,
                    SimpleNamespace(score_map=score, inspected=cached.inspected.astype(np.uint8)),
                )
                cv2.imwrite(str(review_dir / "votes.png"), heatmap)
                np.savez_compressed(
                    review_dir / "validation_scores.npz",
                    score=cached.score,
                    inspected=cached.inspected,
                    labels=editor.view.draft.labels,
                )
                editor.view.native()
                assert editor.close()
            wait(lambda: dialog._editor is None)
            assert dialog._candidate is candidate
            dialog.stage.click()
            assert inspector._plugin_dialog is None, dialog.summary.text()
            staged = inspector.read_parameters()
            assert block.parameters == original
            assert staged["model_path"] != original["model_path"]
            assert (staged["threshold"], staged["min_area_px"]) == (
                original["threshold"],
                original["min_area_px"],
            )
            assert inspector.apply(), warnings
            graph.save(folder / "trained_project.json")
            model = folder / staged["model_path"]
            protected = {
                path: path.read_bytes() for path in model.parent.rglob("*") if path.is_file()
            }
            inspector.open_plugin_dialog()
            restored = inspector._plugin_dialog
            wait(lambda: restored._future is None)
            assert "Restored" in restored.summary.text(), restored.summary.text()
            assert "No prediction" in restored.preview_title.text()
            assert restored._pinned_fingerprints
            assert Path(restored.paths["training"].text()).is_relative_to(folder / ".local")
            restored.labels.click()
            editor = restored._editor
            before = editor.view.draft.labels.copy()
            allowed = editor.view.draft.allowed
            y, x = np.argwhere(allowed)[len(np.argwhere(allowed)) // 2]
            editor._select_label(1 if before[y, x] == 2 else 2)
            editor.view.draft.begin_stroke()
            editor.view.draft.paint((int(x), int(y)), (int(x), int(y)), editor.view.label, 3)
            editor.view.draft.end_stroke()
            assert editor.view.draft.dirty
            editor.undo()
            np.testing.assert_array_equal(editor.view.draft.labels, before)
            assert editor.save()
            assert editor.close()
            wait(lambda: restored._editor is None)
            restored.reject()
            assert block.parameters == staged
            for path, content in protected.items():
                assert path.read_bytes() == content
            assert (folder / original["model_path"]).read_bytes() == original_model
            moved = output / "relocated" / name / "project.json"
            moved.parent.mkdir(parents=True)
            with ProjectSession.open(folder / "trained_project.json", window.registry) as session:
                dependencies = {
                    p: (folder / p).read_bytes()
                    for p in session.graph.blocks["segment"].project_assets()
                }
                session.save(moved)
            for path, content in dependencies.items():
                assert (moved.parent / path).read_bytes() == content
            with ProjectSession.open(moved, window.registry) as session:
                assert session.inspect()["validation"]["valid"]
                params = session.graph.blocks["segment"].parameters
                sources = load_sources(
                    moved.parent / params["model_path"],
                    params["model_sha256"],
                    params["region_sha256"] or None,
                )
                assert sources.files == candidate.sources.files
            receipt = dict(
                case=name,
                health=candidate.sources.health,
                held_out_metrics=candidate.validation_metrics,
                cutoff=original["threshold"],
                minimum_area=original["min_area_px"],
                train_stage_apply_reopen=True,
                private_annotation_draft=True,
                native_edit_undo_save=True,
                prediction_hidden_initially=True,
                cutoff_heatmap_area_reuse_one_inference=True,
                save_as_exact_all_sources=True,
                original_model_preserved=True,
                pinned_training_sources_preserved=True,
                model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                scope=("Fixed existing application settings and example annotations; "
                       "no general accuracy claim."),
            )
            assert not warnings, warnings
            write(folder / "validation_training.json", receipt)
            receipts.append(receipt)
            print(f"{name}: trained, annotated, staged/applied, restored and relocated", flush=True)
        finally:
            window.dirty = False
            window.close()
            app.processEvents()
    write(output / "validation_training.json", receipts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from opencv_dataflow.assistant_api import make_registry

    make_registry(plugins=[str((args.root / "plugin.py").resolve())])
    check(args.root.resolve(), args.output.resolve())
