"""Exercise the actual inspector's training/Stage/Apply/reopen workflow offscreen."""

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from opencv_dataflow.app import MainWindow
from opencv_dataflow.assistant_api import ProjectSession
from opencv_dataflow.core import Graph
from opencv_dataflow.plugins import load_plugin
from PySide6.QtWidgets import QApplication


def write_json(path, value):
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def check(root, output):
    from opencv_geometric_pattern_match.image_input import read_gray

    app = QApplication.instance() or QApplication([])
    output.mkdir(parents=True, exist_ok=False)
    receipts = []
    for name in ("brackets", "visa_pcb2"):
        folder = output / name
        shutil.copytree(root / name, folder)
        window = MainWindow()
        warnings = []
        window.warn = warnings.append
        load_plugin(root / "plugin.py", window.registry)
        graph = Graph.load(folder / "project.json", window.registry)
        block = graph.blocks["matcher"]
        original_parameters = block.parameters.copy()
        original = folder / original_parameters["recipe_path"]
        original_bytes = original.read_bytes()
        window.replace_graph(graph)
        window.scene.nodes[block.id].setSelected(True)
        inspector = window.inspector

        def wait(dialog):
            deadline = time.monotonic() + 90
            while dialog._job is not None:
                app.processEvents()
                if time.monotonic() > deadline:
                    raise TimeoutError(dialog.status.text())
                time.sleep(0.01)

        try:
            inspector.open_plugin_dialog()
            dialog = inspector._plugin_dialog
            assert dialog is not None, warnings
            assert dialog._reference is not None
            assert "Fine edges:" in dialog.health.text()
            original_pattern = dialog._pattern
            scene_path = graph.blocks["scene"].parameters["paths"][-1]
            dialog.use_scene(read_gray(folder / scene_path), "Separate held-out scene")
            dialog.preview_button.click()
            wait(dialog)
            assert dialog._review is not None, dialog.status.text()
            accepted_before = len(dialog._review.matches)
            assert accepted_before > 0
            dialog.candidate.setCurrentIndex(1)
            dialog.refine_button.click()
            wait(dialog)
            comparison = dialog._refinement_comparison
            assert comparison is not None, dialog.status.text()
            assert comparison.previous is original_pattern
            evidence = folder / "comparison"
            evidence.mkdir(exist_ok=True)
            for mode in ("Before", "After", "Feature changes"):
                cv2.imwrite(
                    str(evidence / (mode.lower().replace(" ", "_") + ".png")),
                    comparison.image(mode),
                )
            (evidence / "summary.txt").write_bytes((comparison.summary() + "\n").encode())
            dialog.undo_refinement.click()
            assert dialog._pattern is original_pattern
            assert dialog._refinement_comparison is None
            dialog.preview_button.click()
            wait(dialog)
            dialog.candidate.setCurrentIndex(1)
            dialog.refine_button.click()
            wait(dialog)
            refined = dialog._pattern
            dialog.preview_button.click()
            wait(dialog)
            assert dialog._review is not None, dialog.status.text()
            accepted_after = len(dialog._review.matches)
            dialog.stage.click()
            assert inspector._plugin_dialog is None, dialog.status.text()
            staged = inspector.read_parameters()
            assert staged["recipe_path"] != original_parameters["recipe_path"]
            assert block.parameters == original_parameters
            assert inspector.apply(), warnings
            graph.save(folder / "refined_project.json")
            inspector.open_plugin_dialog()
            restored = inspector._plugin_dialog
            np.testing.assert_array_equal(restored._pattern.points, refined.points)
            np.testing.assert_array_equal(restored._pattern.weights, refined.weights)
            assert restored._reference is not None and restored._scene is not None
            assert restored._refinement_comparison is None  # Session-only comparison is explicit.
            assert "Fine edges:" in restored.health.text()
            assert original.read_bytes() == original_bytes
            inspector.close_plugin_dialog()
            with ProjectSession.open(folder / "refined_project.json", window.registry) as session:
                moved = output / "relocated" / name / "project.json"
                moved.parent.mkdir(parents=True)
                session.save(moved)
            with ProjectSession.open(moved, window.registry) as session:
                assert session.inspect()["validation"]["valid"]
                relocated = moved.parent / session.graph.blocks["matcher"].parameters["recipe_path"]
                assert relocated.read_bytes() == (folder / staged["recipe_path"]).read_bytes()
                assert relocated.with_suffix(".reference.png").is_file()
                assert relocated.with_suffix(".refinement.png").is_file()
            receipt = dict(
                case=name,
                accepted_before=accepted_before,
                accepted_after=accepted_after,
                moved=comparison.result.moved,
                downweighted=comparison.result.downweighted,
                fixed_pose_score_before=comparison.result.before.score,
                fixed_pose_score_after=comparison.result.after.score,
                restored_reference_and_health=True,
                undo_exact=True,
                stage_does_not_apply=True,
                apply_reopen=True,
                save_as_exact_recipe=True,
                relocated_training_images=True,
                original_recipe_preserved=True,
                original_recipe_sha256=hashlib.sha256(original_bytes).hexdigest(),
                scope="Refinement-image fit comparison, not held-out accuracy qualification.",
            )
            assert not warnings, warnings
            write_json(folder / "validation.json", receipt)
            receipts.append(receipt)
            print(
                f"{name}: {accepted_before} -> {accepted_after} accepted; "
                "Stage/Apply/reopen/relocation pass",
                flush=True,
            )
        finally:
            window.dirty = False
            window.close()
            app.processEvents()
    write_json(output / "validation.json", receipts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    # Load the selected code before importing any matcher modules.
    from opencv_dataflow.assistant_api import make_registry

    make_registry(plugins=[str((args.root / "plugin.py").resolve())])
    check(args.root.resolve(), args.output.resolve())
