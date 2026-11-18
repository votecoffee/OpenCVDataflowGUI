"""Native Classify dataset-check/correction and Stage/Apply/reopen qualification."""

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
from opencv_dataflow.app import MainWindow
from opencv_dataflow.assistant_api import ProjectSession
from opencv_dataflow.core import Graph
from opencv_dataflow.plugins import load_plugin
from PySide6.QtWidgets import QApplication


def write(path, value):
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def audit_report(audit):
    return dict(
        ready=audit.ready,
        split=audit.split,
        classes=[
            dict(
                label=row.label,
                assigned=row.assigned,
                readable=row.readable,
                training=row.training,
                validation=row.validation,
            )
            for row in audit.classes
        ],
        issues=[
            dict(
                codes=row.codes,
                class_index=row.class_index,
                image_index=row.image_index,
                message=row.message,
            )
            for row in audit.issues
        ],
        samples=[
            dict(
                class_index=row.class_index,
                image_index=row.image_index,
                name=row.path.name,
                shape=row.shape,
                validation=row.validation,
                readable=row.error is None,
                fingerprint=row.fingerprint,
            )
            for row in audit.samples
        ],
    )


def check(root, output):
    from opencv_simple_classifier.graphics import draw_assessment
    from opencv_simple_classifier.model import read_image
    from opencv_simple_classifier.qt_training import _EXAMPLE

    app = QApplication.instance() or QApplication([])
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(
        root / "vegqual", output / "vegqual", ignore=shutil.ignore_patterns(".local", "__pycache__")
    )
    folder = output / "vegqual"
    receipts = []
    for name in ("color_and_layout", "surface_texture"):
        window = MainWindow()
        warnings = []
        window.warn = warnings.append
        load_plugin(root / "plugin.py", window.registry)
        graph = Graph.load(folder / f"{name}.json", window.registry)
        block = graph.blocks["classifier"]
        original = block.parameters.copy()
        original_files = {p: (folder / p).read_bytes() for p in block.project_assets()}
        window.replace_graph(graph, folder / f"{name}.json")
        window.scene.nodes[block.id].setSelected(True)
        inspector = window.inspector
        review_dir = folder / "training_checks" / name
        review_dir.mkdir(parents=True, exist_ok=True)

        def wait(dialog):
            deadline = time.monotonic() + 90
            while dialog.future is not None:
                app.processEvents()
                if time.monotonic() > deadline:
                    raise TimeoutError(dialog.status.text())
                time.sleep(0.005)

        try:
            inspector.open_plugin_dialog()
            dialog = inspector._plugin_dialog
            assert dialog is not None, warnings
            assert dialog.report.rowCount() == 4
            with patch.object(
                dialog._review_model,
                "predict",
                side_effect=AssertionError("Audit must not predict"),
            ):
                dialog.audit_button.click()
                wait(dialog)
            assert dialog._audit.ready
            write(review_dir / "healthy_dataset.json", audit_report(dialog._audit))
            # Review a real held-out error when one exists; do not relabel it.
            rows = list(dialog.predictions.values())
            if not rows:  # Legacy recipes stored aggregate health but no per-image predictions.
                rows = []
                for class_index, samples in enumerate(dialog.assignments().values()):
                    for image_index, example in enumerate(samples):
                        if example.validation:
                            predicted = dialog._review_model.predict(read_image(example.path))[
                                "label"
                            ]
                            predicted_index = next(
                                i
                                for i in range(dialog.tabs.count())
                                if dialog.tabs.tabText(i) == predicted
                            )
                            rows.append(
                                SimpleNamespace(
                                    class_index=class_index,
                                    image_index=image_index,
                                    predicted_index=predicted_index,
                                )
                            )
            sample = next((row for row in rows if row.class_index != row.predicted_index), rows[0])
            dialog.tabs.setCurrentIndex(sample.class_index)
            strip = dialog.tabs.currentWidget()
            strip.item(sample.image_index).setSelected(True)
            example = strip.item(sample.image_index).data(_EXAMPLE)
            image = read_image(example.path)
            prediction = dialog._review_model.predict(image)
            cv2.imwrite(
                str(review_dir / "sample_assessment.png"), draw_assessment(image, prediction)
            )
            write(
                review_dir / "sample_assessment.json",
                dict(
                    expected=dialog.tabs.tabText(sample.class_index),
                    image=example.path.name,
                    result=prediction,
                    scope=(
                        "Actual restored model/held-out sample; "
                        "no correction or new inference gate."
                    ),
                ),
            )
            dialog.review_button.click()
            assert len(dialog._review_window.views) == 2
            dialog._review_window.views[0].actual_size()
            dialog._review_window.close()

            controls = folder / "audit_controls" / name
            controls.mkdir(parents=True, exist_ok=True)
            first = next(
                example
                for example in next(iter(dialog.assignments().values()))
                if not example.validation
            )
            duplicate = controls / "duplicate.jpg"
            duplicate.write_bytes(first.path.read_bytes())
            corrupt = controls / "unreadable.png"
            corrupt.write_bytes(b"Authored deliberately undecodable dataset-check fixture")
            dialog.tabs.setCurrentIndex(dialog.tabs.count() - 1)
            strip = dialog.tabs.currentWidget()
            previous_count = strip.count()
            strip.clearSelection()
            dialog.add_images([duplicate, corrupt])
            strip.item(previous_count).setSelected(True)
            dialog.split_button.click()
            dialog.audit_button.click()
            wait(dialog)
            assert not dialog._audit.ready
            issues = audit_report(dialog._audit)
            assert {code for row in issues["issues"] for code in row["codes"]} >= {
                "unreadable",
                "duplicate",
                "label_conflict",
                "split_leakage",
            }
            write(review_dir / "deliberate_problems.json", issues)
            for target in (duplicate, corrupt):
                audit = dialog._audit
                key = next(
                    (row.class_index, row.image_index)
                    for row in audit.samples
                    if row.path == target
                )
                row = next(
                    index
                    for index, issue in enumerate(audit.issues)
                    if (issue.class_index, issue.image_index) == key
                )
                dialog.tabs.setCurrentIndex(0)
                dialog.audit_table.setCurrentCell(row, 0)
                dialog.audit_table.cellClicked.emit(row, 0)
                [selected] = dialog.tabs.currentWidget().selectedItems()
                assert selected.data(_EXAMPLE).path == target
                dialog.remove_button.click()
                assert target.is_file() and dialog._audit is None
                dialog.audit_button.click()
                wait(dialog)
            assert dialog._audit.ready
            write(review_dir / "corrected_dataset.json", audit_report(dialog._audit))
            dialog.train_button.click()
            wait(dialog)
            assert dialog.candidate is not None, dialog.status.text()
            _, candidate, health = dialog.candidate
            dialog.stage_button.click()
            assert inspector._plugin_dialog is None
            staged = inspector.read_parameters()
            assert block.parameters == original and staged["model"] != original["model"]
            assert inspector.apply(), warnings
            graph.save(folder / f"retrained_{name}.json")
            inspector.open_plugin_dialog()
            restored = inspector._plugin_dialog
            assert restored.report.rowCount() == 4
            assert restored.profile.currentData() == candidate.profile
            restored.audit_button.click()
            wait(restored)
            assert restored._audit.ready
            assert [row.training for row in restored._audit.classes] == health.train_counts
            assert [row.validation for row in restored._audit.classes] == health.validation_counts
            inspector.close_plugin_dialog()
            moved = output / "relocated" / name / "project.json"
            moved.parent.mkdir(parents=True)
            with ProjectSession.open(folder / f"retrained_{name}.json", window.registry) as session:
                assets = {
                    p: (folder / p).read_bytes()
                    for p in session.graph.blocks["classifier"].project_assets()
                }
                session.save(moved)
            for relative, content in assets.items():
                assert (moved.parent / relative).read_bytes() == content
            for relative, content in original_files.items():
                assert (folder / relative).read_bytes() == content
            receipt = dict(
                profile=name,
                assigned_images=sum(len(rows) for rows in dialog.training_assignments.values()),
                training=health.train_counts,
                held_out=health.validation_counts,
                original_health_restored=True,
                scan_does_not_predict=True,
                deliberate_problem_codes_detected=True,
                linked_correction=True,
                removed_assignments_keep_files=True,
                trained_staged_applied_reopened=True,
                exact_assets_after_save_as=True,
                original_assets_preserved=True,
                new_health=health.score,
                manifest_sha256=hashlib.sha256((folder / staged["model"]).read_bytes()).hexdigest(),
                scope=(
                    "Same fixed profiles/splits; diagnostic faults removed before fitting. "
                    "Not an accuracy improvement claim."
                ),
            )
            assert not warnings, warnings
            receipts.append(receipt)
            print(
                f"{name}: audit, correction, train/Stage/Apply/reopen and Save As pass", flush=True
            )
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
