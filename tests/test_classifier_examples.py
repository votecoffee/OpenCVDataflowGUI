import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QFileDialog
from test_editor import pump_until

from opencv_dataflow.app import MainWindow
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.plugins import load_plugin
from opencv_dataflow.simple_classifier.dataflow import ClassifyImage
from opencv_dataflow.simple_classifier.model import (
    APPEARANCE,
    SURFACE_TEXTURE,
    load_assignments,
    read_image,
)

APPLICATION = Path(__file__).resolve().parents[1] / "examples/applications/image_classification"
VEGQUAL = APPLICATION / "vegqual"


@pytest.mark.parametrize(
    ("filename", "profile", "validation_score", "test_hits"),
    [("project.json", SURFACE_TEXTURE, 95, 10), ("color_and_layout.json", APPEARANCE, 70, 8)],
)
def test_vegqual_saved_project_runs_all_labeled_test_images(
    filename, profile, validation_score, test_hits
):
    truth = json.loads((VEGQUAL / "ground_truth.json").read_text(encoding="utf-8"))
    expected = [row for row in truth if row["source_split"] == "test"]
    labels = {row["label"] for row in truth}
    assert len(expected) == 16
    graph = Graph.load(VEGQUAL / filename, builtin_registry())
    try:
        classifier = graph.blocks["classifier"]
        _, health = load_assignments(VEGQUAL, classifier.parameters["model"])
        assert health.profile == profile
        assert sum(health.validation_counts) == 20
        assert len(health.predictions) == 20
        assert health.score == validation_score
        assert graph.blocks["test_images"].parameters["paths"] == [r["image"] for r in expected]
        correct = 0
        for truth_row in expected:
            graph.run()
            block = graph.blocks["classifier"]
            assert block.state == "pass", block.error
            prediction = block.last_result.outputs
            candidates = prediction["probabilities"]
            assert {row["label"] for row in candidates} == labels
            assert math.fsum(row["score"] for row in candidates) == pytest.approx(100)
            assert prediction["label"] == candidates[0]["label"]
            assert prediction["score"] == candidates[0]["score"]
            assert prediction["confusion"] == pytest.approx(
                100 * candidates[1]["score"] / candidates[0]["score"]
            )
            correct += prediction["label"] == truth_row["label"]
        assert correct == test_hits
        assert graph.sources_exhausted()
    finally:
        graph.close()


def test_vegqual_opens_bundled_trainer_and_manual_plugin_is_repeatable(app, monkeypatch):
    path = VEGQUAL / "project.json"
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(path), ""))
    window = MainWindow()
    warnings = []
    monkeypatch.setattr(window, "warn", warnings.append)
    try:
        assert window.registry.types[ClassifyImage.type_id] is ClassifyImage
        window.open_project()
        assert window.path == path
        assert not warnings
        block = window.graph.blocks["classifier"]
        window.inspector.select(block)
        window.inspector.open_plugin_dialog()
        dialog = window.inspector._plugin_dialog
        assert dialog is not None, warnings
        assert dialog.profile.currentData() == SURFACE_TEXTURE
        dialog.tabs.setCurrentIndex(1)
        dialog.review_filter.setCurrentIndex(2)
        assert dialog.review_counts.text() == "5 held out · 1 misclassified"
        dialog.reject()
        load_plugin(APPLICATION / "classifier_plugin.py", window.registry)
        assert window.registry.types[ClassifyImage.type_id] is ClassifyImage
    finally:
        window.dirty = False
        window.close()


def test_train_from_live_wired_images_and_reopen_staged_captures(app, monkeypatch, tmp_path):
    project = tmp_path / "Capture training.json"
    registry = builtin_registry()
    frames = []
    for index, color in enumerate(((12, 25, 42), (18, 29, 49), (139, 179, 227), (145, 186, 235))):
        path = tmp_path / f"frame_{index}.png"
        assert cv2.imwrite(str(path), np.full((28, 40, 3), color, np.uint8))
        frames.append(path.name)
    graph = Graph()
    try:
        source = graph.add(registry.create("input.file", parameters={
            "source_mode": "files", "paths": frames, "path_base": "project",
            "color_format": "rgb",
        }))
        pyramid = graph.add(registry.create("cv.image_pyramid", parameters={"levels": 2}))
        classifier = graph.add(registry.create(ClassifyImage.type_id, block_id="classifier"))
        classifier.input_levels = {"image": "1"}
        graph.connect(source.id, "image", pyramid.id, "image")
        graph.connect(pyramid.id, "images", classifier.id, "image")
        graph.save(project)
    finally:
        graph.close()

    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(project), ""))
    window = MainWindow()
    warnings = []
    monkeypatch.setattr(window, "warn", warnings.append)
    window.show()
    try:
        window.open_project()
        block = window.graph.blocks["classifier"]
        window.scene.nodes[block.id].setSelected(True)
        inspector = window.inspector
        inspector.open_plugin_dialog()
        dialog = inspector._plugin_dialog
        assert dialog.isVisible() and not dialog.isModal()
        dialog.add_class("Dark")
        dialog.add_class("Bright")
        dialog.capture_panel_button.click()
        picker = dialog.image_picker
        assert picker.isVisible()
        picker.capture_button.click()
        assert "Run the graph" in picker.status_label.text()
        assert dialog.capture_directory is None

        captures = []
        for index, frame in enumerate(frames):
            class_index = index // 2
            label = dialog.tabs.tabText(class_index)
            dialog.tabs.setCurrentIndex(class_index)
            window.start_run(False)
            pump_until(app, lambda: window.runner is None)
            assert inspector._plugin_dialog is dialog and dialog.isVisible()
            picker.capture_button.click()
            sample = dialog.assignments()[label][-1]
            captures.append(sample)
            expected = cv2.pyrDown(read_image(tmp_path / frame))
            np.testing.assert_array_equal(read_image(sample.path), expected)
            assert sample.name == f"Capture {index + 1:03d}"
            assert frame in picker.status_label.text()
            picker.capture_button.click()
            assert "already assigned" in picker.status_label.text()
            assert len(dialog.assignments()[label]) == index % 2 + 1
        assert not (tmp_path / "assets").exists()

        dialog.train_button.click()
        dialog.future.result(timeout=10)
        pump_until(app, dialog.stage_button.isEnabled)
        dialog.stage_button.click()
        assert inspector._plugin_dialog is None
        assert block.parameters["model"] == ""
        assert all(not sample.path.exists() for sample in captures)
        assert inspector.apply()
        assert window.save_project()
        assert not warnings

        saved = Graph.load(project, registry)
        try:
            manifest = saved.blocks["classifier"].parameters["model"]
            assignments, _ = load_assignments(tmp_path, manifest)
            staged = [*assignments["Dark"], *assignments["Bright"]]
            for old, new, frame in zip(captures, staged, frames, strict=True):
                assert new.display_name == old.display_name
                assert new.path.is_relative_to(tmp_path / "assets" / "image_classification")
                np.testing.assert_array_equal(
                    read_image(new.path), cv2.pyrDown(read_image(tmp_path / frame))
                )
        finally:
            saved.close()
    finally:
        window.dirty = False
        window.close()
