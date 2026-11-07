import math
from concurrent.futures import Future

import cv2
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog, QInputDialog

from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import ValuePacket
from opencv_dataflow.plugin_dialog import PluginDialogContext
from opencv_dataflow.simple_classifier.dataflow import ClassifyImage
from opencv_dataflow.simple_classifier.model import (
    APPEARANCE,
    SURFACE_TEXTURE,
    Example,
    load_assignments,
    load_classifier,
    ranked_scores,
    read_image,
    stage_classifier,
    train_classifier,
)
from opencv_dataflow.simple_classifier.qt_training import ClassifierTrainer


@pytest.fixture
def assignments(tmp_path):
    source = tmp_path / "labeled images"
    source.mkdir()
    classes = {}
    for label, value in (("Dark", 30), ("Medium", 130), ("Light", 225)):
        examples = []
        for index in range(6):
            image = np.full((36, 42, 3), value, np.uint8)
            image[0, index, :] = 0
            path = source / f"{label} {index}.png"
            path.write_bytes(cv2.imencode(".png", image)[1].tobytes())
            examples.append(Example(path, validation=index >= 4))
        classes[label] = examples
    return classes


def test_scores_include_all_classes_and_handle_ties():
    scores = ranked_scores(["A", "B", "C"], [50, 30, 20])
    assert scores == {
        "label": "A",
        "score": 50,
        "confusion": 60,
        "probabilities": [
            {"label": "A", "score": 50}, {"label": "B", "score": 30}, {"label": "C", "score": 20}
        ],
    }
    tie = ranked_scores(["A", "B", "C"], [1, 1, 1])
    assert tie["label"] == "A"
    assert tie["confusion"] == 100
    assert math.isclose(sum(row["score"] for row in tie["probabilities"]), 100)
    assert ranked_scores(["A", "B"], [0, 7])["confusion"] == 0


@pytest.mark.parametrize("profile", [APPEARANCE, SURFACE_TEXTURE])
def test_holdout_and_portable_model_roundtrip(assignments, tmp_path, profile):
    model, health = train_classifier(assignments, profile=profile)
    assert health.train_counts == [4, 4, 4]
    assert health.validation_counts == [2, 2, 2]
    assert health.score == 100
    assert health.matrix == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    assert [(row.class_index, row.image_index) for row in health.predictions] == [
        (class_index, image_index) for class_index in range(3) for image_index in (4, 5)
    ]
    assert all(row.class_index == row.predicted_index for row in health.predictions)
    root = tmp_path / "fixture naïve"
    root.mkdir()
    manifest = stage_classifier(root, assignments, model, health)
    reloaded, saved_health = load_assignments(root, manifest)
    assert saved_health == health
    assert saved_health.profile == profile
    assert all(row.path.is_relative_to(root) for rows in reloaded.values() for row in rows)
    loaded = load_classifier(root, manifest)
    assert loaded.profile == profile
    image = read_image(reloaded["Medium"][5].path)
    assert loaded.predict(image) == model.predict(image)
    assert loaded.predict(image)["label"] == "Medium"
    with pytest.raises(ValueError, match="project-relative"):
        load_classifier(root, "../manifest.json")


def test_automatic_holdout_and_duplicate_assignment(assignments):
    automatic = {
        label: [Example(example.path) for example in rows] for label, rows in assignments.items()
    }
    _, health = train_classifier(automatic)
    assert health.train_counts == [5, 5, 5]
    assert health.validation_counts == [1, 1, 1]
    assert {row.class_index for row in health.predictions} == {0, 1, 2}
    assert all(not rows[row.image_index].validation for row, rows in zip(
        health.predictions, automatic.values(), strict=True
    ))
    automatic["Light"].append(Example(automatic["Medium"][0].path))
    with pytest.raises(ValueError, match="Duplicate image"):
        train_classifier(automatic)


@pytest.mark.parametrize("profile", [APPEARANCE, SURFACE_TEXTURE])
def test_filmstrip_stage_and_remove_assignment(app, assignments, tmp_path, monkeypatch, profile):
    draft = {"model": ""}
    graph = Graph()
    graph.bind_project_dir(tmp_path)
    context = PluginDialogContext(graph, graph.add(ClassifyImage()))
    dialog = ClassifierTrainer(None, draft, context)
    try:
        dialog.profile.setCurrentIndex(dialog.profile.findData(profile))
        mislabeled = tmp_path / "Needs review.png"
        image = read_image(assignments["Light"][5].path)
        image[0, 8] = 0
        mislabeled.write_bytes(cv2.imencode(".png", image)[1].tobytes())
        assignments["Dark"].append(Example(mislabeled, validation=True))
        for label, rows in assignments.items():
            dialog.add_class(label, rows)
        dialog.show()
        strip = dialog.tabs.currentWidget()
        strip.setFocus()
        source = assignments["Light"][0].path
        strip.item(0).setSelected(True)
        app.processEvents()
        QTest.keyClick(strip, Qt.Key.Key_Delete)
        assert strip.count() == 5
        assert source.is_file()
        assert draft["model"] == ""
        dialog.train_button.click()
        dialog.future.result(timeout=10)
        QTest.qWait(100)
        assert dialog.stage_button.isEnabled(), dialog.status.text()
        assert dialog.report.rowCount() == 3
        assert draft["model"] == ""
        dialog.stage_button.click()
        assert dialog.result() == QDialog.DialogCode.Accepted
        reopened = ClassifierTrainer(None, draft, context)
        try:
            assert reopened.tabs.count() == 3
            assert reopened.profile.currentData() == profile
            assert len(reopened.assignments()["Light"]) == 5
            assert reopened.report.rowCount() == 3
            reopened.tabs.setCurrentIndex(0)
            reopened.review_filter.setCurrentIndex(2)
            strip = reopened.tabs.currentWidget()
            visible = [strip.item(i) for i in range(strip.count()) if not strip.item(i).isHidden()]
            [wrong] = visible
            assert "Predicted Light" in wrong.toolTip()
            assert "1 misclassified" in reopened.review_counts.text()
            wrong.setSelected(True)
            monkeypatch.setattr(QInputDialog, "getItem", lambda *args, **kwargs: ("Light", True))
            reopened.move_button.click()
            assert not reopened.review_filter.isEnabled()
            assert reopened.report.rowCount() == 0
            assert reopened.assignments()["Light"][-1] == Example(mislabeled, validation=True)
            assert mislabeled.is_file()
        finally:
            reopened.reject()
    finally:
        dialog.reject()
        graph.close()


def test_closing_trainer_keeps_captures_until_the_running_worker_finishes(app):
    graph = Graph()
    context = PluginDialogContext(graph, graph.add(ClassifyImage()))
    dialog = ClassifierTrainer(None, {"model": ""}, context)
    worker = Future()
    try:
        dialog.add_class("Sample")
        dialog._capture_image(ValuePacket(np.zeros((4, 6, 3), np.uint8), {}))
        [capture] = dialog.assignments()["Sample"]
        assert capture.path.is_file()
        assert worker.set_running_or_notify_cancel()
        dialog.future = worker
        dialog.reject()
        assert capture.path.is_file()
        worker.set_result(None)
        assert not capture.path.parent.exists()
    finally:
        if not worker.done():
            worker.cancel()
        dialog.reject()
        graph.close()
