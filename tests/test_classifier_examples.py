import json
import math
from pathlib import Path

import pytest
from PySide6.QtWidgets import QFileDialog

from opencv_dataflow.app import MainWindow
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.plugins import load_plugin
from opencv_dataflow.simple_classifier.dataflow import ClassifyImage
from opencv_dataflow.simple_classifier.model import APPEARANCE, SURFACE_TEXTURE, load_assignments

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
