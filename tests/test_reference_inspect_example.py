import shutil
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.app import MainWindow
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/Simple Plugins/reference_inspect"


def test_relocated_reference_inspection_replays_and_publishes_the_overlay(tmp_path):
    pytest.importorskip("opencv_simple_reference_inspect.dataflow")
    moved = tmp_path / "relocated"
    shutil.copytree(EXAMPLE, moved)
    graph = Graph.load(moved / "project.json", builtin_registry())
    graph.filmstrip_enabled = True
    try:
        for state, area in (("pass", 0), ("fail", 754)):
            graph.run()
            block = graph.blocks["inspection"]
            assert block.state == state, block.error
            outputs = block.last_result.outputs
            assert outputs["inspection"]["evaluated"]
            assert outputs["defect_area"] == area
            assert np.count_nonzero(outputs["defect_mask"]) == area
            assert outputs["overlay"].shape == (192, 256, 3)
        assert len(graph.filmstrip.snapshot()) == 2
        assert all(row["override"] is not None for row in graph.filmstrip.snapshot())
    finally:
        graph.close()


def test_reference_trainer_restores_samples_and_stages_a_retrained_recipe(app, monkeypatch, tmp_path):
    model_io = pytest.importorskip("opencv_simple_reference_inspect")
    read_image = pytest.importorskip("opencv_simple_reference_inspect.cli").read_image
    moved = tmp_path / "training"
    shutil.copytree(EXAMPLE, moved)
    project = moved / "project.json"
    graph = Graph.load(project, builtin_registry())
    window = MainWindow()
    warnings = []
    monkeypatch.setattr(window, "warn", warnings.append)
    try:
        window.replace_graph(graph, project)
        block = graph.blocks["inspection"]
        original = dict(block.parameters)
        window.inspector.select(block)
        window.inspector.open_plugin_dialog()
        dialog = window.inspector._plugin_dialog
        assert dialog.filmstrip.count() == len(dialog.samples) == 8
        assert dialog.model is not None and dialog.inspection_mask is not None
        dialog.filmstrip.setCurrentRow(7)
        dialog._remove()
        assert len(dialog.samples) == 7 and dialog.model is None
        dialog.fields["absolute_threshold"].setText("14")
        dialog.train_button.click()
        assert dialog.model is not None and dialog.model.settings.absolute_threshold == 14
        dialog.test_image = read_image(moved / "assets/changed.png")
        dialog.preview_button.click()
        assert dialog.inspection is not None and not dialog.inspection[0].passed
        assert "Fail" in dialog.status.text()
        dialog.stage_button.click()
        assert block.parameters == original
        window.inspector.apply()
        assert not warnings and block.parameters["recipe_path"] != original["recipe_path"]
        saved = model_io.load_model(moved / block.parameters["recipe_path"], sha256=block.parameters["sha256"])
        assert len(saved.samples) == 7 and saved.settings.absolute_threshold == 14
        window.inspector.open_plugin_dialog()
        assert len(window.inspector._plugin_dialog.samples) == 7
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
