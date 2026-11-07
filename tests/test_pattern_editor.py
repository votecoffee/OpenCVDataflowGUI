"""Pattern training through the inspector and the actual background runner."""

import copy
import json
import time

import cv2
import pytest
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QFileDialog

from opencv_dataflow.app import MainWindow
from opencv_dataflow.core import Graph


def run(window, app):
    window.start_run(False)
    deadline = time.monotonic() + 10
    while window.runner is not None and time.monotonic() < deadline:
        app.processEvents()
        QTest.qWait(10)
    assert window.runner is None


@pytest.mark.parametrize("kind", ["shape", "texture"])
def test_inspector_training_worker_save_load_and_model_files(app, tmp_path, monkeypatch, kind):
    window = MainWindow()
    warnings = []
    monkeypatch.setattr(window, "warn", warnings.append)
    graph = Graph()
    source = graph.add(window.registry.create("input.pattern_demo"))
    matcher = graph.add(window.registry.create(f"pattern.match_{kind}"))
    graph.connect(source.id, f"{kind}_reference", matcher.id, "image")
    window.replace_graph(graph)
    run(window, app)
    assert matcher.state == "error" and "image" in matcher.last_inputs
    window.inspector.select(matcher)
    window.inspector.train_pattern()
    assert matcher.trained_model is not None and window.dirty
    model = copy.deepcopy(matcher.trained_model)
    run(window, app)
    assert matcher.state == "pass", matcher.error
    project = tmp_path / f"{kind}.json"
    graph.save(project)
    window.replace_graph(Graph.load(project, window.registry), project)
    matcher = window.graph.blocks[matcher.id]
    run(window, app)
    assert matcher.state == "pass", matcher.error
    assert matcher.trained_model == model
    window.inspector.select(matcher)
    model_file = tmp_path / f"{kind}_model.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(model_file), ""))
    window.inspector.save_pattern_model()
    saved = model_file.read_bytes()
    window.inspector.save_pattern_model()
    assert warnings and model_file.read_bytes() == saved
    window.inspector.clear_pattern_model()
    assert matcher.trained_model is None and window.dirty
    window.inspector.select(matcher)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(model_file), ""))
    window.inspector.load_pattern_model()
    assert matcher.trained_model == model
    bad = copy.deepcopy(model)
    bad["points"][0][0] += 1
    model_file.write_text(json.dumps(bad), encoding="utf-8")
    window.inspector.load_pattern_model()
    assert "checksum" in warnings[-1] and matcher.trained_model == model
    run(window, app)
    assert matcher.state == "pass", matcher.error

    # Native training failures must leave a usable previous model and warn in Qt.
    def failed_training(*args):
        raise cv2.error("native training failed")

    monkeypatch.setattr(type(matcher), "build_model", failed_training)
    window.inspector.train_pattern()
    assert "native training failed" in warnings[-1] and matcher.trained_model == model
    window.dirty = False
    window.close()


@pytest.mark.local_examples
def test_pose_inspection_edit_and_background_reload(app, load_example, tmp_path):
    window = MainWindow()
    load_example(window, "pose_inspection")
    run(window, app)
    assert all(b.state == "pass" for b in window.graph.blocks.values())
    matcher = window.graph.blocks["pose_match"]
    before = matcher.last_result.outputs["pose"]["origin"]
    window.inspector.select(matcher)
    window.inspector.fields["pose_offset_x"].setText("25.5")
    assert window.inspector.apply()
    run(window, app)
    after = matcher.last_result.outputs["pose"]["origin"]
    assert after[0] > before[0] + 4 and after[1] < before[1]
    region = window.graph.blocks["pose_region"]
    window.inspector.select(region)
    assert window.inspector.fields["shape"].currentText() == "circle"
    project = tmp_path / "pose_editor.json"
    window.graph.save(project)
    window.replace_graph(Graph.load(project, window.registry), project)
    run(window, app)
    assert all(b.state == "pass" for b in window.graph.blocks.values())
    assert window.graph.blocks["pose_match"].parameters["pose_offset_x"] == 25.5
    window.dirty = False
    window.close()
