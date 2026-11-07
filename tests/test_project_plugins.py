"""Verify installed integrations resolve through the normal project open route."""

import builtins
import time
from pathlib import Path

import pytest
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QFileDialog

from opencv_dataflow.app import MainWindow
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.plugins import (
    GEOMETRIC_MATCHER_TYPES,
    IMAGE_CLASSIFIER_TYPES,
    PIXEL_SEGMENTATION_TYPES,
)


def test_installed_matcher_opens_an_application_without_manual_plugin(app, monkeypatch, tmp_path):
    pytest.importorskip("opencv_geometric_pattern_match.dataflow")
    project = tmp_path / "inspection.json"
    graph = Graph()
    graph.add(builtin_registry().create("pattern.ogpm_recipe", block_id="matcher"))
    graph.save(project)
    graph.close()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(project), ""))
    window = MainWindow()
    warnings = []
    monkeypatch.setattr(window, "warn", warnings.append)
    try:
        assert not set(GEOMETRIC_MATCHER_TYPES) & window.registry.types.keys()
        window.open_project()
        assert not warnings
        assert window.path == project
        assert window.graph.blocks["matcher"].type_id == "pattern.ogpm_recipe"
        assert set(GEOMETRIC_MATCHER_TYPES) <= window.registry.types.keys()
    finally:
        window.dirty = False
        window.close()


@pytest.mark.local_examples
def test_installed_segmentation_opens_and_locates_boards_without_manual_plugin(app, monkeypatch):
    pytest.importorskip("opencv_defect_segmentation.dataflow")
    project = (
        Path(__file__).resolve().parents[1]
        / "examples/applications/pixel_segmentation/pcb_laminate/project.json"
    )
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(project), ""))
    window = MainWindow()
    warnings = []
    monkeypatch.setattr(window, "warn", warnings.append)
    try:
        assert not set(PIXEL_SEGMENTATION_TYPES) & window.registry.types.keys()
        window.open_project()
        assert not warnings
        assert window.path == project
        assert set(PIXEL_SEGMENTATION_TYPES) <= window.registry.types.keys()
        centers = []
        for _ in range(2):
            window.start_run(False)
            deadline = time.monotonic() + 10
            while window.runner is not None and time.monotonic() < deadline:
                app.processEvents()
                QTest.qWait(10)
            assert window.runner is None
            assert not warnings
            board = window.graph.blocks["board"]
            assert board.state == "pass", board.error
            assert board.last_result.outputs["count"] == 1
            [item] = board.last_result.outputs["collection"]["items"]
            x, y = (round(item["metrics"][key]) for key in ("center_x", "center_y"))
            preview = window.graph.blocks["board_preview"]
            assert preview.state == "pass", preview.error
            assert preview.last_result.outputs["overlay"][y, x].tolist() == [80, 230, 140]
            centers.append((x, y))
        assert centers[0][0] > centers[1][0] + 3
    finally:
        window.dirty = False
        window.close()


@pytest.mark.parametrize(
    ("module", "type_id", "known_types", "setup_path"),
    [
        (
            "opencv_geometric_pattern_match",
            "pattern.ogpm_recipe",
            GEOMETRIC_MATCHER_TYPES,
            "examples/applications/README.md",
        ),
        (
            "opencv_simple_classifier",
            "inspect.simple_classify_image",
            IMAGE_CLASSIFIER_TYPES,
            "examples/applications/README.md",
        ),
        (
            "opencv_defect_segmentation",
            "inspection.ocv_segment_image",
            PIXEL_SEGMENTATION_TYPES,
            "examples/applications/pixel_segmentation/README.md",
        ),
    ],
)
def test_missing_plugin_has_setup_guidance_and_unknown_types_stay_explicit(
    monkeypatch, module, type_id, known_types, setup_path
):
    import_module = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == f"{module}.dataflow":
            raise ModuleNotFoundError(name=module)
        return import_module(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    registry = builtin_registry()
    with pytest.raises(ValueError, match=setup_path):
        registry.create(type_id)
    assert not set(known_types) & registry.types.keys()
    with pytest.raises(ValueError, match="Load its plugin first"):
        registry.create("example.unregistered")


def test_installed_image_classifier_resolves_on_project_open(tmp_path):
    pytest.importorskip("opencv_simple_classifier.dataflow")
    registry = builtin_registry()
    assert not set(IMAGE_CLASSIFIER_TYPES) & registry.types.keys()
    project = tmp_path / "classifier.json"
    graph = Graph()
    try:
        graph.add(builtin_registry().create(IMAGE_CLASSIFIER_TYPES[0], block_id="classifier"))
        graph.save(project)
    finally:
        graph.close()
    graph = Graph.load(project, registry)
    try:
        assert graph.blocks["classifier"].plugin_dialog_label == "Train image classifier…"
    finally:
        graph.close()
    assert set(IMAGE_CLASSIFIER_TYPES) <= registry.types.keys()
