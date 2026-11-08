"""Verify installed integrations resolve through the normal project open route."""

import builtins
import time
from pathlib import Path

import pytest
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QFileDialog

from opencv_dataflow.app import MainWindow
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph, block_display_title
from opencv_dataflow.plugins import (
    GEOMETRIC_MATCHER_TYPES,
    PATH_WIDTH_TYPES,
    PIXEL_SEGMENTATION_TYPES,
)
from opencv_dataflow.toolbox import block_entry


@pytest.mark.parametrize(
    ("module", "expected"),
    [
        (None, {"inspect.simple_classify_image": "SimpleClassify"}),
        ("opencv_simple_path_width.dataflow", {"inspect.simple_path_width": "SimplePathWidth"}),
        ("opencv_geometric_pattern_match.dataflow", {
            "pattern.ogpm_recipe": "SimplePatternMatch",
            "pattern.ogpm_load_recipe": "SimplePatternMatch · Load recipe",
            "pattern.ogpm_train_recipe": "SimplePatternMatch · Train recipe",
            "pattern.ogpm_match_model": "SimplePatternMatch · Match wired recipe",
        }),
        ("opencv_defect_segmentation.dataflow", {
            "inspection.ocv_segment_image": "SimpleSegmentation",
            "inspection.ocv_segment_parts": "SimpleSegmentation · Matched parts",
        }),
    ],
)
def test_simple_plugin_titles_preserve_saved_node_names(module, expected):
    if module:
        pytest.importorskip(module)
    registry = builtin_registry()
    for type_id, title in expected.items():
        graph = Graph()
        try:
            block = graph.add(registry.create(type_id))
            assert block.name == block_display_title(block) == title
            assert block_entry(registry.types[type_id]).title == title
            block.name = "Saved application-specific label"
            saved = graph.to_dict()
        finally:
            graph.close()
        reopened = Graph.from_dict(saved, registry)
        try:
            block = next(iter(reopened.blocks.values()))
            assert block.type_id == type_id
            assert block.name == "Saved application-specific label"
            assert block_display_title(block) == title
        finally:
            reopened.close()


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
        block = window.graph.blocks["matcher"]
        assert block.type_id == "pattern.ogpm_recipe"
        window.inspector.select(block)
        assert window.inspector.caption.toolTip().startswith("SimplePatternMatch /")
        assert set(GEOMETRIC_MATCHER_TYPES) <= window.registry.types.keys()
    finally:
        window.dirty = False
        window.close()


@pytest.mark.local_examples
def test_installed_segmentation_opens_and_locates_boards_without_manual_plugin(app, monkeypatch):
    pytest.importorskip("opencv_defect_segmentation.dataflow")
    project = (
        Path(__file__).resolve().parents[1]
        / "examples/Simple Plugins/pixel_segmentation/pcb_laminate/project.json"
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
            "opencv_simple_path_width",
            "inspect.simple_path_width",
            PATH_WIDTH_TYPES,
            "examples/Simple Plugins/path_width/README.md",
        ),
        (
            "opencv_geometric_pattern_match",
            "pattern.ogpm_recipe",
            GEOMETRIC_MATCHER_TYPES,
            "examples/Simple Plugins/README.md",
        ),
        (
            "opencv_defect_segmentation",
            "inspection.ocv_segment_image",
            PIXEL_SEGMENTATION_TYPES,
            "examples/Simple Plugins/pixel_segmentation/README.md",
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
