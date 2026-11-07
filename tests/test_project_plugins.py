"""Verify installed integrations resolve through the normal project open route."""

import builtins

import pytest
from PySide6.QtWidgets import QFileDialog

from opencv_dataflow.app import MainWindow
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.plugins import GEOMETRIC_MATCHER_TYPES


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


def test_missing_matcher_has_setup_guidance_and_unknown_types_stay_explicit(monkeypatch):
    import_module = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "opencv_geometric_pattern_match.dataflow":
            raise ModuleNotFoundError(name="opencv_geometric_pattern_match")
        return import_module(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    registry = builtin_registry()
    with pytest.raises(ValueError, match="examples/applications/README.md"):
        registry.create("pattern.ogpm_recipe")
    assert not set(GEOMETRIC_MATCHER_TYPES) & registry.types.keys()
    with pytest.raises(ValueError, match="Load its plugin first"):
        registry.create("example.unregistered")
