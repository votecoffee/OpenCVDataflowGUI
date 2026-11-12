import builtins

import pytest
from PySide6.QtCore import Qt

from opencv_dataflow.app import MainWindow

MATCH_TITLE = Qt.MatchFlag.MatchExactly | Qt.MatchFlag.MatchRecursive


@pytest.mark.parametrize(
    ("title", "type_id", "package"),
    [
        ("SimplePatternMatch", "pattern.ogpm_recipe", "opencv_geometric_pattern_match"),
        ("SimpleSegmentation", "inspection.ocv_segment_image", "opencv_defect_segmentation"),
        ("SimplePathWidth", "inspect.simple_path_width", "opencv_simple_path_width"),
        ("SimpleReferenceInspect", "inspect.simple_reference", "opencv_simple_reference_inspect"),
    ],
)
def test_optional_simple_tools_are_searchable_without_importing(
    app, monkeypatch, title, type_id, package
):
    imports = []
    original_import = builtins.__import__

    def without_optional_package(name, *args, **kwargs):
        if name == f"{package}.dataflow":
            imports.append(name)
            raise ModuleNotFoundError(name=package)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_optional_package)
    window = MainWindow()
    warnings = []
    monkeypatch.setattr(window, "warn", warnings.append)
    try:
        window.search.setText(title)
        (item,) = window.library.findItems(title, MATCH_TITLE, 0)
        assert item.parent().parent().text(0) == "Simple Plugins"
        assert item.data(0, Qt.ItemDataRole.UserRole) == type_id
        assert type_id not in window.registry.types
        assert imports == []

        before = len(window.graph.blocks)
        window.add_block(item)
        assert imports == [f"{package}.dataflow"]
        assert len(window.graph.blocks) == before
        assert len(warnings) == 1 and "Install " in warnings[0]
    finally:
        window.dirty = False
        window.close()


@pytest.mark.parametrize(
    ("title", "type_id", "module"),
    [
        ("SimpleClassify", "inspect.simple_classify_image", None),
        ("SimplePatternMatch", "pattern.ogpm_recipe", "opencv_geometric_pattern_match.dataflow"),
        ("SimplePathWidth", "inspect.simple_path_width", "opencv_simple_path_width.dataflow"),
        ("SimpleReferenceInspect", "inspect.simple_reference", "opencv_simple_reference_inspect.dataflow"),
        (
            "SimpleSegmentation", "inspection.ocv_segment_image",
            "opencv_defect_segmentation.dataflow",
        ),
    ],
)
def test_simple_tools_add_to_a_new_graph_from_the_same_toolbox_group(app, title, type_id, module):
    if module:
        pytest.importorskip(module)
    window = MainWindow()
    try:
        window.search.setText(title)
        (item,) = window.library.findItems(title, MATCH_TITLE, 0)
        assert item.parent().parent().text(0) == "Simple Plugins"
        window.add_block(item)
        assert window.inspector.block.type_id == type_id
        assert window.inspector.block.name == title
        assert len(window.library.findItems(title, MATCH_TITLE, 0)) == 1
    finally:
        window.dirty = False
        window.close()
