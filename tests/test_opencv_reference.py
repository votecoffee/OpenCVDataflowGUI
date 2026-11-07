"""Check the offline OpenCV reference and the editable notes workflow."""

import cv2
from PySide6.QtWidgets import QPushButton, QTextBrowser

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Block, Graph
from opencv_dataflow.documentation_editor import DocumentationEditor
from opencv_dataflow.opencv_reference import BINDINGS, opencv_reference
from opencv_dataflow.program_documentation import render_program_documentation


def test_opencv_reference_maps_registered_native_tools(monkeypatch):
    registry = builtin_registry()
    assert BINDINGS.keys() <= registry.types.keys()
    reference = opencv_reference(registry.create("cv.morphology"))
    assert "cv2.morphologyEx" in reference and "cv2.getStructuringElement" in reference
    assert f"OpenCV {cv2.__version__} Python reference" in reference
    assert opencv_reference(registry.create("input.demo")) == ""

    class PluginBlock(Block):
        type_id = "cv.resize"

    assert opencv_reference(PluginBlock()) == ""
    monkeypatch.delattr(cv2, "GaussianBlur")
    assert opencv_reference(registry.create("cv.gaussian_blur")) == ""


def test_reference_opens_for_undocumented_block_without_changing_notes(app):
    block = builtin_registry().create("cv.canny")
    dialog = DocumentationEditor(block)
    try:
        assert dialog.tabs.tabText(dialog.tabs.currentIndex()) == "OpenCV reference"
        reference = dialog.findChild(QTextBrowser).toPlainText()
        assert "Canny(image," in reference and "Canny(dx, dy," in reference
        assert "apertureSize" in reference
        assert dialog.documentation() == block.documentation == ""
        assert "documentation" not in block.to_project_item()
    finally:
        dialog.deleteLater()


def test_copy_reference_appends_notes_and_round_trips(app, tmp_path):
    registry = builtin_registry()
    graph = Graph()
    block = graph.add(registry.create("cv.gaussian_blur"))
    block.documentation = "<p><strong>Retain the original settings.</strong></p>"
    dialog = DocumentationEditor(block)
    try:
        assert dialog.tabs.tabText(dialog.tabs.currentIndex()) == "Block notes"
        assert dialog.documentation() == block.documentation
        dialog.tabs.setCurrentIndex(1)
        button = dialog.tabs.currentWidget().findChild(QPushButton)
        assert button.text() == "Copy reference to notes"
        button.click()
        assert dialog.tabs.currentIndex() == 0
        assert dialog.editor.toPlainText().startswith("Retain the original settings.")
        assert "GaussianBlur(src, ksize, sigmaX" in dialog.editor.toPlainText()
        assert dialog.documentation() != block.documentation
        block.documentation = dialog.documentation()

        path = tmp_path / "reference.json"
        graph.save(path)
        restored = Graph.load(path, registry)
        assert restored.blocks[block.id].documentation == block.documentation
        guide = render_program_documentation(restored, "Blur test")
        assert "<strong>Retain the original settings.</strong>" in guide
        assert "GaussianBlur(src, ksize, sigmaX" in guide
    finally:
        dialog.deleteLater()
