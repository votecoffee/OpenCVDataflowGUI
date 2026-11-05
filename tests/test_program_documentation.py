"""Program guide export, layout and safe rendering of stored notes."""

import copy

import pytest
from PySide6.QtWidgets import QFileDialog

from opencv_dataflow.app import MainWindow
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.program_documentation import (
    export_program_documentation,
    render_program_documentation,
)
from opencv_dataflow.toolblocks import Binding, ToolBlock
from opencv_dataflow.workspace import add_group


def test_guide_contains_linked_layout_connections_and_nested_block_notes():
    registry = builtin_registry()
    graph, inner = Graph(), Graph()
    source = graph.add(registry.create("input.demo", block_id="duplicate"))
    source.name = "Source <fixture>"
    source.description = "A & B\nSecond line."
    source.position = [-750, 150]
    child = inner.add(registry.create("cv.canny", block_id="duplicate", parameters={"low": 55.0}))
    child.documentation = (
        "<html><head><style>discard this stylesheet</style></head><body>"
        '<p onclick="discard()"><span style="font-weight:700; font-style:italic; '
        'text-decoration:underline">Tuned &amp; tested</span></p>'
        "<ul><li>Keep the calibrated thresholds.</li></ul>"
        '<img src="https://example.invalid/remote.png">'
        "<script>discard this script</script></body></html>"
    )
    tool = graph.add(
        ToolBlock(
            inner, {"camera": Binding(child.id, "image")}, {"edges": Binding(child.id, "image")}
        )
    )
    tool.name = "Detect edges"
    graph.connect(source.id, "image", tool.id, "camera")
    add_group(graph, [source.id, tool.id], "Acquisition & inspection")
    before = copy.deepcopy(graph.to_dict())

    document = render_program_documentation(graph, "Job <A>")

    assert graph.to_dict() == before
    assert "<h1>Job &lt;A&gt;</h1>" in document
    assert "A &amp; B\nSecond line." in document
    assert "Acquisition &amp; inspection" in document
    assert document.count('aria-label="Simplified block layout"') == 2
    assert 'href="#job-1"' in document and 'id="job-1"' in document
    assert 'href="#job-2-1"' in document and 'id="job-2-1"' in document
    assert "Source &lt;fixture&gt;.image" in document and "Detect edges.camera" in document
    assert "Public terminals" in document and "Canny edges.image" in document
    assert "<code>55.0</code>" in document
    assert "<strong><em><u>Tuned &amp; tested</u></em></strong>" in document
    assert "<ul><li>Keep the calibrated thresholds.</li></ul>" in document
    assert all(text not in document for text in ("onclick", "discard this", "example.invalid"))


def test_invalid_export_keeps_existing_document(tmp_path):
    graph = Graph()
    block = graph.add(builtin_registry().create("cv.canny"))
    block.parameters["low"] = float("nan")
    target = tmp_path / "guide.html"
    target.write_text("Previous guide", encoding="utf-8")
    with pytest.raises(ValueError):
        export_program_documentation(graph, target)
    assert target.read_text(encoding="utf-8") == "Previous guide"


def test_parallel_inspection_documents_its_template():
    registry = builtin_registry()
    graph = Graph()
    parallel = graph.add(registry.create("flow.inspect_each"))
    child = next(iter(parallel.inspection.graph.blocks.values()))
    child.description = "This note belongs to the per-part template."

    document = render_program_documentation(graph, "Parallel inspections")

    assert document.count('aria-label="Simplified block layout"') == 2
    assert "This note belongs to the per-part template." in document
    assert "Public terminals" in document
    assert 'id="job-1-1"' in document


def test_file_menu_exports_current_program_documentation(app, monkeypatch, tmp_path):
    window = MainWindow()
    block = next(iter(window.graph.blocks.values()))
    block.description = "Applied before export"
    window.mark_dirty()
    target = tmp_path / "guide"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_: (str(target), ""))
    file_menu = window.menuBar().actions()[0].menu()
    action = next(a for a in file_menu.actions() if a.text() == "Export program documentation…")
    try:
        action.trigger()
        document = target.with_suffix(".html").read_text(encoding="utf-8")
        assert "Applied before export" in document
        assert "Untitled program" in document
        assert window.dirty and window.path is None
    finally:
        window.dirty = False
        window.close()
