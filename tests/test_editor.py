"""Offscreen Qt interaction checks: ports, inspector, and worker lifecycle."""

import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QFont, QHelpEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QInputDialog,
    QMessageBox,
    QToolTip,
)

from opencv_dataflow.app import MainWindow
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.canvas import CanvasScene, CanvasView
from opencv_dataflow.core import Block, Graph, Port, Result
from opencv_dataflow.documentation_editor import DocumentationEditor
from opencv_dataflow.images import ImageArray
from opencv_dataflow.kernel_editor import KernelEditor
from opencv_dataflow.project_compare import ProjectCompare
from opencv_dataflow.project_xml import read_project_xml
from opencv_dataflow.toolblock_editor import InterfaceDialog
from opencv_dataflow.toolblocks import encapsulate
from opencv_dataflow.visual_style import BLOCK_STATE_COLORS


def pump_until(app, predicate, seconds=8):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        QTest.qWait(10)
    raise AssertionError("Qt condition timed out")


def test_live_results_debounce_use_captured_input_and_leave_configuration_untouched(
    app, monkeypatch
):
    window = MainWindow()
    window.show()
    window.graph.run()
    block = list(window.graph.blocks.values())[-1]
    window.scene.nodes[block.id].setSelected(True)
    inspector = window.inspector
    inspector.live_delay_ms = 75
    captured = block.last_inputs["image"].copy()
    original_count = block.last_result.outputs["count"]
    original_parameters = dict(block.parameters)
    calls = []
    original = type(block).process

    def process(candidate, inputs):
        calls.append((candidate.parameters["expected_count"], inputs["image"].copy()))
        return original(candidate, inputs)

    monkeypatch.setattr(type(block), "process", process)
    try:
        assert inspector.live_results.isEnabled()
        assert not inspector.live_results.isChecked()
        inspector.fields["expected_count"].setText("4")
        QTest.qWait(150)
        assert not calls
        inspector.live_results.setChecked(True)
        for value in ("5", "6", "7"):
            inspector.fields["expected_count"].setText(value)
        assert not calls
        pump_until(app, lambda: inspector._live_result is not None)
        assert [count for count, _ in calls] == [7]
        assert (calls[0][1] == captured).all()
        assert "BLOCK PREVIEW" in inspector.details.toPlainText()
        assert not inspector._live_result.last_result.passed
        assert block.parameters == original_parameters
        assert block.last_result.outputs["count"] == original_count
        assert not window.dirty
        inspector.fields["expected_count"].setText("8")
        window.scene.clearSelection()
        assert not inspector.live_results.isChecked()
        QTest.qWait(150)
        assert len(calls) == 1
        window.scene.nodes[block.id].setSelected(True)
        assert not inspector.live_results.isChecked()
        inspector.live_results.setChecked(True)
        inspector.fields["expected_count"].setText("9")
        inspector.hide()
        QTest.qWait(150)
        assert not inspector.live_results.isChecked()
        assert len(calls) == 1
    finally:
        window.close()


def test_run_block_works_without_live_results_and_consumes_pending_live_edit(app, monkeypatch):
    window = MainWindow()
    window.show()
    block = list(window.graph.blocks.values())[-1]
    window.scene.nodes[block.id].setSelected(True)
    inspector = window.inspector
    assert not inspector.run_block.isEnabled()
    window.graph.run()
    inspector.refresh_result()
    saved_result = block.last_result
    calls = []
    original = type(block).process

    def process(candidate, inputs):
        calls.append(candidate.parameters["expected_count"])
        np.testing.assert_array_equal(inputs["image"], block.last_inputs["image"])
        return original(candidate, inputs)

    monkeypatch.setattr(type(block), "process", process)
    try:
        assert inspector.run_block.isEnabled()
        assert not inspector.live_results.isChecked()
        inspector.fields["expected_count"].setText("4")
        inspector.run_block.click()
        assert not inspector.run_block.isEnabled()
        pump_until(app, lambda: inspector.run_block.isEnabled())
        assert calls == [4]
        assert inspector._live_result.state == "fail"
        assert "BLOCK PREVIEW" in inspector.details.toPlainText()
        assert not inspector.live_results.isChecked()
        assert block.last_result is saved_result
        assert not window.dirty

        inspector.live_delay_ms = 150
        inspector.live_results.setChecked(True)
        inspector.fields["expected_count"].setText("5")
        assert inspector.live_timer.isActive()
        inspector.run_block.click()
        assert not inspector.live_timer.isActive()
        pump_until(app, lambda: inspector.run_block.isEnabled())
        QTest.qWait(200)
        assert calls == [4, 5]

        inspector.live_results.setChecked(False)
        inspector.run_block.click()
        window.scene.clearSelection()
        pump_until(app, lambda: inspector._live_worker is None)
        assert inspector._live_result is None
        assert not inspector.run_block.isEnabled()
    finally:
        window.close()


def test_run_block_accepts_captured_text_input(app):
    window = MainWindow()
    graph = Graph()
    verify = graph.add(window.registry.create("inspect.verify_text"))
    window.replace_graph(graph)
    verify.execute({"text": "part-42"})
    window.scene.nodes[verify.id].setSelected(True)
    inspector = window.inspector
    try:
        assert inspector.run_block.isEnabled()
        inspector.fields["minimum_length"].setText("10")
        inspector.run_block.click()
        pump_until(app, lambda: inspector._live_result is not None)
        assert inspector._live_result.state == "fail"
        assert verify.state == "pass"
        assert verify.parameters["minimum_length"] == 1
    finally:
        window.close()


def test_invalid_parameter_draft_marks_field_and_node_and_pauses_live_preview(app, monkeypatch):
    window = MainWindow()
    window.show()
    graph = Graph()
    source = graph.add(window.registry.create("input.demo"))
    blur = graph.add(window.registry.create("cv.gaussian_blur"))
    graph.connect(source.id, "image", blur.id, "image")
    window.replace_graph(graph)
    graph.run()
    node = window.scene.nodes[blur.id]
    node.setSelected(True)
    inspector = window.inspector
    inspector.live_results.setChecked(True)
    inspector.live_delay_ms = 40
    field = inspector.fields["kernel"]
    warnings = []
    monkeypatch.setattr(window, "warn", warnings.append)
    try:
        field.setText("4097")
        assert not inspector.apply_button.isEnabled()
        assert not inspector.run_block.isEnabled()
        assert not inspector.apply()
        assert warnings == []
        assert "between 1 and 4095" in field.toolTip()
        assert "Invalid configuration" in inspector.validation_label.text()
        assert "Invalid configuration" in node.toolTip()
        assert [badge[0] for badge in node.issue_badges()] == ["!"]
        assert not inspector.live_timer.isActive()
        assert blur.parameters["kernel"] == 5
        assert not window.dirty

        field.setText("7")
        assert inspector.apply_button.isEnabled() and inspector.run_block.isEnabled()
        assert inspector.validation_label.isHidden()
        assert not node.issue_badges()
        pump_until(app, lambda: inspector._live_result is not None)
        assert inspector._live_result.parameters["kernel"] == 7
        assert blur.parameters["kernel"] == 5
    finally:
        window.close()


def test_cross_field_parameter_errors_highlight_both_controls(app):
    window = MainWindow()
    graph = Graph()
    canny = graph.add(window.registry.create("cv.canny"))
    window.replace_graph(graph)
    window.scene.nodes[canny.id].setSelected(True)
    inspector = window.inspector
    try:
        inspector.fields["low"].setText("200")
        assert "low <= high" in inspector.fields["low"].toolTip()
        assert "low <= high" in inspector.fields["high"].toolTip()
        assert not inspector.apply_button.isEnabled()
        inspector.fields["high"].setText("250")
        assert inspector.apply()
        assert canny.parameters["low"] == 200.0 and canny.parameters["high"] == 250.0
    finally:
        window.dirty = False
        window.close()


def test_custom_invert_example_shows_validation_only_for_invalid_drafts(app):
    from opencv_dataflow.plugins import load_plugin

    folder = Path("examples/custom_invert")
    window = MainWindow()
    load_plugin(folder / "custom_invert.py", window.registry)
    graph = Graph.load(folder / "custom_invert.json", window.registry)
    window.replace_graph(graph)
    graph.run()
    source, invert = graph.blocks["source"], graph.blocks["invert"]
    assert invert.state == "pass", invert.error
    original = source.last_result.outputs["image"]
    actual = invert.last_result.outputs["image"]
    left = original.shape[1] * 20 // 100
    right = (original.shape[1] * 80 + 99) // 100
    np.testing.assert_array_equal(actual[:, :left], original[:, :left])
    np.testing.assert_array_equal(actual[:, left:right], cv2.bitwise_not(original[:, left:right]))
    np.testing.assert_array_equal(actual[:, right:], original[:, right:])

    window.scene.nodes[invert.id].setSelected(True)
    inspector = window.inspector
    start, end = inspector.fields["start_percent"], inspector.fields["end_percent"]
    try:
        assert inspector.validation_label.isHidden()
        assert "between 0 and 100" in start.toolTip()
        start.setText("-1")
        assert not inspector.validation_label.isHidden()
        assert start.toolTip().count("between 0 and 100") == 1
        assert not inspector.apply_button.isEnabled()
        start.setText("90")
        assert "must be less than" in start.toolTip() and "must be less than" in end.toolTip()
        assert not inspector.run_block.isEnabled()
        start.setText("30")
        assert inspector.validation_label.isHidden()
        assert "between 0 and 100" in end.toolTip()
        assert inspector.run_block.isEnabled()
        inspector.run_block.click()
        pump_until(app, lambda: inspector._live_result is not None)
        assert inspector._live_result.parameters["start_percent"] == 30
        assert invert.parameters["start_percent"] == 20
        invalid = window.registry.create("example.invert", parameters={"start_percent": 100})
        assert invalid.execute({"image": original}) is None
        assert invalid.error_kind == "invalid"
        assert "Start percent must be less than end percent" in invalid.error
    finally:
        window.close()


def test_morphology_kernel_editor_stages_a_custom_grid_for_live_preview(app, monkeypatch):
    window = MainWindow()
    graph = Graph()
    block = graph.add(window.registry.create("cv.morphology"))
    window.replace_graph(graph)
    window.show()
    block.execute({"image": np.zeros((12, 16), np.uint8)})
    window.scene.nodes[block.id].setSelected(True)
    inspector = window.inspector
    inspector.live_results.setChecked(True)
    inspector.live_delay_ms = 50

    def edit(dialog):
        dialog.show()
        dialog.shape.setCurrentIndex(dialog.shape.findData("diamond"))
        dialog.width.setValue(5)
        dialog.height.setValue(5)
        assert dialog.grid.cells[1].tolist() == [0, 1, 1, 1, 0]
        dialog.shape.setCurrentIndex(dialog.shape.findData("custom"))
        app.processEvents()
        cell = dialog.grid.index(0, 0)
        QTest.mouseClick(
            dialog.table.viewport(),
            Qt.MouseButton.LeftButton,
            pos=dialog.table.visualRect(cell).center(),
        )
        assert dialog.parameters["custom_kernel"][0] == [1, 0, 1, 0, 0]
        return QDialog.DialogCode.Accepted

    try:
        monkeypatch.setattr(KernelEditor, "exec", edit)
        inspector.edit_kernel()
        pump_until(app, lambda: inspector._live_result is not None)
        assert inspector._live_result.state == "pass"
        assert inspector._live_result.parameters["custom_kernel"][0] == [1, 0, 1, 0, 0]
        assert block.parameters["shape"] == "ellipse"
        assert not window.dirty
        assert inspector.apply()
        assert block.parameters["shape"] == "custom"
        assert block.parameters["custom_kernel"][0] == [1, 0, 1, 0, 0]
    finally:
        window.dirty = False
        window.close()


@pytest.mark.parametrize("reverse", [False, True])
def test_drag_ports_both_directions(app, reverse):
    registry = builtin_registry()
    graph = Graph()
    a = graph.add(registry.create("input.demo"))
    b = graph.add(registry.create("gapi.gray"))
    b.position = [350, 0]
    scene = CanvasScene(graph)
    view = CanvasView(scene)
    view.resize(1000, 500)
    view.show()
    view.fit_graph()
    app.processEvents()
    start = view.mapFromScene(scene.nodes[a.id].sockets[True, "image"].scenePos())
    end = view.mapFromScene(scene.nodes[b.id].sockets[False, "image"].scenePos())
    if reverse:
        start, end = end, start
    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(view.viewport(), (start + end) / 2)
    QTest.mouseMove(view.viewport(), end)
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=end)
    assert len(graph.connections) == 1
    assert graph.connections[0].source == a.id
    assert graph.connections[0].target == b.id
    view.close()


def test_result_image_popouts_follow_ticket_state_and_block_previews(app):
    window = MainWindow()
    window.show()
    block = list(window.graph.blocks.values())[-1]
    window.scene.nodes[block.id].setSelected(True)
    inspector = window.inspector

    def run_once():
        window.start_run(False)
        pump_until(app, lambda: window.runner is None)

    def check(viewer, ticket, state):
        assert f"Ticket {ticket} · {state.upper()}" in viewer.windowTitle()
        assert BLOCK_STATE_COLORS[state] in viewer.frame.styleSheet()

    try:
        run_once()
        assert inspector.pop_out_image.isEnabled()
        inspector.preview_choice.setCurrentText("Input / image")
        assert not inspector.pop_out_image.isEnabled()
        inspector.preview_choice.setCurrentText("Output / overlay")
        inspector.pop_out_image.click()
        output = next(iter(inspector.image_popouts.values()))
        assert not output.image.pixmap().isNull()
        check(output, 1, "pass")
        inspector.pop_out_image.click()
        assert len(inspector.image_popouts) == 1
        inspector.preview_choice.setCurrentText("Graphic / contours")
        inspector.pop_out_image.click()
        graphic = list(inspector.image_popouts.values())[-1]
        check(graphic, 1, "pass")

        inspector.fields["expected_count"].setText("3")
        inspector.run_block.click()
        pump_until(app, lambda: inspector._live_result is not None)
        check(output, 1, "fail")
        check(graphic, 1, "fail")
        window.scene.clearSelection()
        check(output, 1, "pass")
        check(graphic, 1, "pass")

        block.parameters["expected_count"] = 3
        run_once()
        check(output, 2, "fail")
        check(graphic, 2, "fail")
        block.parameters["min_area"] = -1.0
        run_once()
        check(output, 3, "error")
        assert output.image.text() == "No image for this result"
        assert output.image.pixmap().isNull()

        output.close()
        assert len(inspector.image_popouts) == 1
        window.replace_graph(Graph())
        assert not inspector.image_popouts
    finally:
        window.close()


def test_result_image_popout_stays_pinned_inside_a_toolblock(app):
    window = MainWindow()
    window.show()
    graph = window.graph
    ids = list(graph.blocks)
    owner = encapsulate(graph, ids[-2:], "Inspection", window.registry)
    graph.run()
    window.enter_toolblock(owner)
    window.scene.nodes[ids[-1]].setSelected(True)
    inspector = window.inspector
    try:
        inspector.preview_choice.setCurrentText("Output / overlay")
        inspector.pop_out_image.click()
        viewer = next(iter(inspector.image_popouts.values()))
        assert "Ticket 1" in viewer.windowTitle()
        window.leave_toolblock()
        window.start_run(False)
        pump_until(app, lambda: window.runner is None)
        assert "Ticket 2" in viewer.windowTitle()
        window.scene.nodes[owner.id].setSelected(True)
        window.scene.delete_selected()
        assert not inspector.image_popouts
    finally:
        window.dirty = False
        window.close()


def test_run_inspector_delete_and_continuous_stop(app):
    window = MainWindow()
    window.show()
    app.processEvents()
    last = list(window.graph.blocks.values())[-1]
    window.scene.nodes[last.id].setSelected(True)
    assert window.inspector.block is last
    window.start_run(False)
    assert window.scene.locked
    pump_until(app, lambda: window.runner is None)
    assert all(b.state == "pass" for b in window.graph.blocks.values())
    assert "2 contours" in window.inspector.details.toPlainText()
    assert window.inspector.preview_choice.currentText() == "Graphic / contours"
    assert window.inspector.preview_choice.count() == 4  # input, two outputs, one graphic
    assert not window.inspector.image.pixmap().isNull()
    # Applying exposure and thumbnail choices must update the node and persist.
    window.inspector.thumb.setChecked(False)
    window.inspector.fields["expected_count"].setText("3")
    window.inspector.apply()
    assert last.parameters["expected_count"] == 3
    assert not last.thumbnail
    assert window.dirty
    window.start_run(True)
    pump_until(app, lambda: last.state == "fail")
    window.stop_run()
    pump_until(app, lambda: window.runner is None)
    assert last.last_result.score == 1 - 1 / 3
    assert not window.scene.locked
    wire = window.scene.wires[-1]
    window.scene.clearSelection()
    wire.setSelected(True)
    window.scene.delete_selected()
    assert len(window.graph.connections) == 3
    assert last.state == "idle"
    window.dirty = False
    window.close()


def test_block_description_appears_as_plain_text_and_survives_project_save(app, tmp_path):
    window = MainWindow()
    window.show()
    block = next(iter(window.graph.blocks.values()))
    window.scene.nodes[block.id].setSelected(True)
    description = "Locate <reference> & verify contrast.\nCheck the upper-left corner."
    try:
        assert window.inspector.description_field.toPlainText() == ""
        assert "description" not in window.graph.to_dict()["blocks"][0]
        window.inspector.description_field.setPlainText(description)
        assert window.inspector.apply()
        tooltip = window.scene.nodes[block.id].toolTip()
        assert "Locate &lt;reference&gt; &amp; verify contrast.<br>Check" in tooltip
        assert "Border: tool category" in tooltip
        window.path = tmp_path / "description.json"
        assert window.save_project()

        graph = Graph.load(window.path, window.registry)
        assert graph.blocks[block.id].description == description
        window.replace_graph(graph, window.path)
        window.scene.nodes[block.id].setSelected(True)
        assert window.inspector.description_field.toPlainText() == description
        assert window.scene.nodes[block.id].toolTip() == tooltip
    finally:
        window.dirty = False
        window.close()


def test_block_documentation_keeps_formatting_and_does_not_invalidate_results(
    app, monkeypatch, tmp_path
):
    window = MainWindow()
    window.show()
    window.graph.run()
    block = next(iter(window.graph.blocks.values()))
    window.scene.nodes[block.id].setSelected(True)
    result = block.last_result
    tooltip = window.scene.nodes[block.id].toolTip()
    assert not block.documentation

    def edit(dialog):
        dialog.editor.insertPlainText("Inspect the reference surface.")
        dialog.editor.selectAll()
        dialog.bold.trigger()
        return QDialog.DialogCode.Accepted

    def cancel(dialog):
        dialog.editor.insertPlainText("Discard this edit.")
        return QDialog.DialogCode.Rejected

    try:
        monkeypatch.setattr(DocumentationEditor, "exec", edit)
        window.inspector.edit_documentation()
        assert window.dirty
        assert block.last_result is result
        assert window.scene.nodes[block.id].toolTip() == tooltip
        saved = block.documentation

        monkeypatch.setattr(DocumentationEditor, "exec", cancel)
        window.inspector.edit_documentation()
        assert block.documentation == saved

        window.path = tmp_path / "documentation.json"
        assert window.save_project()
        graph = Graph.load(window.path, window.registry)
        restored = graph.blocks[block.id]
        assert restored.documentation == saved
        dialog = DocumentationEditor(restored, window)
        assert dialog.editor.toPlainText() == "Inspect the reference surface."
        first_fragment = dialog.editor.document().begin().begin().fragment()
        assert first_fragment.charFormat().fontWeight() == QFont.Weight.Bold
        dialog.deleteLater()
    finally:
        window.dirty = False
        window.close()


def test_file_menu_exports_current_project_and_compares_two_snapshots(app, monkeypatch, tmp_path):
    window = MainWindow()
    window.show()
    block = list(window.graph.blocks.values())[-1]
    before, after = tmp_path / "before.xml", tmp_path / "after.xml"
    actions = {action.text(): action for action in window.menuBar().actions()[0].menu().actions()}
    selected = iter((before, after))
    monkeypatch.setattr(QFileDialog, "exec", lambda _: QDialog.DialogCode.Accepted)
    monkeypatch.setattr(QFileDialog, "selectedFiles", lambda _: [str(next(selected))])
    compared = []

    def compare(dialog):
        dialog.before_path.setText(str(before))
        dialog.after_path.setText(str(after))
        dialog.compare()
        compared.append(
            (
                dialog.model.changes,
                dialog.before_value.toPlainText(),
                dialog.after_value.toPlainText(),
            )
        )
        return QDialog.DialogCode.Accepted

    try:
        actions["Export project XML…"].trigger()
        block.parameters["expected_count"] = 3
        window.mark_dirty()
        actions["Export project XML…"].trigger()
        assert window.path is None and window.dirty
        assert read_project_xml(after)["blocks"][-1]["parameters"]["expected_count"] == 3
        monkeypatch.setattr(ProjectCompare, "exec", compare)
        actions["Compare project XML…"].trigger()
        [(changes, old, new)] = compared
        assert len(changes) == 1
        assert changes[0].path == f"/blocks/{block.id}/parameters/expected_count"
        assert (old, new) == ("2", "3")
    finally:
        window.dirty = False
        window.close()


def test_inspector_block_xml_asks_before_disconnecting_wires(app, monkeypatch, tmp_path):
    window = MainWindow()
    graph = Graph()
    source = graph.add(window.registry.create("input.demo"))
    target = graph.add(window.registry.create("cv.morphology"))
    preset = graph.add(window.registry.create("cv.morphology", parameters={"operation": "close"}))
    preset.exposed_inputs.remove("image")
    graph.connect(source.id, "image", target.id, "image")
    window.replace_graph(graph)
    window.show()
    path = tmp_path / "saved.cv.morphology.xml"
    errors, prompts = [], []
    replies = iter((QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes))

    def confirm(dialog):
        prompts.append(dialog.detailedText())
        assert dialog.defaultButton() == dialog.button(QMessageBox.StandardButton.No)
        return next(replies)

    monkeypatch.setattr(window, "warn", errors.append)
    monkeypatch.setattr(QFileDialog, "exec", lambda _: QDialog.DialogCode.Accepted)
    monkeypatch.setattr(QFileDialog, "selectedFiles", lambda _: [str(path)])
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(path), ""))
    monkeypatch.setattr(QMessageBox, "exec", confirm)
    try:
        window.scene.nodes[preset.id].setSelected(True)
        window.inspector.save_selected_block_xml()
        assert path.is_file() and not window.dirty
        window.scene.clearSelection()
        window.scene.nodes[target.id].setSelected(True)
        window.inspector.load_selected_block_xml()
        assert graph.blocks[target.id] is target
        assert len(graph.connections) == 1 and not window.dirty
        window.inspector.load_selected_block_xml()
        assert graph.blocks[target.id].parameters["operation"] == "close"
        assert not graph.connections and window.dirty
        assert prompts == [f"{source.name}.image → {target.name}.image"] * 2
        assert not errors
    finally:
        window.dirty = False
        window.close()


def test_socket_tooltips_show_full_names_and_latest_scalar_values(app):
    name = "a_long_input_name_that_is_truncated_on_the_node"

    class Echo(Block):
        inputs = (Port(name, "any"),)
        outputs = (Port("echoed_value", "any"),)

        def process(self, inputs):
            return Result({"echoed_value": inputs[name]})

    graph = Graph()
    block = graph.add(Echo())
    scene = CanvasScene(graph)
    view = CanvasView(scene)
    view.resize(800, 500)
    view.show()
    view.fit_graph()
    app.processEvents()
    node = scene.nodes[block.id]
    sockets = node.sockets

    def hover(output, port_name):
        socket = sockets[output, port_name]
        label_x = node.width - 30 if output else 30
        for point in (socket.pos(), QPointF(label_x, socket.y())):
            tooltip = canvas_tooltip(app, view, node.mapToScene(point))
            assert port_name in tooltip
            assert ("output" if output else "input") in tooltip
        return tooltip

    try:
        assert "Value:" not in hover(False, name)
        assert "Value:" not in hover(True, "echoed_value")
        for value, text in ((np.float32(2.5), "2.5"), (False, "False"), ("<x>", "&lt;x&gt;")):
            block.execute({name: value})
            assert block.state == "pass", block.error
            assert text in hover(False, name)
            assert text in hover(True, "echoed_value")
        block.execute({name: np.zeros((4, 4), np.uint8)})
        assert "Value:" not in hover(False, name)
        assert "Value:" not in hover(True, "echoed_value")
    finally:
        QToolTip.hideText()
        view.close()


def canvas_tooltip(app, view, scene_point):
    viewport = view.viewport()
    point = view.mapFromScene(scene_point)
    event = QHelpEvent(QEvent.Type.ToolTip, point, viewport.mapToGlobal(point))
    app.sendEvent(viewport, event)
    return QToolTip.text()


def test_block_tooltip_reads_current_description_and_issue(app):
    graph = Graph()
    block = graph.add(builtin_registry().create("cv.threshold"))
    block.description = "Original description"
    scene = CanvasScene(graph)
    node = scene.nodes[block.id]
    view = CanvasView(scene)
    view.resize(800, 500)
    view.show()
    view.fit_graph()
    app.processEvents()
    header = node.mapToScene(QPointF(30, 18))
    try:
        assert "Original description" in canvas_tooltip(app, view, header)
        block.name = "New block name"
        block.description = "New <description>\nSecond line"
        block.state, block.error_kind, block.error = "error", "exception", "Run failed"
        tooltip = canvas_tooltip(app, view, header)
        assert "New block name" in tooltip
        assert "New &lt;description&gt;<br>Second line" in tooltip
        assert "Run failed" in tooltip
        block.state = "pass"
        assert "Run failed" not in canvas_tooltip(app, view, header)
    finally:
        QToolTip.hideText()
        view.close()


def test_canvas_node_movement_updates_wires(app):
    window = MainWindow()
    node = next(iter(window.scene.nodes.values()))
    old = window.scene.wires[0].path().pointAtPercent(0)
    node.setPos(node.pos() + QPoint(40, 30))
    new = window.scene.wires[0].path().pointAtPercent(0)
    assert new.x() - old.x() == 40
    assert new.y() - old.y() == 30
    assert node.block.position == [40.0, 30.0]
    window.dirty = False
    window.close()


def test_toolblock_group_navigation_run_and_save(app, monkeypatch, tmp_path):
    window = MainWindow()
    window.show()
    app.processEvents()
    original = list(window.graph.blocks.values())
    for b in original[1:]:
        window.scene.nodes[b.id].setSelected(True)
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Inspect", True))
    window.create_toolblock()
    block = window.inspector.block
    assert block.name == "Inspect"
    assert len(window.graph.blocks) == 2
    assert window.toolblock_templates[-1]["name"] == "Inspect"
    # Enter by double-clicking the actual node header.
    center = window.view.mapFromScene(window.scene.nodes[block.id].scenePos() + QPoint(80, 20))
    QTest.mouseDClick(window.view.viewport(), Qt.MouseButton.LeftButton, pos=center)
    app.processEvents()
    assert window.graph is block.graph
    assert "Inspect" in window.breadcrumb.text()
    inner_nodes = list(block.graph.blocks.values())
    inner = encapsulate(
        window.graph,
        [inner_nodes[0].id, inner_nodes[1].id],
        "Preprocess",
        window.registry,
        owner=block,
    )
    assert next(iter(block.input_bindings.values())).block == inner.id
    window.scene.rebuild(inner.id)
    window.enter_toolblock(inner)
    assert len(window.graph_stack) == 2
    window.start_run(False)  # Still runs the complete job, including its image source.
    pump_until(app, lambda: window.runner is None)
    assert block.state == "pass", block.error
    assert all(b.state == "pass" for b in window.graph.blocks.values())
    assert window.back_action.isEnabled()
    path = tmp_path / "job.json"
    window.path = path
    assert window.save_project()
    loaded = Graph.load(path, window.registry)
    assert len(loaded.blocks) == 2  # Save targets the root while editing nested graphs.
    window.leave_toolblock()
    assert window.graph is block.graph
    window.leave_toolblock()
    assert window.graph is window.root_graph and not window.graph_stack
    assert window.inspector.block is block
    window.dirty = False
    window.close()


def test_toolblock_interface_dialog_and_library_instances(app):
    window = MainWindow()
    original = list(window.graph.blocks.values())
    block = encapsulate(window.graph, [b.id for b in original[1:]], "Inspect", window.registry)
    dialog = InterfaceDialog(block, window.graph)
    assert dialog.table.rowCount() == len(block.inputs) + len(block.outputs)
    for row, binding in enumerate(
        [*block.input_bindings.values(), *block.output_bindings.values()]
    ):
        assert dialog.table.cellWidget(row, 2).currentData() == binding
    dialog.table.item(0, 1).setText("image")
    dialog.apply()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert window.graph.connections[0].input == "image"
    window.toolblock_templates.append(block.definition())
    window.populate_library()
    item = window.library.findItems(
        "Inspect", Qt.MatchFlag.MatchExactly | Qt.MatchFlag.MatchRecursive
    )[0]
    window.add_block(item)
    first = window.inspector.block
    window.add_block(item)
    second = window.inspector.block
    assert first.id != second.id
    assert first.graph is not second.graph
    first.graph.blocks[original[-1].id].parameters["expected_count"] = 9
    assert second.graph.blocks[original[-1].id].parameters["expected_count"] == 2
    window.dirty = False
    window.close()


@pytest.mark.parametrize(
    "folder,count,shape_id",
    [
        ("regions", 10, "roi_circle"),
        ("combined_regions", 11, "combined_outer"),
        ("transformed_regions", 12, "tracked_shape"),
    ],
)
def test_regions_example_editor_load_save_and_shape_dropdown(
    app, load_example, tmp_path, folder, count, shape_id
):
    from PySide6.QtWidgets import QComboBox

    window = MainWindow()
    window.show()
    app.processEvents()

    load_example(window, folder)
    assert len(window.graph.blocks) == count
    circle = window.graph.blocks[shape_id]
    window.scene.nodes[circle.id].setSelected(True)
    assert isinstance(window.inspector.fields["shape"], QComboBox)
    assert window.inspector.fields["shape"].currentText() == "circle"
    window.path = tmp_path / "regions.json"
    assert window.save_project()
    restored = Graph.load(window.path, window.registry)
    assert restored.to_dict() == window.graph.to_dict()
    window.start_run(False)
    pump_until(app, lambda: window.runner is None)
    assert all(b.state == "pass" for b in window.graph.blocks.values())
    window.dirty = False
    window.close()
    app.processEvents()


def test_channel_pyramid_editor_preview_level_selection_and_save(app, load_example, tmp_path):
    window = MainWindow()
    load_example(window, "channels_pyramids")
    window.graph.run()
    window.scene.nodes["pyr_set"].setSelected(True)
    window.inspector.preview_choice.setCurrentText("Output / images")
    assert not window.inspector.image.pixmap().isNull()
    window.scene.clearSelection()
    window.scene.nodes["pyr_threshold"].setSelected(True)
    assert window.inspector.level_fields["image"].currentText() == "2"
    window.inspector.level_fields["image"].setCurrentText("1")
    window.inspector.apply()
    assert window.graph.blocks["pyr_threshold"].input_levels == {"image": "1"}
    window.path = tmp_path / "pyramids.json"
    assert window.save_project()
    loaded = Graph.load(window.path, window.registry)
    loaded.run()
    assert loaded.blocks["pyr_threshold"].last_output_images["image"].shape == (110, 190)
    assert loaded.blocks["pyr_components"].last_result.outputs["count"] == 2
    window.start_run(False)
    pump_until(app, lambda: window.runner is None)
    assert all(block.state == "pass" for block in window.graph.blocks.values())
    assert (
        window.graph.blocks["pyr_components"].last_result.payloads["data"]["pyramid"]["level"] == 1
    )
    window.dirty = False
    window.close()


def test_image_file_dropdowns_metadata_and_preview(app, tmp_path):
    import cv2
    import numpy as np

    window = MainWindow()
    window.show()
    path = tmp_path / "red16.png"
    image = np.zeros((20, 30, 3), np.uint16)
    image[..., 2] = 65535
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(encoded.tobytes())
    block = window.graph.add(window.registry.create("input.file"))
    window.scene.rebuild(block.id)
    window.inspector.fields["path"].setText(str(path))
    window.inspector.fields["color_format"].setCurrentText("rgb")
    window.inspector.fields["pixel_depth"].setCurrentText("uint16")
    window.inspector.apply()
    assert block.parameters["color_format"] == "rgb"
    window.start_run(False)
    pump_until(app, lambda: window.runner is None)
    assert block.state == "pass", block.error
    assert isinstance(block.last_output_images["image"], ImageArray)
    assert block.last_output_images["image"].color_format == "rgb"
    window.inspector.preview_choice.setCurrentText("Output / image")
    shown = window.inspector.image.pixmap().toImage()
    color = shown.pixelColor(shown.width() // 2, shown.height() // 2)
    assert color.red() == 255 and color.blue() == 0
    assert '"bits_per_channel": 16' in window.inspector.details.toPlainText()
    window.dirty = False
    window.close()


def test_toolbox_tree_types_subtypes_search_and_leaf_activation(app, monkeypatch):
    window = MainWindow()
    tree = window.library
    input_group = tree.findItems("Input", Qt.MatchFlag.MatchExactly)[0]
    assert input_group.childCount() >= 3
    histogram = tree.findItems(
        "Histogram", Qt.MatchFlag.MatchExactly | Qt.MatchFlag.MatchRecursive
    )[0]
    assert histogram.parent().text(0) == "Statistics"
    assert histogram.parent().parent().text(0) == "Image processing"
    before = len(window.graph.blocks)
    window.add_block(input_group)
    assert len(window.graph.blocks) == before
    window.add_block(histogram)
    assert window.inspector.block.type_id == "inspect.histogram"
    window.search.setText("Lens distortion")
    assert tree.topLevelItemCount() == 1 and tree.topLevelItem(0).text(0) == "Calibration"
    leaf = tree.findItems(
        "Calibrate lens ToolBlock", Qt.MatchFlag.MatchExactly | Qt.MatchFlag.MatchRecursive
    )[0]
    window.add_block(leaf)
    toolblock = window.inspector.block
    assert toolblock.block_type == "Calibration"
    window.inspector.type_field.setText("Custom calibration")
    window.inspector.subtype_field.setText("Camera A")
    window.inspector.apply()
    assert toolblock.definition()["subtype"] == "Camera A"
    window.enter_toolblock(toolblock)
    child = next(iter(window.graph.blocks.values()))
    window.scene.nodes[child.id].setSelected(True)
    window.inspector.fields["image_paths"].setText('["a.png", "b.png", "c.png"]')
    window.inspector.apply()
    assert child.parameters["image_paths"] == ["a.png", "b.png", "c.png"]
    errors = []
    monkeypatch.setattr(window, "warn", errors.append)
    window.inspector.fields["image_paths"].setText('"not a list"')
    assert not window.inspector.apply()
    assert not errors and "requires a JSON list" in window.inspector.validation_label.text()
    assert child.parameters["image_paths"] == ["a.png", "b.png", "c.png"]
    window.dirty = False
    window.close()


def test_starter_project_editor_run_edit_save_and_template_example(app, load_example, tmp_path):
    window = MainWindow()
    window.show()
    load_example(window, "starter_inspection")
    assert len(window.graph.blocks) == 9
    components = window.graph.blocks["starter_components"]
    window.scene.nodes[components.id].setSelected(True)
    window.start_run(False)
    pump_until(app, lambda: window.runner is None)
    assert components.state == "pass" and components.last_result.outputs["count"] == 2
    window.inspector.preview_choice.setCurrentText("Output / overlay")
    assert not window.inspector.image.pixmap().isNull()
    window.inspector.fields["expected_count"].setText("3")
    window.inspector.apply()
    window.start_run(False)
    pump_until(app, lambda: window.runner is None)
    assert components.state == "fail" and components.last_result.outputs["count"] == 2
    window.inspector.fields["expected_count"].setText("2")
    window.inspector.apply()
    window.path = tmp_path / "edited-project.json"
    assert window.save_project()
    window.replace_graph(Graph.load(window.path, window.registry), window.path)
    window.start_run(False)
    pump_until(app, lambda: window.runner is None)
    assert all(b.state == "pass" for b in window.graph.blocks.values())
    load_example(window, "template_matching")
    window.start_run(False)
    pump_until(app, lambda: window.runner is None)
    assert window.graph.blocks["match_result"].last_result.outputs["location"]["top_left"] == [
        250,
        85,
    ]
    window.dirty = False
    window.close()


def test_project_relative_image_works_in_editor_worker(app, tmp_path, monkeypatch):
    from test_lena_example import make_bundle

    project = make_bundle(tmp_path / "bundle")
    monkeypatch.chdir(tmp_path)
    window = MainWindow()
    window.replace_graph(Graph.load(project, window.registry), project)
    window.show()
    window.start_run(False)
    pump_until(app, lambda: window.runner is None)
    assert all(b.state == "pass" for b in window.graph.blocks.values())
    source = window.graph.blocks["lena_source"]
    assert source.parameters["path"] == "../assets/lena.jpg"
    window.dirty = False
    window.close()


def test_block_sizes_space_every_socket_and_scale_the_thumbnail(app):
    class ManyPorts(Block):
        inputs = tuple(Port(f"input_{index}") for index in range(12))
        outputs = tuple(Port(f"output_{index}") for index in range(9))

    graph = Graph()
    block = graph.add(ManyPorts())
    target = graph.add(ManyPorts())
    target.position = [500, 0]
    graph.connect(block.id, "output_8", target.id, "input_11")
    scene = CanvasScene(graph)
    node = scene.nodes[block.id]
    assert block.node_size == "medium"
    medium_thumbnail = node.thumbnail_rect.size()
    medium_width = node.width
    for size in ("small", "large"):
        node.set_size(size)
        for output in (False, True):
            sockets = [
                socket for (is_output, _), socket in node.sockets.items() if is_output == output
            ]
            assert all(
                node.sceneBoundingRect().contains(socket.sceneBoundingRect()) for socket in sockets
            )
            assert all(
                a.sceneBoundingRect().bottom() < b.sceneBoundingRect().top()
                for a, b in zip(sockets, sockets[1:])
            )
        wire = scene.wires[0]
        assert wire.path().pointAtPercent(0) == node.sockets[True, "output_8"].scenePos()
        if size == "small":
            assert node.width < medium_width
            assert node.thumbnail_rect.isEmpty()
        else:
            assert node.width > medium_width
            assert node.thumbnail_rect.width() > medium_thumbnail.width()
            assert node.thumbnail_rect.height() > medium_thumbnail.height()


def test_layout_moves_and_sizes_persist_root_and_nested_without_rewiring_or_result_loss(
    app, tmp_path
):
    window = MainWindow()
    graph = window.graph
    original = list(graph.blocks.values())
    toolblock = encapsulate(graph, [original[1].id, original[2].id], "Preprocess", window.registry)
    window.scene.rebuild()
    window.start_run(False)
    pump_until(app, lambda: window.runner is None)
    root_wires = list(graph.connections)
    nested_wires = list(toolblock.graph.connections)
    root_node = window.scene.nodes[toolblock.id]
    root_node.setSelected(True)
    window.inspector.node_size.setCurrentText("Large")
    assert window.dirty and toolblock.state == "pass"
    root_node.setPos(235.5, -128.25)
    assert window.dirty and toolblock.state == "pass"
    assert graph.connections == root_wires
    window.enter_toolblock(toolblock)
    child = next(iter(toolblock.graph.blocks.values()))
    child_node = window.scene.nodes[child.id]
    child_node.setSelected(True)
    window.inspector.node_size.setCurrentText("Small")
    child_node.setPos(-61.25, 290.75)
    assert toolblock.state == "pass" and child.state == "pass"
    assert toolblock.graph.connections == nested_wires
    # Running copies results, without replacing either saved layout.
    window.start_run(False)
    pump_until(app, lambda: window.runner is None)
    assert child.position == [-61.25, 290.75]
    window.path = tmp_path / "layout.json"
    assert window.save_project()
    restored = Graph.load(window.path, window.registry)
    assert restored.blocks[toolblock.id].position == [235.5, -128.25]
    assert restored.blocks[toolblock.id].graph.blocks[child.id].position == [-61.25, 290.75]
    assert restored.blocks[toolblock.id].node_size == "large"
    assert restored.blocks[toolblock.id].graph.blocks[child.id].node_size == "small"
    assert restored.connections == root_wires
    assert restored.blocks[toolblock.id].graph.connections == nested_wires
    window.replace_graph(restored, window.path)
    assert window.scene.nodes[toolblock.id].pos().x() == 235.5
    window.scene.nodes[toolblock.id].setSelected(True)
    assert window.inspector.node_size.currentText() == "Large"
    window.close()


def test_enum_fields_are_dropdowns_and_unknown_values_are_not_silently_replaced(app, monkeypatch):
    from PySide6.QtWidgets import QComboBox

    window = MainWindow()
    for kind, cls in window.registry.types.items():
        block = window.registry.create(kind)
        window.inspector.select(block)
        for key in cls.parameter_choices:
            if key in window.inspector.kernel_parameters:
                dialog = KernelEditor(block, window.inspector.kernel_parameters)
                assert isinstance(dialog.shape, QComboBox)
                assert dialog.shape.currentData() == block.parameters[key]
                dialog.deleteLater()
                continue
            field = window.inspector.fields[key]
            assert isinstance(field, QComboBox)
            assert field.currentText() == str(block.parameters[key])
    block = window.graph.add(window.registry.create("inspect.orb"))
    block.parameters["score_type"] = "obsolete"
    window.inspector.select(block)
    assert window.inspector.fields["score_type"].currentText() == "obsolete"
    errors = []
    monkeypatch.setattr(window, "warn", errors.append)
    assert not window.inspector.apply()
    assert not errors and "must be one of" in window.inspector.validation_label.text()
    assert block.parameters["score_type"] == "obsolete"
    window.dirty = False
    window.close()


def test_object_collection_editor_filter_sort_preview_worker_and_save(app, load_example, tmp_path):
    from PySide6.QtWidgets import QComboBox

    window = MainWindow()
    load_example(window, "object_collections")
    window.scene.nodes["objects_blob_filter"].setSelected(True)
    assert isinstance(window.inspector.fields["metric"], QComboBox)
    assert window.inspector.fields["metric"].currentText() == "height"
    window.start_run(False)
    pump_until(app, lambda: window.runner is None)
    assert all(block.state == "pass" for block in window.graph.blocks.values())
    window.scene.clearSelection()
    window.scene.nodes["objects_blob_draw"].setSelected(True)
    window.inspector.preview_choice.setCurrentText("Output / overlay")
    assert not window.inspector.image.pixmap().isNull()
    window.scene.clearSelection()
    window.scene.nodes["objects_blob_sort"].setSelected(True)
    window.inspector.fields["direction"].setCurrentText("descending")
    window.inspector.apply()
    window.path = tmp_path / "objects.json"
    assert window.save_project()
    loaded = Graph.load(window.path, window.registry)
    loaded.run()
    objects = loaded.blocks["objects_blob_sort"].last_result.outputs["collection"]["items"]
    assert objects[0]["metrics"]["center_x"] > objects[1]["metrics"]["center_x"]
    assert loaded.blocks["objects_blob_draw"].last_result.outputs["count"] == 2
    window.dirty = False
    window.close()


def test_inspector_prefers_result_or_output_and_preserves_manual_choice(app):
    window = MainWindow()
    graph = Graph()
    source = graph.add(window.registry.create("cv.threshold"))
    overlay = graph.add(window.registry.create("cv.overlay_result_graphics"))
    window.replace_graph(graph)
    inspector = window.inspector
    image = np.zeros((10, 12), np.uint8)
    try:
        source.execute({"image": image})
        overlay.execute({"image": image})
        inspector.select(source)
        assert inspector.preview_choice.currentText() == "Output / image"
        inspector.preview_choice.setCurrentText("Input / image")
        inspector.preview_choice.textActivated.emit("Input / image")
        inspector.refresh_result()
        assert inspector.preview_choice.currentText() == "Input / image"
        inspector.select(overlay)
        assert inspector.preview_choice.currentText() == "Result / image"
        inspector.select(source)
        assert inspector.preview_choice.currentText() == "Output / image"
    finally:
        window.close()


def test_inspector_advanced_visibility_and_unapplied_drafts(app):
    window = MainWindow()
    window.show()
    graph = Graph()
    block = graph.add(window.registry.create("cv.threshold"))
    other = graph.add(window.registry.create("cv.threshold"))
    window.replace_graph(graph)
    inspector = window.inspector
    try:
        inspector.select(block)
        app.processEvents()
        assert not inspector.show_advanced.isChecked()
        assert inspector.fields["value"].isVisible()
        assert not inspector.recipe_group.isVisible()
        inspector.show_advanced.setChecked(True)
        assert inspector.recipe_group.isVisible()
        assert inspector.node_size.isVisible()
        inspector.fields["value"].setText("42")
        inspector.show_advanced.setChecked(False)
        assert inspector.fields["value"].text() == "42"
        inspector.select(other)
        inspector.select(block)
        assert inspector.fields["value"].text() == str(block.parameters["value"])
        assert inspector.fields["value"].text() != "42"
        assert not inspector.show_advanced.isChecked()
    finally:
        window.close()


def test_file_menu_opens_example_from_disk(app, monkeypatch):
    window = MainWindow()
    project = Path(__file__).resolve().parents[1] / "examples" / "regions" / "regions.json"
    file_menu = window.menuBar().actions()[0].menu()
    entries = {action.text(): action for action in file_menu.actions()}
    try:
        assert all("example" not in name.lower() for name in entries)
        assert "Load demo" not in entries
        assert "Load starter inspection" not in entries
        monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(project), ""))
        entries["Open…"].trigger()
        assert window.path == project
        assert "roi_circle" in window.graph.blocks
    finally:
        window.close()
