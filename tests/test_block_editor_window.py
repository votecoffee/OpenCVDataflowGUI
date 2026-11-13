"""Expanded editor layout and lifecycle using headless Qt widget events."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from opencv_dataflow.app import MainWindow


def select_inspection(window):
    block = list(window.graph.blocks.values())[-1]
    window.scene.nodes[block.id].setSelected(True)
    return block


def double_click_block(window, block):
    point = window.view.mapFromScene(window.scene.nodes[block.id].scenePos() + QPoint(80, 20))
    QTest.mouseDClick(window.view.viewport(), Qt.MouseButton.LeftButton, pos=point)


def test_sidebar_stays_full_height_with_both_bottom_panels(app):
    window = MainWindow()
    window.show()
    try:
        window.filmstrip_dock.show()
        window.diagnostics_dock.show()
        app.processEvents()
        side = window.inspector_dock.geometry()
        for panel in (window.filmstrip_dock, window.diagnostics_dock):
            assert panel.geometry().right() < side.left()
            assert panel.geometry().bottom() <= side.bottom()
        bottom = window.inspector.mapTo(window, QPoint(0, window.inspector.height())).y()
        assert abs(bottom - side.bottom()) <= 3
        assert window.inspector.isVisible()
    finally:
        window.close()


def test_double_click_separates_results_and_keeps_draft_when_redocked(app):
    window = MainWindow()
    window.show()
    app.processEvents()
    block = select_inspection(window)
    inspector = window.inspector
    field = inspector.fields["expected_count"]
    original = block.parameters["expected_count"]
    try:
        assert window.block_editor is None
        field.setText("7")
        double_click_block(window, block)
        app.processEvents()
        dialog = window.block_editor
        assert dialog is not None and dialog.isVisible() and not dialog.isModal()
        assert inspector.fields["expected_count"] is field and field.text() == "7"
        assert inspector.tabs.indexOf(inspector.results_page) == -1
        assert window.editor_split.orientation() == Qt.Orientation.Horizontal
        assert window.editor_split.widget(0) is inspector
        assert window.editor_split.widget(1) is inspector.results_page
        assert inspector.results_page.isVisible()
        before = window.editor_split.sizes()
        window.editor_split.setSizes([600, 400])
        app.processEvents()
        sizes = window.editor_split.sizes()
        assert sizes[0] > before[0]
        window.open_block_editor(block)
        assert window.block_editor is dialog
        QTest.keyClick(dialog, Qt.Key.Key_Escape)
        app.processEvents()
        assert window.block_editor is None
        assert window.inspector_host.currentWidget() is inspector
        assert inspector.tabs.indexOf(inspector.results_page) == 2
        assert field.text() == "7" and block.parameters["expected_count"] == original
        assert not window.dirty
        window.open_block_editor(block)
        app.processEvents()
        assert abs(window.editor_split.sizes()[0] - sizes[0]) <= 3
    finally:
        window.close()


def test_apply_rebuild_keeps_popup_open_and_updates_configuration(app):
    window = MainWindow()
    window.show()
    block = select_inspection(window)
    window.open_block_editor(block)
    dialog = window.block_editor
    try:
        window.inspector.fields["expected_count"].setText("3")
        window.inspector.name_field.setText("Expanded inspection")
        assert window.inspector.apply()
        app.processEvents()
        assert window.block_editor is dialog and dialog.isVisible()
        assert block.parameters["expected_count"] == 3 and window.dirty
        assert window.inspector.block is block
        assert "Expanded inspection" in dialog.windowTitle()
        assert window.inspector.fields["expected_count"].text() == "3"
    finally:
        window.dirty = False
        window.close()


def test_changing_selection_or_project_restores_sidebar_and_stops_live_preview(app):
    window = MainWindow()
    window.show()
    window.graph.run()
    block = select_inspection(window)
    window.open_block_editor(block)
    try:
        window.inspector.live_results.setChecked(True)
        window.inspector.fields["expected_count"].setText("9")
        other = list(window.graph.blocks.values())[0]
        window.scene.clearSelection()
        window.scene.nodes[other.id].setSelected(True)
        app.processEvents()
        assert window.block_editor is None
        assert window.inspector.block is other
        assert not window.inspector.live_results.isChecked()
        assert not window.inspector.live_timer.isActive()
        assert block.parameters["expected_count"] != 9
        window.open_block_editor(other)
        window.scene.clearSelection()
        app.processEvents()
        assert window.block_editor is None and window.inspector.block is None
        select_inspection(window)
        window.open_block_editor(block)
        window.new_project()
        assert window.block_editor is None and not window.graph.blocks
    finally:
        window.close()


def test_popup_opens_while_running_and_receives_results(app):
    window = MainWindow()
    window.show()
    app.processEvents()
    block = select_inspection(window)
    try:
        window.start_run(False)
        assert window.scene.locked
        double_click_block(window, block)
        assert window.block_editor is not None
        assert not window.inspector.apply_button.isEnabled()
        for _ in range(300):
            app.processEvents()
            if window.runner is None:
                break
            QTest.qWait(10)
        assert window.runner is None
        assert block.last_result.outputs["count"] == 2
        assert "count" in window.inspector.details.toPlainText()
        assert window.inspector.apply_button.isEnabled()
        assert not window.inspector.image.pixmap().isNull()
    finally:
        if window.runner is not None:
            window.stop_run()
            for _ in range(300):
                app.processEvents()
                if window.runner is None:
                    break
                QTest.qWait(10)
        window.close()


def test_run_block_in_popup_uses_retained_input_and_close_cancels_live_edits(app):
    window = MainWindow()
    window.show()
    window.graph.run()
    block = select_inspection(window)
    inspector = window.inspector
    inspector.tabs.setCurrentWidget(inspector.results_page)
    window.open_block_editor(block)
    try:
        assert inspector.tabs.currentWidget() is inspector.parameters_page
        inspector.fields["expected_count"].setText("1")
        inspector.run_block.click()
        for _ in range(300):
            app.processEvents()
            if inspector._live_worker is None and inspector._live_result is not None:
                break
            QTest.qWait(10)
        assert inspector._live_worker is None
        assert inspector._live_result.last_result.outputs["count"] == 2
        assert not inspector._live_result.last_result.passed
        assert "BLOCK PREVIEW" in inspector.details.toPlainText()
        assert not inspector.image.pixmap().isNull()
        assert block.last_result.passed and block.parameters["expected_count"] == 2
        inspector.live_results.setChecked(True)
        inspector.fields["expected_count"].setText("3")
        assert inspector.live_timer.isActive()
        window.close_block_editor()
        assert not inspector.live_results.isChecked() and not inspector.live_timer.isActive()
        assert inspector.tabs.currentWidget() is inspector.results_page
        assert inspector.fields["expected_count"].text() == "3" and not window.dirty
    finally:
        window.close()
