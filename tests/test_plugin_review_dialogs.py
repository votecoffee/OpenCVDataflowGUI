"""Offscreen read-only review factories, paired captures and popup lifecycle."""

import os
import subprocess
import sys

import numpy as np
import pytest
from PySide6.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout
from test_editor import pump_until
from test_plugin_dialogs import example_window as example_window

from opencv_dataflow.assistant_api import block_schema
from opencv_dataflow.core import Graph


@pytest.fixture
def review_window(example_window, monkeypatch):
    window, block = example_window
    state = {"factory_calls": 0, "captures": [], "revisions": []}

    def factory(self, parent, context):
        state["factory_calls"] += 1
        state["context"] = context
        dialog = QDialog(parent)
        layout = QVBoxLayout(dialog)
        label = QLabel("No explicit capture")
        layout.addWidget(label)
        capture = QPushButton("Capture latest")
        layout.addWidget(capture)

        def snapshot():
            pair = context.snapshot_processed_execution(["image"], ["image"])
            state["captures"].append(pair)
            label.setText(str(pair.execution_count))

        capture.clicked.connect(snapshot)
        unsubscribe = context.subscribe_outputs(state["revisions"].append)
        dialog.finished.connect(lambda _: unsubscribe())
        return dialog

    def training(*args):
        pytest.fail("Result review must not invoke the training/configuration factory")

    monkeypatch.setattr(type(block), "plugin_review_dialog_label", "Review completed result…")
    monkeypatch.setattr(type(block), "create_plugin_review_dialog", factory)
    monkeypatch.setattr(type(block), "create_plugin_dialog", training)
    window.inspector.select(block)
    window.inspector.tabs.setCurrentWidget(window.inspector.results_page)
    return window, block, state


def test_review_opens_without_execution_or_reading_invalid_draft_and_accept_does_not_stage(
    review_window,
):
    window, block, state = review_window
    inspector = window.inspector
    inspector.fields["end_percent"].setText("invalid draft")
    before = dict(block.parameters)
    inspector.plugin_review_button.click()
    dialog = inspector._plugin_review_dialog
    assert dialog is not None and dialog.isVisible() and not dialog.isModal()
    assert state["factory_calls"] == 1 and block.timing.count == 0
    assert not state["captures"] and not window.dirty
    inspector.plugin_review_button.click()
    assert state["factory_calls"] == 1  # Reopen raises the existing popup.
    context = state["context"]
    dialog.accept()
    assert inspector._plugin_review_dialog is None and inspector._plugin_review_context is None
    assert inspector.fields["end_percent"].text() == "invalid draft"
    assert block.parameters == before and not window.dirty
    with pytest.raises(RuntimeError, match="closed"):
        context.snapshot_processed_execution()


def test_actual_worker_publications_refresh_review_without_auto_capture_or_configuration_edits(
    app, review_window
):
    window, block, state = review_window
    inspector = window.inspector
    inspector.plugin_review_button.click()
    dialog = inspector._plugin_review_dialog
    for revision in (1, 2):
        window.start_run(False)
        assert inspector._plugin_review_dialog is dialog
        pump_until(
            app,
            lambda: window.runner is None and state["revisions"] == list(range(1, revision + 1)),
        )
        assert len(state["captures"]) == revision - 1
        dialog.findChild(QPushButton).click()
        pair = state["captures"][-1]
        assert pair.output_revision == revision and pair.execution_count == revision
        np.testing.assert_array_equal(
            pair.outputs["image"].value, block.last_result.outputs["image"]
        )
        assert pair.inputs["image"].payload["ticket"] == pair.outputs["image"].payload["ticket"]
    assert not window.dirty
    first = state["captures"][0]
    dialog.reject()
    assert first.outputs["image"].value.size
    with pytest.raises(RuntimeError, match="closed"):
        state["context"].refresh_images()


@pytest.mark.parametrize("action", ["selection", "graph", "hide", "close"])
def test_review_disposes_context_on_normal_editor_lifetime_events(review_window, action):
    window, _, state = review_window
    inspector = window.inspector
    inspector.plugin_review_button.click()
    context = state["context"]
    if action == "selection":
        inspector.select(None)
    elif action == "graph":
        window.replace_graph(Graph())
    elif action == "hide":
        inspector.hide()
    else:
        window.dirty = False
        window.close()
    assert inspector._plugin_review_dialog is None and inspector._plugin_review_context is None
    with pytest.raises(RuntimeError, match="closed"):
        context.snapshot_processed_execution()


@pytest.mark.parametrize("result", ["wrong_type", "exception", "system_exit"])
def test_bad_review_factory_is_contained_and_retry_works(review_window, monkeypatch, result):
    window, block, state = review_window
    original = type(block).create_plugin_review_dialog
    warnings = []
    monkeypatch.setattr(window, "warn", warnings.append)

    def broken(self, parent, context):
        state["broken_context"] = context
        if result == "wrong_type":
            return object()
        if result == "system_exit":
            raise SystemExit("synthetic review exit")
        raise ValueError("synthetic review error")

    monkeypatch.setattr(type(block), "create_plugin_review_dialog", broken)
    window.inspector.open_plugin_review_dialog()
    assert warnings and window.inspector._plugin_review_dialog is None
    with pytest.raises(RuntimeError, match="closed"):
        state["broken_context"].refresh_images()
    monkeypatch.setattr(type(block), "create_plugin_review_dialog", original)
    window.inspector.open_plugin_review_dialog()
    assert window.inspector._plugin_review_dialog is not None and not window.dirty


def test_review_callback_failure_is_logged_without_closing_popup_or_stopping_other_callbacks(
    app, review_window
):
    window, _, state = review_window
    inspector = window.inspector
    inspector.open_plugin_review_dialog()
    context = state["context"]
    calls = []

    def broken(_revision):
        raise ValueError("synthetic review subscriber")

    context.subscribe_outputs(broken)
    context.subscribe_outputs(calls.append)
    inspector.plugin_outputs_published()
    pump_until(app, lambda: calls == [1])
    assert inspector._plugin_review_dialog is not None and calls == [1]
    inspector.plugin_outputs_published()
    pump_until(app, lambda: calls == [1, 2])


def test_assistant_catalog_advertises_review_capability_without_opening_factory(review_window):
    window, block, state = review_window
    description = block_schema(block.type_id, window.registry)
    assert description["capabilities"]["plugin_review_dialog"] is True
    assert state["factory_calls"] == 0


def test_blocks_without_review_hook_keep_results_button_hidden(example_window):
    window, _ = example_window
    assert window.inspector.plugin_review_button.isHidden()


@pytest.mark.parametrize("close_side", ["training", "review"])
def test_training_and_review_contexts_receive_publications_and_close_independently(
    app, example_window, monkeypatch, close_side
):
    window, block = example_window
    calls = {"training": [], "review": []}

    def factory(self, parent, context):
        context.subscribe_outputs(calls["review"].append)
        return QDialog(parent)

    monkeypatch.setattr(type(block), "plugin_dialog_modeless", True)
    monkeypatch.setattr(type(block), "plugin_review_dialog_label", "Review")
    monkeypatch.setattr(type(block), "create_plugin_review_dialog", factory)
    inspector = window.inspector
    inspector.select(block)
    inspector.open_plugin_dialog()
    inspector._plugin_context.subscribe_outputs(calls["training"].append)
    inspector.open_plugin_review_dialog()
    inspector.plugin_outputs_published()
    pump_until(app, lambda: calls == {"training": [1], "review": [1]})
    closed = (
        inspector._plugin_dialog if close_side == "training" else inspector._plugin_review_dialog
    )
    closed.reject()
    inspector.plugin_outputs_published()
    surviving = "review" if close_side == "training" else "training"
    pump_until(app, lambda: calls[surviving] == [1, 2])
    assert calls[close_side] == [1]


def test_review_context_release_is_idempotent_after_qt_timer_destruction(
    review_window, monkeypatch
):
    window, _, state = review_window
    inspector = window.inspector
    inspector.open_plugin_review_dialog()
    context = state["context"]

    class DeletedTimer:
        def stop(self):
            raise RuntimeError("Internal C++ QTimer already deleted")

    monkeypatch.setattr(inspector, "_plugin_output_timer", DeletedTimer())
    inspector._release_plugin_context(context)
    inspector._release_plugin_context(context)
    inspector.close_plugin_review_dialog()
    with pytest.raises(RuntimeError, match="closed"):
        context.refresh_images()


def test_native_deleted_timer_and_popup_teardown_exit_cleanly_in_an_isolated_qt_process():
    # Native object deletion is deliberately destructive; do not leave its Qt
    # wrappers for collection by a later unrelated graph-worker test.
    script = r"""
import gc
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QDialog
from shiboken6 import delete
from opencv_dataflow.app import MainWindow
from opencv_dataflow.core import Block, Graph
class Review(Block):
    plugin_review_dialog_label = "Review"
    def create_plugin_review_dialog(self, parent, context):
        return QDialog(parent)
app = QApplication([])
window = MainWindow()
graph = Graph()
block = graph.add(Review())
window.replace_graph(graph)
window.inspector.select(block)
window.inspector.open_plugin_review_dialog()
context = window.inspector._plugin_review_context
assert context is not None
window.show()
app.processEvents()  # Finish the unrelated initial canvas viewport restoration.
delete(window.inspector._plugin_output_timer)
window.inspector._release_plugin_context(context)
window.inspector._release_plugin_context(context)
window.inspector.close_plugin_review_dialog()
window.dirty = False
window.close()
window.deleteLater()
QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
app.processEvents()
del window, graph, block, context
gc.collect()
print("native review teardown closed", flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", script],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "native review teardown closed" in result.stdout
    assert "Traceback" not in result.stderr
