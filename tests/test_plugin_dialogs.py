"""Offscreen coverage for plugin dialogs and their cached image context."""

from pathlib import Path
from threading import get_ident

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QDialog, QFileDialog, QLabel, QPushButton, QSpinBox
from test_editor import pump_until

from opencv_dataflow.app import MainWindow
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Block, Graph, Port, Result
from opencv_dataflow.images import DepthArray
from opencv_dataflow.payloads import frame, metadata
from opencv_dataflow.plugin_dialog import PluginDialogContext
from opencv_dataflow.plugin_image_picker import PluginImagePicker
from opencv_dataflow.plugins import load_plugin

_EXAMPLE = Path(__file__).parents[1] / "examples/custom_invert"


@pytest.fixture
def example_window(app):
    window = MainWindow()
    load_plugin(_EXAMPLE / "custom_invert.py", window.registry)
    graph = Graph.load(_EXAMPLE / "custom_invert.json", window.registry)
    window.replace_graph(graph)
    window.show()
    block = graph.blocks["invert"]
    window.scene.nodes[block.id].setSelected(True)
    try:
        yield window, block
    finally:
        window.dirty = False
        window.close()
        app.processEvents()


def test_popup_uses_current_draft_and_stages_only_accepted_values(example_window, monkeypatch):
    window, block = example_window
    inspector = window.inspector
    start, end = inspector.fields["start_percent"], inspector.fields["end_percent"]
    end.setText("77")
    button = next(
        button
        for button in inspector.editor.findChildren(QPushButton)
        if button.text() == "Edit inversion band…"
    )
    schedules = []
    original_schedule = inspector.schedule_live_preview

    def schedule(*args):
        schedules.append(inspector.read_parameters())
        original_schedule(*args)

    monkeypatch.setattr(inspector, "schedule_live_preview", schedule)
    button.click()
    dialog = inspector._plugin_dialog
    assert dialog.isVisible() and dialog.parentWidget() is inspector
    assert dialog.findChild(QSpinBox, "end_percent").value() == 77
    dialog.findChild(QSpinBox, "start_percent").setValue(35)
    dialog.reject()
    assert inspector._plugin_dialog is None
    assert start.text() == "20" and end.text() == "77"
    assert not schedules and not window.dirty

    button.click()
    dialog = inspector._plugin_dialog
    dialog.findChild(QSpinBox, "start_percent").setValue(30)
    dialog.findChild(QSpinBox, "end_percent").setValue(70)
    dialog.accept()
    assert inspector._plugin_dialog is None
    assert start.text() == "30" and end.text() == "70"
    assert schedules == [{"start_percent": 30, "end_percent": 70}]
    assert block.parameters == {"start_percent": 20, "end_percent": 80}
    assert not window.dirty
    assert inspector.apply()
    assert block.parameters == schedules[0] and window.dirty


@pytest.mark.parametrize(
    ("updates", "warning"),
    [
        ({"start_percent": 90}, "Start percent must be less"),
        ({"start_percent": "25"}, "start_percent must have type int"),
        ({"extra": True}, "declared parameter names"),
    ],
)
def test_invalid_plugin_edits_are_atomic(example_window, monkeypatch, updates, warning):
    window, block = example_window
    inspector = window.inspector
    warnings = []
    monkeypatch.setattr(window, "warn", warnings.append)

    def create(self, parent, parameters, context):
        assert context.block_id == block.id and context.project_dir == _EXAMPLE
        parameters.update(updates)
        return QDialog(parent)

    monkeypatch.setattr(type(block), "create_plugin_dialog", create)
    inspector.open_plugin_dialog()
    inspector._plugin_dialog.accept()
    assert warnings and warning in warnings[0]
    assert inspector.read_parameters() == {"start_percent": 20, "end_percent": 80}
    assert not window.dirty and inspector._plugin_dialog is None
    assert block.logger.snapshot()["events"][-1]["kind"] == "exception"


def test_factory_failure_and_selection_change_keep_configuration(example_window, monkeypatch):
    window, block = example_window
    inspector = window.inspector
    warnings = []
    monkeypatch.setattr(window, "warn", warnings.append)

    def fail(self, parent, parameters, context):
        raise SystemExit("test plugin exit")

    with monkeypatch.context() as patch:
        patch.setattr(type(block), "create_plugin_dialog", fail)
        inspector.open_plugin_dialog()
    assert inspector._plugin_dialog is None
    assert warnings and "test plugin exit" in warnings[0]
    inspector.open_plugin_dialog()
    dialog = inspector._plugin_dialog
    rejected = []
    dialog.rejected.connect(lambda: rejected.append(True))
    dialog.findChild(QSpinBox, "start_percent").setValue(30)
    inspector.select(window.graph.blocks["source"])
    assert rejected == [True] and not dialog.isVisible()
    dialog.accept()
    inspector.select(block)
    assert inspector.read_parameters() == {"start_percent": 20, "end_percent": 80}
    assert not window.dirty and inspector._plugin_dialog is None


def test_modeless_dialog_captures_two_runs_and_stages_during_execution(
    app, example_window, monkeypatch, tmp_path
):
    window, example = example_window
    graph = Graph()
    paths = []
    for index, value in enumerate((23, 89, 181)):
        path = tmp_path / f"frame_{index}.png"
        assert cv2.imwrite(str(path), np.full((20, 24), value, np.uint8))
        paths.append(str(path))
    source = graph.add(
        window.registry.create("input.file", parameters={"source_mode": "files", "paths": paths})
    )
    block = graph.add(window.registry.create(example.type_id, parameters=example.parameters))
    graph.connect(source.id, "image", block.id, "image")
    window.replace_graph(graph)
    window.scene.nodes[block.id].setSelected(True)
    inspector = window.inspector
    state = {}
    create_example_dialog = type(block).create_plugin_dialog

    def create(self, parent, parameters, context):
        state.update(parameters=parameters, context=context)
        return create_example_dialog(self, parent, parameters, context)

    monkeypatch.setattr(type(block), "create_plugin_dialog", create)
    inspector.open_plugin_dialog()
    dialog = inspector._plugin_dialog
    assert dialog.isVisible() and not dialog.isModal()
    context = state["context"]
    assert context.image_outputs == ()
    picker = dialog.findChild(PluginImagePicker)
    summary = dialog.findChild(QLabel, "capture_summary")
    output_notice = dialog.findChild(QLabel, "output_notice")

    for revision, expected in enumerate((23, 89), 1):
        window.start_run(False)
        pump_until(
            app,
            lambda: window.runner is None and f"output {revision}" in output_notice.text(),
        )
        assert inspector._plugin_dialog is dialog and dialog.isVisible()
        assert context.output_revision == revision
        assert any(
            picker.source_combo.itemData(row) == ("output", source.id, "image")
            for row in range(picker.source_combo.count())
        )
        picker.capture_button.click()
        assert any(row.block_id == source.id for row in context.refresh_images())
        image = context.snapshot_image(source.id, "image")
        assert np.all(image.value == expected)
        if expected == 23:
            reference = image
    assert np.all(reference.value == 23)
    assert reference.payload["ticket"] != image.payload["ticket"]
    assert "Previous" in summary.text() and "Latest" in summary.text()

    state["parameters"]["start_percent"] = 35
    window.start_run(False)
    assert window.runner is not None and inspector._plugin_dialog is dialog
    dialog.accept()
    assert inspector.read_parameters()["start_percent"] == 35
    assert window.runner.graph.blocks[block.id].parameters["start_percent"] == 20
    assert not inspector.apply() and block.parameters["start_percent"] == 20
    pump_until(app, lambda: window.runner is None)
    assert inspector.apply() and block.parameters["start_percent"] == 35
    assert np.all(reference.value == 23)
    with pytest.raises(RuntimeError, match="closed"):
        context.refresh_images()


def test_modeless_dialog_merges_independent_edits_and_rejects_conflicts(
    example_window, monkeypatch
):
    window, block = example_window
    inspector = window.inspector
    warnings = []
    monkeypatch.setattr(window, "warn", warnings.append)

    inspector.open_plugin_dialog()
    dialog = inspector._plugin_dialog
    dialog.findChild(QSpinBox, "start_percent").setValue(30)
    inspector.fields["end_percent"].setText("70")
    dialog.accept()
    assert inspector.read_parameters() == {"start_percent": 30, "end_percent": 70}

    inspector.open_plugin_dialog()
    dialog = inspector._plugin_dialog
    dialog.findChild(QSpinBox, "start_percent").setValue(35)
    dialog.findChild(QSpinBox, "end_percent").setValue(65)
    inspector.fields["start_percent"].setText("40")
    dialog.accept()
    assert warnings and "both changed start_percent" in warnings[0]
    assert inspector.read_parameters() == {"start_percent": 40, "end_percent": 70}
    assert block.parameters == {"start_percent": 20, "end_percent": 80}
    assert not window.dirty


def test_modal_dialog_closes_when_graph_starts(app, example_window, monkeypatch):
    window, block = example_window
    inspector = window.inspector
    monkeypatch.setattr(type(block), "plugin_dialog_modeless", False)
    inspector.open_plugin_dialog()
    dialog = inspector._plugin_dialog
    assert dialog.isModal()
    dialog.findChild(QSpinBox, "start_percent").setValue(35)
    window.start_run(False)
    pump_until(app, lambda: window.runner is None)
    assert inspector._plugin_dialog is None
    assert inspector.read_parameters()["start_percent"] == 20


def test_changing_project_directory_discards_open_dialog(example_window, monkeypatch, tmp_path):
    window, block = example_window
    inspector = window.inspector
    inspector.open_plugin_dialog()
    dialog = inspector._plugin_dialog
    dialog.findChild(QSpinBox, "start_percent").setValue(35)
    path = tmp_path / "relocated.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), ""))
    assert window.save_project(save_as=True)
    assert inspector._plugin_dialog is None
    assert window.graph.project_dir == tmp_path
    assert block.parameters["start_percent"] == 20


def test_output_notifications_coalesce_and_cancel_with_dialog(app, example_window, monkeypatch):
    window, block = example_window
    inspector = window.inspector
    seen = []
    dialogs = []

    def create(self, parent, parameters, context):
        dialogs.append(context)
        return QDialog(parent)

    monkeypatch.setattr(type(block), "create_plugin_dialog", create)
    inspector.open_plugin_dialog()
    (context,) = dialogs
    assert context.output_revision == 0
    unsubscribe = context.subscribe_outputs(seen.append)
    for _ in range(3):
        inspector.plugin_outputs_published()
    assert not seen and context.output_revision == 3
    pump_until(app, lambda: seen == [3])

    unsubscribe()
    unsubscribe()
    inspector.plugin_outputs_published()
    app.processEvents()
    assert seen == [3] and context.output_revision == 4

    context.subscribe_outputs(seen.append)
    inspector.plugin_outputs_published()
    inspector.close_plugin_dialog()
    app.processEvents()
    assert seen == [3]
    with pytest.raises(RuntimeError, match="closed"):
        context.subscribe_outputs(seen.append)


def test_output_callback_errors_are_isolated_from_other_callbacks_and_runs(
    app, example_window, monkeypatch
):
    window, block = example_window
    inspector = window.inspector
    source = window.graph.blocks["source"]
    main_thread = get_ident()
    failed, received = [], []

    def bad(revision):
        failed.append(revision)
        raise SystemExit("bad output callback")

    def create(self, parent, parameters, context):
        def observe(revision):
            catalog_was_empty = not context.image_outputs
            context.refresh_images()
            snapshot = context.snapshot_image(source.id, "image")
            received.append((revision, get_ident(), catalog_was_empty, snapshot.payload["ticket"]))

        context.subscribe_outputs(bad)
        context.subscribe_outputs(observe)
        return QDialog(parent)

    monkeypatch.setattr(type(block), "create_plugin_dialog", create)
    inspector.open_plugin_dialog()
    for count in (1, 2):
        window.start_run(False)
        pump_until(app, lambda: window.runner is None and len(received) == count)
        assert block.state == "pass"
    assert failed == [1]
    assert [entry[:3] for entry in received] == [(1, main_thread, True), (2, main_thread, False)]
    assert received[0][3] != received[1][3]
    errors = [event for event in block.logger.snapshot()["events"] if event["kind"] == "exception"]
    assert any("Plugin output subscriber failed" in event["message"] for event in errors)


def test_snapshots_copy_pixels_and_keep_selected_level_provenance(tmp_path):
    class MaskSource(Block):
        type_id = "input.test_dialog_mask"
        outputs = (Port("image"),)

        def process(self, inputs):
            return Result({"image": np.tile(np.uint8([0, 64, 255, 0]), (8, 3))})

    graph = Graph()
    graph.bind_project_dir(tmp_path)
    source = graph.add(MaskSource())
    pyramid = graph.add(builtin_registry().create("cv.image_pyramid", parameters={"levels": 2}))
    graph.connect(source.id, "image", pyramid.id, "image")
    graph.run()
    context = PluginDialogContext(graph, pyramid)
    assert context.block_id == pyramid.id and context.project_dir == tmp_path
    assert {(entry.port, entry.level_count) for entry in context.image_outputs} == {
        ("image", 1),
        ("images", 2),
    }
    mask = context.snapshot_image(source.id, "image")
    assert mask.value.dtype == np.uint8 and set(np.unique(mask.value)) == {0, 64, 255}
    mask.value[:] = 17
    mask.payload["spatial"]["image_size"][0] = 99
    again = context.snapshot_image(source.id, "image")
    assert set(np.unique(again.value)) == {0, 64, 255}
    assert again.payload["spatial"]["image_size"] == [12, 8]
    level = context.snapshot_image(pyramid.id, "images", level=1)
    assert level.value.shape == (4, 6)
    assert level.payload["spatial"]["image_size"] == [6, 4]
    assert level.payload["pyramid"]["level"] == 1
    assert level.payload["ticket"] == again.payload["ticket"]
    assert metadata(level.value) == level.payload
    with pytest.raises(ValueError, match="outside the available range"):
        context.snapshot_image(pyramid.id, "images", level=2)


def test_popup_captures_wired_draft_level_and_exact_processed_level(example_window, monkeypatch):
    window, example = example_window
    graph = Graph()
    source = graph.add(window.registry.create("input.demo"))
    pyramid = graph.add(window.registry.create("cv.image_pyramid", parameters={"levels": 3}))
    block = graph.add(window.registry.create(example.type_id, parameters=example.parameters))
    block.input_levels = {"image": "1"}
    graph.connect(source.id, "image", pyramid.id, "image")
    graph.connect(pyramid.id, "images", block.id, "image")
    window.replace_graph(graph)
    graph.run()
    window.scene.nodes[block.id].setSelected(True)
    inspector = window.inspector
    inspector.level_fields["image"].setCurrentText("2")
    contexts = []

    def create(self, parent, parameters, context):
        contexts.append(context)
        return QDialog(parent)

    monkeypatch.setattr(type(block), "create_plugin_dialog", create)
    inspector.open_plugin_dialog()
    (context,) = contexts
    connected = context.snapshot_input_image("image")
    processed = context.snapshot_input_image("image", source="processed")
    largest = context.snapshot_input_image("image", level=0)
    levels = pyramid.last_result.outputs["images"].levels

    np.testing.assert_array_equal(connected.value, levels[2])
    np.testing.assert_array_equal(processed.value, block.last_inputs["image"])
    np.testing.assert_array_equal(largest.value, levels[0])
    assert connected.payload["pyramid"]["level"] == 2
    assert processed.payload == block.input_payloads["image"]
    assert largest.payload["pyramid"]["level"] == 0
    assert connected.payload["ticket"] == processed.payload["ticket"]
    assert connected.payload["spatial"]["to_root"] == metadata(levels[2])["spatial"]["to_root"]
    assert block.input_levels == {"image": "1"}
    connected.value[:] = 0
    assert np.any(levels[2])
    with pytest.raises(ValueError, match="single image"):
        context.snapshot_input_image("image", source="processed", level=2)


def test_connected_image_can_advance_when_the_block_rejects_it():
    class Source(Block):
        type_id = "input.test_dialog_capture"
        outputs = (Port("image"),)
        defaults = {"intensity": 20, "passed": True}

        def process(self, inputs):
            image = np.full((5, 6), self.parameters["intensity"], np.uint8)
            return Result({"image": image}, passed=self.parameters["passed"])

    graph = Graph()
    source = graph.add(Source())
    consumer = graph.add(builtin_registry().create("cv.gray"))
    consumer.input_failure_policies = {"image": "reject_failed"}
    graph.connect(source.id, "image", consumer.id, "image")
    context = PluginDialogContext(graph, consumer)
    with pytest.raises(ValueError, match="No connected image"):
        context.snapshot_input_image("image")
    graph.run()
    context.refresh_images()
    reference = context.snapshot_input_image("image", source="processed")
    source.parameters.update(intensity=90, passed=False)
    graph.run()
    assert consumer.input_rejected
    assert np.all(context.snapshot_input_image("image").value == 20)
    context.refresh_images()

    connected = context.snapshot_input_image("image")
    processed = context.snapshot_input_image("image", source="processed")
    assert np.all(connected.value == 90) and np.all(processed.value == 20)
    assert connected.payload["ticket"] != processed.payload["ticket"]
    assert not connected.payload["outcome"]["chain_passed"]
    assert processed.payload == reference.payload


def test_processed_image_capture_supports_injected_depth_inputs():
    class DepthTarget(Block):
        inputs = (Port("depth", "depth"),)

        def process(self, inputs):
            return Result({})

    graph = Graph()
    block = graph.add(DepthTarget())
    depth = DepthArray(np.full((2, 3), 0.75), units="m", display_max=2)
    depth.metadata["spatial"] = frame([3, 2], "depth camera")
    graph.run(injected={(block.id, "depth"): depth})
    context = PluginDialogContext(graph, block)
    with pytest.raises(ValueError, match="Connect a source"):
        context.snapshot_input_image("depth")
    captured = context.snapshot_input_image("depth", source="processed")
    assert isinstance(captured.value, DepthArray)
    assert captured.value.units == "m" and captured.value.display_max == 2
    assert captured.payload["spatial"]["root_id"] == "depth camera"
    np.testing.assert_array_equal(captured.value, depth)
