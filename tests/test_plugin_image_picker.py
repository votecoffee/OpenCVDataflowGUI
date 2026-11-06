"""Offscreen coverage for the reusable plugin image picker."""

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QDialog

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import metadata
from opencv_dataflow.plugin_dialog import PluginDialogContext
from opencv_dataflow.plugin_image_picker import PluginImagePicker


@pytest.fixture
def sources():
    graph = Graph()
    registry = builtin_registry()
    source = graph.add(registry.create("input.demo"))
    pyramid = graph.add(registry.create("cv.image_pyramid", parameters={"levels": 3}))
    block = graph.add(registry.create("cv.gray"))
    block.input_levels = {"image": "1"}
    graph.connect(source.id, "image", pyramid.id, "image")
    graph.connect(pyramid.id, "images", block.id, "image")
    return graph, source, pyramid, block


def _source_row(picker, key):
    return next(
        row
        for row in range(picker.source_combo.count())
        if picker.source_combo.itemData(row) == key
    )


def test_picker_retains_source_and_level_across_cache_changes(app, sources):
    graph, source, pyramid, block = sources
    context = PluginDialogContext(graph, block)
    picker = PluginImagePicker(context)
    packets = []
    picker.captured.connect(packets.append)
    try:
        assert not picker.capture_button.isEnabled()
        graph.run()
        picker.refresh()
        output = ("output", pyramid.id, "images")
        picker.source_combo.setCurrentIndex(_source_row(picker, output))
        picker.level_combo.setCurrentIndex(2)
        assert picker.capture_button.isEnabled()
        picker.capture_button.click()
        assert len(packets) == 1
        captured = packets[0]
        original = pyramid.last_result.outputs["images"].levels[2]
        np.testing.assert_array_equal(captured.value, original)
        assert captured.payload["pyramid"] == metadata(original)["pyramid"]
        assert captured.payload["spatial"] == metadata(original)["spatial"]
        captured.value[:] = 0
        assert np.any(original)

        result = pyramid.last_result
        pyramid.last_result = None
        picker.refresh()
        assert _source_row(picker, ("output", source.id, "image")) >= 0
        assert picker.source_combo.currentIndex() == -1
        assert not picker.capture_button.isEnabled()
        with pytest.raises(ValueError, match="available image source"):
            picker.snapshot()
        pyramid.last_result = result
        picker.refresh()
        assert picker.source_combo.currentData() == output
        assert picker.level_combo.currentIndex() == 2

        pyramid.parameters["levels"] = 1
        block.input_levels = {"image": "0"}
        graph.run()
        picker.refresh()
        assert picker.source_combo.currentData() == output
        assert picker.level_combo.currentIndex() == -1
        assert not picker.capture_button.isEnabled()
        assert picker.level_combo.isEnabled()
        assert len(packets) == 1
        picker.level_combo.setCurrentIndex(0)
        assert picker.snapshot().payload["pyramid"]["level"] == 0
    finally:
        picker.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not context._output_callbacks


def test_two_pickers_can_choose_independent_wired_and_processed_images(app, sources):
    graph, source, _, block = sources
    context = PluginDialogContext(graph, block, input_levels={"image": "2"})
    dialog = QDialog()
    reference = PluginImagePicker(context, dialog, input_port="image")
    inspection = PluginImagePicker(context, dialog, input_port="image")
    try:
        captured = []
        reference.captured.connect(captured.append)
        reference.capture_button.click()
        assert not captured
        assert "No connected image" in reference.status_label.text()

        graph.run()
        assert context._outputs_published()
        context._notify_outputs(lambda message: pytest.fail(message))
        assert not captured
        assert _source_row(reference, ("output", source.id, "image")) >= 0
        inspection.source_combo.setCurrentIndex(_source_row(inspection, ("processed", "", "image")))
        assert reference.snapshot().payload["pyramid"]["level"] == 2
        assert inspection.snapshot().payload["pyramid"]["level"] == 1
    finally:
        dialog.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not context._output_callbacks
