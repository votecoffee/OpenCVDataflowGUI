"""Offscreen Qt interaction checks: ports, inspector, and worker lifecycle."""

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog, QInputDialog

from opencv_dataflow.app import MainWindow
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.canvas import CanvasScene, CanvasView
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.kernel_editor import KernelEditor
from opencv_dataflow.toolblock_editor import InterfaceDialog
from opencv_dataflow.toolblocks import encapsulate


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
        assert "LIVE PREVIEW" in inspector.details.toPlainText()
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
    "loader,count,shape_id",
    [
        ("load_regions_example", 10, "roi_circle"),
        ("load_combined_regions_example", 11, "combined_outer"),
        ("load_transformed_regions_example", 12, "tracked_shape"),
    ],
)
def test_regions_example_editor_load_save_and_shape_dropdown(
    app, tmp_path, loader, count, shape_id
):
    from PySide6.QtWidgets import QComboBox

    window = MainWindow()
    window.show()
    app.processEvents()

    getattr(window, loader)()
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


def test_channel_pyramid_editor_preview_level_selection_and_save(app, tmp_path):
    window = MainWindow()
    window.load_channel_pyramid_example()
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
    window.inspector.apply()
    assert errors and child.parameters["image_paths"] == ["a.png", "b.png", "c.png"]
    window.dirty = False
    window.close()


def test_starter_project_editor_run_edit_save_and_template_example(app, tmp_path):
    window = MainWindow()
    window.show()
    window.load_starter()
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
    window.load_template_example()
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
    assert source.parameters["path"] == "assets/lena.jpg"
    window.dirty = False
    window.close()


def test_layout_only_moves_persist_root_and_nested_without_rewiring_or_result_loss(app, tmp_path):
    window = MainWindow()
    graph = window.graph
    original = list(graph.blocks.values())
    toolblock = encapsulate(graph, [original[1].id, original[2].id], "Preprocess", window.registry)
    window.scene.rebuild()
    window.start_run(False)
    pump_until(app, lambda: window.runner is None)
    root_wires = list(graph.connections)
    nested_wires = list(toolblock.graph.connections)
    window.scene.nodes[toolblock.id].setPos(235.5, -128.25)
    assert window.dirty and toolblock.state == "pass"
    assert graph.connections == root_wires
    window.enter_toolblock(toolblock)
    child = next(iter(toolblock.graph.blocks.values()))
    window.scene.nodes[child.id].setPos(-61.25, 290.75)
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
    assert restored.connections == root_wires
    assert restored.blocks[toolblock.id].graph.connections == nested_wires
    window.replace_graph(restored, window.path)
    assert window.scene.nodes[toolblock.id].pos().x() == 235.5
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
    window.inspector.apply()
    assert errors and block.parameters["score_type"] == "obsolete"
    window.dirty = False
    window.close()


def test_object_collection_editor_filter_sort_preview_worker_and_save(app, tmp_path):
    from PySide6.QtWidgets import QComboBox

    window = MainWindow()
    window.load_object_collections_example()
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
