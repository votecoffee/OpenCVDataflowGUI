"""Large workspaces, finite image sources and actual ticket verdict filmstrips."""

import copy
import json
import time
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, QSize, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QFileDialog, QPushButton

from opencv_dataflow.app import MainWindow
from opencv_dataflow.blocks import ImageFile, builtin_registry, demo_graph
from opencv_dataflow.canvas import CanvasScene, CanvasView
from opencv_dataflow.core import Graph
from opencv_dataflow.filmstrip import Filmstrip
from opencv_dataflow.filmstrip_viewer import FilmstripViewer
from opencv_dataflow.queued_execution import QueuedExecution
from opencv_dataflow.tickets import ticket_key
from opencv_dataflow.toolblocks import encapsulate
from opencv_dataflow.validation import run_set
from opencv_dataflow.workspace import add_group


def pump(app, predicate):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        QTest.qWait(10)
    raise AssertionError("Worker did not finish")


def images(tmp_path):
    paths = []
    for name, value in (("z.png", 50), ("a.png", 200), ("b.png", 125)):
        path = tmp_path / name
        assert cv2.imwrite(str(path), np.full((20, 30, 3), value, np.uint8))
        paths.append(str(path))
    return paths


def file_graph(selected_paths, **parameters):
    graph = Graph()
    source = graph.add(
        ImageFile(parameters={"source_mode": "files", "paths": selected_paths, **parameters})
    )
    graph.filmstrip_enabled = True
    return graph, source


def test_auto_arrange_places_branches_and_groups_without_rewiring(app):
    registry = builtin_registry()
    graph = Graph()
    merge = graph.add(registry.create("cv.blend"))
    branch_a = graph.add(registry.create("cv.gray"))
    branch_b = graph.add(registry.create("cv.gray"))
    source = graph.add(registry.create("input.demo"))
    independent = graph.add(registry.create("input.demo"))
    branch_a.node_size = "large"
    branch_b.node_size = "small"
    for block in (branch_a, branch_b):
        graph.connect(source.id, "image", block.id, "image")
    graph.connect(branch_a.id, "image", merge.id, "image_a")
    graph.connect(branch_b.id, "image", merge.id, "image_b")
    group = add_group(graph, [source.id, branch_a.id], "Acquisition")
    graph.run()
    results = {key: block.last_result for key, block in graph.blocks.items()}
    connections = list(graph.connections)
    scene = CanvasScene(graph)
    updates = []
    scene.layout_changed.connect(lambda: updates.append(True))
    scene.set_locked(True)
    assert not scene.auto_arrange()
    scene.set_locked(False)
    assert scene.auto_arrange()
    assert not scene.auto_arrange()
    assert updates == [True]

    bounds = {key: node.sceneBoundingRect() for key, node in scene.nodes.items()}
    for left, right in combinations(bounds.values(), 2):
        assert not left.intersects(right)
    for wire in scene.wires:
        assert bounds[wire.connection.source].right() < bounds[wire.connection.target].left()
        assert wire.path().pointAtPercent(0) == wire.source.scenePos()
        assert wire.path().pointAtPercent(1) == wire.target.scenePos()
    assert branch_a.position[0] == branch_b.position[0]
    group_bounds = scene.groups[group["id"]].sceneBoundingRect()
    assert group_bounds.contains(bounds[source.id])
    assert group_bounds.contains(bounds[branch_a.id])
    assert not group_bounds.intersects(bounds[branch_b.id])
    assert bounds[independent.id].top() > max(
        bounds[key].bottom() for key in bounds if key != independent.id
    )
    assert graph.connections == connections and graph.groups == [group]
    assert all(block.last_result is results[key] for key, block in graph.blocks.items())


def test_workspace_auto_arrange_menu_saves_only_the_open_toolblock_layout(app, tmp_path):
    window = MainWindow()
    graph = window.graph
    ids = list(graph.blocks)
    tool = encapsulate(graph, ids[1:3], "Preprocess", window.registry)
    root_positions = {key: list(block.position) for key, block in graph.blocks.items()}
    window.enter_toolblock(tool)
    window.show()
    try:
        workspace = next(
            action.menu() for action in window.menuBar().actions()
            if action.text() == "&Workspace"
        )
        action = next(
            action for action in workspace.actions() if action.text() == "Auto arrange blocks"
        )
        action.trigger()
        assert window.dirty
        assert {key: block.position for key, block in graph.blocks.items()} == root_positions
        assert tool.graph.blocks[ids[1]].position[0] == 0
        assert tool.graph.blocks[ids[2]].position[0] > 0
        window.path = tmp_path / "arranged.json"
        assert window.save_project()
        restored = Graph.load(window.path, window.registry)
        assert restored.to_dict() == graph.to_dict()
    finally:
        window.dirty = False
        window.close()


def test_group_moves_members_wires_without_invalidating_and_roundtrips(app, tmp_path):
    graph = demo_graph(builtin_registry())
    graph.run()
    scene = CanvasScene(graph)
    keys = list(graph.blocks)[:2]
    for key in keys:
        scene.nodes[key].setSelected(True)
    before = {k: list(b.position) for k, b in graph.blocks.items()}
    results = {k: b.last_result for k, b in graph.blocks.items()}
    connections = list(graph.connections)
    group = scene.group_selected("Acquisition")
    item = scene.groups[group["id"]]
    item.setPos(8000, -6000)
    for key in keys:
        assert graph.blocks[key].position == [before[key][0] + 8000, before[key][1] - 6000]
        assert item.sceneBoundingRect().contains(scene.nodes[key].sceneBoundingRect())
    assert all(b.last_result is results[k] for k, b in graph.blocks.items())
    assert graph.connections == connections
    assert scene.sceneRect().contains(item.sceneBoundingRect())
    for wire in scene.wires:
        assert wire.path().pointAtPercent(0) == wire.source.scenePos()
    graph.viewport = {"center": [8050.0, -6000.0], "zoom": 0.2}
    path = tmp_path / "workspace.json"
    graph.save(path)
    restored = Graph.load(path, builtin_registry())
    assert restored.to_dict() == graph.to_dict()
    scene.clearSelection()
    item.setSelected(True)
    scene.delete_selected()
    assert not graph.groups and len(graph.blocks) == len(before)
    assert all(b.last_result is results[k] for k, b in graph.blocks.items())


@pytest.mark.parametrize(
    "bad",
    [
        [{"id": "g", "name": "Bad", "members": ["missing"]}],
        [{"id": "g", "name": "Bad", "members": ["a", "a"]}],
        [{"id": "g", "name": "Bad", "members": []}],
        [{"id": "g", "name": "", "members": ["a"]}],
        [{"id": "g", "name": "Bad", "members": [[]]}],
    ],
)
def test_invalid_groups_rejected(bad):
    graph = Graph()
    graph.add(ImageFile(block_id="a"))
    data = graph.to_dict()
    data["groups"] = bad
    with pytest.raises(ValueError):
        Graph.from_dict(data, builtin_registry())


@pytest.mark.parametrize(
    "viewport",
    [
        None,
        [],
        {"zoom": 1},
        {"center": [0, 0], "zoom": 0},
        {"center": [float("nan"), 0], "zoom": 1},
    ],
)
def test_invalid_viewport_rejected(viewport):
    graph = Graph()
    data = graph.to_dict()
    data["viewport"] = viewport
    with pytest.raises(ValueError):
        Graph.from_dict(data, builtin_registry())


def test_groups_follow_toolblock_encapsulation():
    graph = demo_graph(builtin_registry())
    keys = list(graph.blocks)[:2]
    group = add_group(graph, keys, "Inner")
    block = encapsulate(graph, keys, "Reusable", builtin_registry())
    assert not graph.groups
    assert block.graph.groups == [group]
    Graph.from_dict(graph.to_dict(), builtin_registry())


def test_viewport_overview_navigation_and_initial_view_is_clean(app):
    window = MainWindow()
    window.show()
    app.processEvents()
    assert not window.dirty
    graph = window.graph
    key = next(iter(graph.blocks))
    window.scene.nodes[key].setPos(20000, -15000)
    window.view.fit_graph()
    app.processEvents()
    assert window.view.transform().m11() < 0.2
    overview = window.view.overview
    assert overview.isVisible()
    workspace = window.centralWidget()
    for width, sizes in ((1500, [210, 940, 350]), (1200, [180, 540, 480])):
        window.resize(width, 920)
        workspace.setSizes(sizes)
        app.processEvents()
        corner = overview.mapTo(workspace, QPoint(overview.width(), 0))
        assert 0 <= workspace.width() - corner.x() <= 4
        assert 0 <= corner.y() <= 4
        assert overview.mapTo(window, QPoint(0, overview.height())).y() <= (
            window.inspector.mapTo(window, QPoint()).y()
        )
    before_navigation = copy.deepcopy(graph.viewport)
    QTest.mouseClick(overview, Qt.MouseButton.LeftButton, pos=QPoint(160, 50))
    assert graph.viewport != before_navigation and window.dirty
    saved = copy.deepcopy(graph.viewport)
    window.view.centerOn(0, 0)
    window.view.restore_view()
    assert graph.viewport == saved
    assert (
        abs(window.view.mapToScene(window.view.viewport().rect().center()).x() - saved["center"][0])
        < 100
    )
    window.dirty = False
    window.close()


def test_file_selection_order_filename_tickets_cursor_and_reset(tmp_path):
    paths = images(tmp_path)
    graph, source = file_graph(paths, color_format="gray", pixel_depth="uint16")
    tickets = []
    for path in paths:
        graph.run()
        assert source.state == "pass", source.error
        assert source.last_result.outputs["info"]["filename"] == Path(path).name
        assert source.last_result.outputs["image"].shape == (20, 30)
        assert source.last_result.outputs["image"].dtype == np.uint16
        tickets.append(source.last_result.outputs["ticket"])
    assert [t["value"] for t in tickets] == [Path(p).name for p in paths]
    assert graph.sources_exhausted()
    graph.run()
    assert not source.current_emissions.get("image")
    assert len(graph.filmstrip.snapshot()) == 3
    assert graph.ticket_results.statistics()["total"] == 3
    graph.invalidate()
    graph.run()
    assert source.last_result.outputs["info"]["filename"] == "z.png"


def test_directory_snapshot_recursion_and_project_relative_paths(tmp_path):
    images(tmp_path)
    (tmp_path / "readme.txt").write_text("ignore")
    sub = tmp_path / "sub"
    sub.mkdir()
    images(sub)
    graph, source = file_graph([], source_mode="directory", directory=".", path_base="project")
    graph.bind_project_dir(tmp_path)
    names = []
    while not graph.sources_exhausted():
        graph.run()
        names.append(source.last_result.outputs["info"]["filename"])
        if len(names) == 1:
            cv2.imwrite(str(tmp_path / "new.png"), np.zeros((2, 2), np.uint8))
    assert names == ["a.png", "b.png", "z.png"]
    source.parameters["recursive"] = True
    graph.invalidate()
    graph.run()
    assert len(source._sequence["files"]) == 7


def test_duplicate_names_loop_and_explicit_external_ticket(tmp_path):
    paths = images(tmp_path)
    graph, source = file_graph([paths[0], paths[0]], repeat="loop")
    tickets = []
    for _ in range(4):
        graph.run()
        tickets.append(source.last_result.outputs["ticket"])
    assert len({ticket_key(t) for t in tickets}) == 4
    assert {t["value"] for t in tickets} == {"z.png"}
    assert not graph.sources_exhausted()
    source.ticket_settings = {"mode": "fixed", "value": "PLC-42"}
    graph.run()
    assert source.last_result.outputs["ticket"]["value"] == "PLC-42"


def test_corrupt_image_records_error_and_continues(tmp_path):
    paths = images(tmp_path)
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"not an image")
    graph, source = file_graph([str(bad), paths[0]])
    graph.run()
    assert source.state == "error"
    assert graph.filmstrip.snapshot()[0]["state"] == "error"
    graph.run()
    assert source.state == "pass"
    assert [r["state"] for r in graph.filmstrip.snapshot()] == ["error", "pass"]


@pytest.mark.parametrize(
    "parameters",
    [
        {"source_mode": "files", "paths": []},
        {"source_mode": "directory", "directory": "missing"},
        {"source_mode": "files", "paths": [1]},
    ],
)
def test_invalid_sequences_fail_once_and_finish(parameters):
    graph, source = file_graph([], **parameters)
    graph.run()
    assert source.state == "error"
    assert graph.sources_exhausted()


def test_source_runtime_copy_resumes_instead_of_replaying(tmp_path):
    paths = images(tmp_path)
    graph, source = file_graph(paths)
    graph.run()
    clone = Graph.from_dict(graph.to_dict(), builtin_registry())
    clone.copy_runtime_from(graph)
    clone.run()
    assert clone.blocks[source.id].last_result.outputs["info"]["filename"] == "a.png"
    graph.copy_source_runtime_from(clone)
    graph.run()
    assert source.last_result.outputs["info"]["filename"] == "b.png"


def test_background_queue_drains_finite_source_without_hanging(tmp_path):
    paths = images(tmp_path)
    graph, source = file_graph(paths)
    registry = builtin_registry()
    queue = graph.add(
        registry.create(
            "flow.image_queue", parameters={"mode": "background_acquisition", "capacity": 2}
        )
    )
    graph.connect(source.id, "image", queue.id, "image")
    names = []
    with QueuedExecution(graph, registry) as runner:
        while runner.run_one():
            names.append(source.last_result.outputs["info"]["filename"])
    assert names == ["z.png", "a.png", "b.png"]
    assert source.exhausted
    assert len(graph.filmstrip.snapshot()) == 3
    assert graph.ticket_results.statistics()["total"] == 3


@pytest.mark.local_examples
def test_filmstrip_actual_verdicts_bounded_and_validation(app):
    registry = builtin_registry()
    graph = Graph.load("examples/demos/validation_images/validation_images.json", registry)
    strip = Filmstrip(capacity=3)
    report = run_set(
        graph.to_dict(), graph.validation_sets[0], registry, graph.project_dir, filmstrip=strip
    )
    assert report["summary"]["score_percent"] == 100
    assert [r["state"] for r in strip.snapshot()] == ["pass", "fail", "fail"]
    assert strip.dropped == 2
    assert all(len(r["thumbnail"]) > 0 for r in strip.snapshot())
    viewer = FilmstripViewer(lambda: strip)
    viewer.refresh()
    assert viewer.items.count() == 3
    assert viewer.items.item(1).data(Qt.ItemDataRole.UserRole) == "fail"
    assert "fail.png" in viewer.items.item(1).toolTip()
    assert not viewer.items.item(1).icon().isNull()
    small = Filmstrip(max_bytes=1)
    small.add(strip.snapshot()[0]["ticket"], "image", np.zeros((4, 4), np.uint8))
    assert small.snapshot() == [] and small.dropped == 1
    viewer.close()


def test_filmstrip_double_click_zoom_and_thumbnail_fallback(app, tmp_path):
    path = tmp_path / "detail.png"
    source_image = np.full((480, 640, 3), (20, 150, 250), np.uint8)
    assert cv2.imwrite(str(path), source_image)
    graph, _ = file_graph([str(path)], color_format="gray")
    graph.run()
    viewer = FilmstripViewer(lambda: graph.filmstrip)
    viewer.resize(460, 260)
    viewer.show()
    viewer.refresh()
    app.processEvents()

    def double_click():
        item = viewer.items.item(0)
        viewport = viewer.items.viewport()
        center = viewer.items.visualItemRect(item).center()
        QTest.mouseClick(viewport, Qt.MouseButton.LeftButton, pos=center)
        QTest.mouseDClick(viewport, Qt.MouseButton.LeftButton, pos=center)
        assert viewer.popout is not None
        app.processEvents()
        return viewer.popout

    try:
        dialog = double_click()
        assert dialog.isVisible() and "Ticket detail.png" in dialog.windowTitle()
        full = dialog.view.image_item.pixmap()
        assert full.size() == QSize(640, 480)
        pixel = full.toImage().pixelColor(0, 0)
        assert pixel.red() == pixel.green() == pixel.blue()
        assert "Reloaded source image" in dialog.source_label.text()

        buttons = {b.text(): b for b in dialog.controls.findChildren(QPushButton)}
        buttons["100%"].click()
        assert dialog.view.transform().m11() == 1
        viewport = dialog.view.viewport()
        point = viewport.rect().center()
        wheel = QWheelEvent(
            QPointF(point), QPointF(viewport.mapToGlobal(point)), QPoint(), QPoint(0, 120),
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        app.sendEvent(viewport, wheel)
        assert dialog.view.transform().m11() > 1
        buttons["Fit"].click()
        old_scale = dialog.view.transform().m11()
        dialog.resize(980, 760)
        app.processEvents()
        assert dialog.view.transform().m11() > old_scale

        graph.filmstrip.update_results(SimpleNamespace(verdict=lambda ticket: ("fail", ["test"])))
        viewer.refresh()
        assert dialog.windowTitle().endswith("FAIL")
        assert double_click() is dialog
        dialog.close()
        assert viewer.popout is None

        # The stored thumbnail continues to represent the result if a file was overwritten.
        assert cv2.imwrite(str(path), np.zeros_like(source_image))
        fallback = double_click()
        assert fallback.view.image_item.pixmap().height() == 100
        assert "Captured thumbnail" in fallback.source_label.text()
        fallback.close()
    finally:
        if viewer.popout is not None:
            viewer.popout.close()
        viewer.close()
        graph.close()


def test_filmstrip_captures_multipage_tiff_images(app, tmp_path):
    path = tmp_path / "pages.tiff"
    first = np.full((32, 40), 80, np.uint8)
    second = np.full_like(first, 160)
    ok, encoded = cv2.imencodemulti(".tiff", [first, second])
    assert ok
    path.write_bytes(encoded.tobytes())
    graph = Graph()
    graph.add(builtin_registry().create("input.tiff_file", parameters={"path": str(path)}))
    graph.filmstrip_enabled = True
    viewer = FilmstripViewer(lambda: graph.filmstrip)
    try:
        graph.run()
        graph.run()
        viewer.refresh()
        assert viewer.items.count() == 2
        viewer.open_image(viewer.items.item(1))
        assert "page-000001" in viewer.popout.windowTitle()
        assert viewer.popout.view.image_item.pixmap().toImage().pixelColor(0, 0).red() == 160
    finally:
        if viewer.popout is not None:
            viewer.popout.close()
        viewer.close()
        graph.close()


def test_gui_multifile_picker_continuous_and_lean_opt_in(app, tmp_path, monkeypatch):
    paths = images(tmp_path)
    graph, source = file_graph(paths)
    window = MainWindow()
    window.replace_graph(graph)
    window.show()
    app.processEvents()
    window.scene.nodes[source.id].setSelected(True)
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *a, **k: (paths, ""))
    window.inspector.browse_images()
    assert json.loads(window.inspector.fields["paths"].text()) == paths
    assert window.inspector.fields["source_mode"].currentText() == "files"
    window.start_run(True)
    pump(app, lambda: window.runner is None)
    window.filmstrip_view.refresh()
    assert window.filmstrip_view.items.count() == 3
    assert not window.filmstrip_dock.isHidden()
    window.change_run_profile("lean")
    assert not window.filmstrip_view.capture.isChecked()
    window.root_graph.invalidate()
    window.root_graph.filmstrip.clear()
    window.start_run(True)
    pump(app, lambda: window.runner is None)
    assert window.root_graph.filmstrip.snapshot() == []
    window.dirty = False
    window.close()


def test_group_header_drag_and_locked_layout(app):
    graph = demo_graph(builtin_registry())
    scene = CanvasScene(graph)
    keys = list(graph.blocks)[:2]
    for key in keys:
        scene.nodes[key].setSelected(True)
    group = scene.group_selected("Move together")
    item = scene.groups[group["id"]]
    view = CanvasView(scene)
    view.resize(1100, 650)
    view.show()
    view.fit_graph()
    app.processEvents()
    start = view.mapFromScene(item.sceneBoundingRect().topLeft() + QPointF(50, 20))
    before = {k: scene.nodes[k].pos() for k in keys}
    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(view.viewport(), start + QPoint(80, 40))
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=start + QPoint(80, 40))
    delta = scene.nodes[keys[0]].pos() - before[keys[0]]
    assert delta.manhattanLength() > 50
    assert scene.nodes[keys[1]].pos() - before[keys[1]] == delta
    scene.set_locked(True)
    assert not item.flags() & item.GraphicsItemFlag.ItemIsMovable
    scene.ungroup_selected()
    assert len(graph.groups) == 1
    view.close()


def test_buffered_ticket_thumbnails_update_when_batch_finishes(tmp_path):
    paths = images(tmp_path)
    graph, source = file_graph(paths)
    registry = builtin_registry()
    batch = graph.add(registry.create("flow.batch", parameters={"size": 2}))
    graph.connect(source.id, "image", batch.id, "item_1")
    graph.run()
    assert [r["state"] for r in graph.filmstrip.snapshot()] == ["pending"]
    graph.run()
    assert [r["state"] for r in graph.filmstrip.snapshot()] == ["pass", "pass"]
    graph.run()
    assert [r["state"] for r in graph.filmstrip.snapshot()] == ["pass", "pass", "pending"]


@pytest.mark.local_examples
def test_image_set_example_cli_finite_guard_and_report(tmp_path):
    from opencv_dataflow.examples import image_set_workspace_graph
    from opencv_dataflow.runner import main

    registry = builtin_registry()
    graph = Graph.load("examples/demos/image_set_workspace/image_set_workspace.json", registry)
    assert graph.to_dict() == image_set_workspace_graph(registry).to_dict()
    receipt = tmp_path / "receipt.json"
    assert (
        main(
            [
                "examples/demos/image_set_workspace/image_set_workspace.json",
                "--all-images",
                "--report",
                str(receipt),
            ]
        )
        == 1
    )
    report = json.loads(receipt.read_text())
    assert len(report["runs"]) == 3
    assert report["ticket_results"]["production_statistics"]["total"] == 3
    ordinary = tmp_path / "ordinary.json"
    demo_graph(registry).save(ordinary)
    assert main([str(ordinary), "--all-images"]) == 2


@pytest.mark.local_examples
def test_directory_picker_and_validation_filmstrip_worker(app, monkeypatch):
    from opencv_dataflow.validation_editor import ValidationManager

    window = MainWindow()
    registry = window.registry
    window.replace_graph(
        Graph.load("examples/demos/validation_images/validation_images.json", registry)
    )
    source = window.graph.blocks["validation_source"]
    window.scene.nodes[source.id].setSelected(True)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: "C:/images")
    window.inspector.browse_directory()
    assert window.inspector.fields["source_mode"].currentText() == "directory"
    assert window.inspector.fields["directory"].text() == "C:/images"
    manager = ValidationManager(window)
    manager.run_images()
    pump(app, lambda: manager.worker is None)
    manager.filmstrip_view.refresh()
    assert manager.report["summary"]["score_percent"] == 100
    assert manager.filmstrip_view.items.count() == 5
    manager.close()
    window.dirty = False
    window.close()


@pytest.mark.parametrize("zoom", [0.005, 0.05, 1.0])
def test_sparse_large_workspace_restores_distant_viewport(app, zoom):
    graph = demo_graph(builtin_registry())
    graph.viewport = {"center": [75000.0, -100000.0], "zoom": zoom}
    scene = CanvasScene(graph)
    view = CanvasView(scene)
    view.resize(1000, 600)
    view.show()
    app.processEvents()
    view.restore_view()
    center = view.mapToScene(view.viewport().rect().center())
    assert abs(center.x() - 75000) < 2 / zoom
    assert abs(center.y() + 100000) < 2 / zoom
    view.close()


def test_group_and_members_selected_do_not_double_move(app):
    graph = demo_graph(builtin_registry())
    scene = CanvasScene(graph)
    keys = list(graph.blocks)[:2]
    for key in keys:
        scene.nodes[key].setSelected(True)
    group = scene.group_selected("Group")
    item = scene.groups[group["id"]]
    view = CanvasView(scene)
    view.resize(1100, 650)
    view.show()
    view.fit_graph()
    app.processEvents()
    # A full-area rubber band can select the dashed border and its members.
    for node in scene.nodes.values():
        node.setSelected(True)
    before = {k: n.pos() for k, n in scene.nodes.items()}
    start = view.mapFromScene(item.sceneBoundingRect().topLeft() + QPointF(50, 20))
    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(view.viewport(), start + QPoint(80, 40))
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=start + QPoint(80, 40))
    deltas = [n.pos() - before[k] for k, n in scene.nodes.items()]
    assert deltas[0].manhattanLength() > 50
    assert all(d == deltas[0] for d in deltas)
    view.close()
