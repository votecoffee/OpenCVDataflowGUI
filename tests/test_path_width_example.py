import copy
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest

from opencv_dataflow.app import MainWindow
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

_EXAMPLE = Path(__file__).resolve().parents[1] / "examples/Simple Plugins/path_width"


def test_relocated_bead_project_reports_gaps_and_publishes_overlay(tmp_path):
    pytest.importorskip("opencv_simple_path_width.dataflow")
    moved = tmp_path / "moved"
    shutil.copytree(_EXAMPLE, moved)
    graph = Graph.load(moved / "project.json", builtin_registry())
    graph.filmstrip_enabled = True
    try:
        inspection = graph.blocks["inspection"]
        graph.run()
        assert inspection.state == "pass", inspection.error
        assert inspection.last_result.outputs["mean_width"] == pytest.approx(12., abs=.2)
        graph.run()
        assert inspection.state == "fail", inspection.error
        result = inspection.last_result
        assert result.outputs["longest_gap"] == pytest.approx(24., abs=2.)
        assert {item["code"] for item in result.fail_reasons} == {
            "gap_above_maximum", "station_out_of_tolerance",
        }
        assert result.outputs["trend"].shape == (292, 630, 3)
        rows = graph.filmstrip.snapshot()
        assert [(row["filename"], row["state"]) for row in rows] == [
            ("reference.png", "pass"), ("bead.png", "fail"),
        ]
        assert all(row["override"] is not None and row["thumbnail"] for row in rows)
    finally:
        graph.close()


def test_real_weld_example_replays_with_edge_search_bands():
    pytest.importorskip("opencv_simple_path_width.dataflow")
    measurement = pytest.importorskip("opencv_simple_path_width.measurement")
    if not hasattr(measurement.PathWidthSettings, "nominal_width"):
        pytest.skip("Weld example requires SimplePathWidth with edge search bands")
    graph = Graph.load(_EXAMPLE / "weld_bead/project.json", builtin_registry())
    graph.filmstrip_enabled = True
    try:
        inspection = graph.blocks["inspection"]
        for state, stations in (("pass", 31), ("pass", 31), ("fail", 18)):
            graph.run()
            assert inspection.state == state, inspection.error
            assert inspection.last_result.outputs["coverage"] == pytest.approx(stations / 31)
        assert len(graph.filmstrip.snapshot()) == 3
    finally:
        graph.close()


def test_moved_weld_project_uses_wired_affine_transform(tmp_path):
    pytest.importorskip("opencv_simple_path_width.dataflow")
    measurement = pytest.importorskip("opencv_simple_path_width.measurement")
    if not hasattr(measurement.PathWidthSettings, "edge_continuity"):
        pytest.skip("Fixture replay requires SimplePathWidth with edge continuity")
    moved = tmp_path / "moved"
    shutil.copytree(_EXAMPLE / "weld_bead/pose_replay", moved)
    graph = Graph.load(moved / "project.json", builtin_registry())
    try:
        inspection = graph.blocks["inspection"]
        graph.run()
        assert inspection.state == "pass", inspection.error
        result = inspection.last_result.outputs["measurements"]
        assert len(result["stations"]) == 31 and result["coverage"] == 1
        assert abs(result["path"][0][1] - result["path"][-1][1]) > 30

        graph.connections = [wire for wire in graph.connections if wire.input != "pose"]
        graph.run()
        assert inspection.state == "fail", inspection.error
        assert [item["code"] for item in inspection.last_result.fail_reasons] == ["station_out_of_tolerance"]
    finally:
        graph.close()


def test_calibrated_bead_reopens_with_portable_plane_profile(tmp_path):
    measurement = pytest.importorskip("opencv_simple_path_width.measurement")
    if not hasattr(measurement.PathWidthSettings, "physical_units"):
        pytest.skip("Calibrated example requires a SimplePathWidth build with physical limits")
    moved = tmp_path / "calibrated"
    shutil.copytree(_EXAMPLE / "calibrated_bead", moved)
    graph = Graph.load(moved / "project.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["width"]
        assert block.state == "pass", block.error
        outputs = block.last_result.outputs
        assert outputs["mean_width"] > 30
        assert outputs["mean_width_physical"] == pytest.approx(6., abs=.09)
        report = outputs["physical_measurements"]
        assert report["units"] == "mm" and report["path_length"] == pytest.approx(40.)
        assert report["stations"][0]["center_offset"] == pytest.approx(1.3, abs=.04)
        assert outputs["physical_trend"].shape == (292, 630, 3)
        assert outputs["longest_gap_physical"] == 0
        for issue in ("width_below_minimum", "center_out_of_tolerance"):
            graph.run()
            assert block.state == "fail", block.error
            result = block.last_result
            assert [item["code"] for item in result.fail_reasons] == ["physical_station_out_of_tolerance"]
            report = result.outputs["physical_measurements"]
            assert report["stations"][0]["issues"] == (issue,)
            overlay = result.outputs["overlay"]
            assert np.any(overlay[..., 2] > overlay[..., 1])
        block.execute({"image": graph.blocks["source"].last_result.outputs["image"]})
        assert block.state == "error" and "Physical limits require a planar calibration" in block.error
        assert block.last_result is None
    finally:
        graph.close()


def test_physical_limits_can_be_edited_and_staged_in_trainer(app, monkeypatch, tmp_path):
    trainer = pytest.importorskip("opencv_simple_path_width.qt_training")
    recipe_io = pytest.importorskip("opencv_simple_path_width.recipe")
    if not hasattr(trainer.PathWidthSettings, "physical_units"):
        pytest.skip("Physical tolerance training requires a build with physical limits")
    moved = tmp_path / "calibrated_training"
    shutil.copytree(_EXAMPLE / "calibrated_bead", moved)
    project = moved / "project.json"
    graph = Graph.load(project, builtin_registry())
    block = graph.blocks["width"]
    original_path = block.parameters["recipe_path"]
    window = MainWindow()
    warnings, previews = [], []
    monkeypatch.setattr(window, "warn", warnings.append)
    monkeypatch.setattr(trainer.MeasurementPreview, "exec", lambda self: previews.append(self.result))
    try:
        window.replace_graph(graph, project)
        window.inspector.select(block)
        window.inspector.open_plugin_dialog()
        dialog = window.inspector._plugin_dialog
        assert dialog.fields["physical_units"].text() == "mm"
        assert float(dialog.fields["min_width_physical"].text()) == 5.8
        dialog.fields["min_width_physical"].setText("5.9")
        dialog.preview_button.click()
        assert len(previews) == 1 and previews[0].physical_measurements is None
        assert "Pixel preview" in dialog.status.text()
        dialog.stage_button.click()
        assert block.parameters["recipe_path"] == original_path
        window.inspector.apply()
        assert not warnings
        assert block.parameters["recipe_path"] != original_path
        settings = recipe_io.load_recipe(moved / block.parameters["recipe_path"]).settings
        assert settings.physical_units == "mm" and settings.min_width_physical == 5.9
    finally:
        window.dirty = False
        window.close()
        app.processEvents()


def test_calibrated_preview_selects_plane_station_and_keeps_draft_uncommitted(app, monkeypatch, tmp_path):
    trainer = pytest.importorskip("opencv_simple_path_width.qt_training")
    if not hasattr(trainer.PathWidthTrainingDialog, "_use_wired_calibration"):
        pytest.skip("Calibrated preview requires SimplePathWidth with preview calibration")
    moved = tmp_path / "calibrated_preview"
    shutil.copytree(_EXAMPLE / "calibrated_bead", moved)
    project = moved / "project.json"
    graph = Graph.load(project, builtin_registry())
    block = graph.blocks["width"]
    original = dict(block.parameters)
    window = MainWindow()
    warnings, previews = [], []
    monkeypatch.setattr(window, "warn", warnings.append)

    def preview(view):
        view.show()
        app.processEvents()
        report = view.result.physical_measurements
        assert not view.result.passed and report.units == "mm"
        assert view.tabs.currentIndex() == 1
        # With perspective, the midpoint of the plane trend and the midpoint
        # of the pixel trend correspond to different image stations.
        index = min(range(len(report.stations)), key=lambda i: abs(report.stations[i].distance - 20))
        pixel_middle = min(
            range(len(view.result.stations)),
            key=lambda i: abs(view.result.stations[i].distance_px - view.result.path_length_px / 2),
        )
        assert index != pixel_middle
        position = view.physical_trend.mapFromScene(QPointF(342.5, 90.))
        QTest.mouseClick(view.physical_trend.viewport(), Qt.MouseButton.LeftButton, pos=position)
        assert view.selected_station == index
        x, y = view.result.stations[index].point
        assert view.marker.rect().center().x() == pytest.approx(x + .5)
        assert view.marker.rect().center().y() == pytest.approx(y + .5)
        assert "mm" in view.windowTitle() and "width below minimum" in view.windowTitle()
        assert "width below minimum" in view.station_details.text()
        view.tabs.setCurrentIndex(0)
        assert view.selected_station == index and view.cursor.isVisible()
        assert "px" in view.windowTitle()
        previews.append(report)

    monkeypatch.setattr(trainer.MeasurementPreview, "exec", preview)
    try:
        window.replace_graph(graph, project)
        graph.run()
        window.inspector.select(block)
        window.inspector.open_plugin_dialog()
        dialog = window.inspector._plugin_dialog
        dialog.fields["min_width_physical"].setText("6.09")
        dialog.wired_calibration_button.click()
        assert "Wired · mm" in dialog.calibration_label.text()
        dialog.preview_button.click()
        assert len(previews) == 1 and "Calibrated preview · Fail" in dialog.status.text()
        assert block.parameters == original

        dialog.clear_calibration_button.click()
        assert dialog.preview_calibration is None
        monkeypatch.setattr(dialog, "_file", lambda *_: moved / "assets/calibration.json")
        dialog.open_calibration_button.click()
        assert "calibration.json · mm" in dialog.calibration_label.text()
        dialog._capture_reference()
        assert dialog.preview_calibration is None
        assert not warnings and block.parameters == original
    finally:
        window.dirty = False
        window.close()
        app.processEvents()


def test_calibrated_curve_replays_and_trainer_can_switch_scan_direction(app, monkeypatch, tmp_path):
    trainer = pytest.importorskip("opencv_simple_path_width.qt_training")
    recipe_io = pytest.importorskip("opencv_simple_path_width.recipe")
    if not hasattr(trainer.PathWidthSettings, "scan_mode"):
        pytest.skip("Calibrated curve requires SimplePathWidth with plane-normal scans")
    moved = tmp_path / "calibrated_curve"
    shutil.copytree(_EXAMPLE / "calibrated_curve", moved)
    project = moved / "project.json"
    graph = Graph.load(project, builtin_registry())
    baseline, plane = graph.blocks["image-normal"], graph.blocks["plane-normal"]
    original_path = baseline.parameters["recipe_path"]
    window = MainWindow()
    warnings, previews = [], []
    monkeypatch.setattr(window, "warn", warnings.append)
    monkeypatch.setattr(trainer.MeasurementPreview, "exec", lambda view: previews.append(view.result))
    try:
        window.replace_graph(graph, project)
        graph.run()
        assert baseline.state == "fail" and plane.state == "pass", (baseline.error, plane.error)
        for block in (baseline, plane):
            outputs = block.last_result.outputs
            assert outputs["coverage"] == 1
            assert len(outputs["physical_measurements"]["stations"]) == 161
        window.inspector.select(baseline)
        window.inspector.open_plugin_dialog()
        dialog = window.inspector._plugin_dialog
        assert dialog.scan_mode.currentData() == "image_normal"
        dialog.scan_mode.setCurrentIndex(dialog.scan_mode.findData("plane_normal"))
        dialog.preview_button.click()
        assert not previews and "Attach a preview calibration" in dialog.status.text()
        dialog.wired_calibration_button.click()
        dialog.preview_button.click()
        assert len(previews) == 1 and previews[0].passed
        dialog.stage_button.click()
        assert baseline.parameters["recipe_path"] == original_path
        window.inspector.apply()
        assert not warnings
        saved = recipe_io.load_recipe(moved / baseline.parameters["recipe_path"])
        assert saved.settings.scan_mode == "plane_normal"
    finally:
        window.dirty = False
        window.close()
        app.processEvents()


def test_calibrated_edge_bands_replay_and_preview(app, monkeypatch, tmp_path):
    trainer = pytest.importorskip("opencv_simple_path_width.qt_training")
    recipe_io = pytest.importorskip("opencv_simple_path_width.recipe")
    if not hasattr(trainer.PathWidthSettings, "nominal_width_physical"):
        pytest.skip("Calibrated bands require SimplePathWidth with physical edge search bands")
    moved = tmp_path / "edge_bands"
    shutil.copytree(_EXAMPLE / "calibrated_curve", moved)
    project = moved / "bands-project.json"
    graph = Graph.load(project, builtin_registry())
    pixel, plane = graph.blocks["pixel-bands"], graph.blocks["physical-bands"]
    original = plane.parameters["recipe_path"]
    window = MainWindow()
    warnings, previews = [], []
    monkeypatch.setattr(window, "warn", warnings.append)
    monkeypatch.setattr(trainer.MeasurementPreview, "exec", lambda view: previews.append(view.result))
    try:
        window.replace_graph(graph, project)
        graph.run()
        assert pixel.state == "fail", pixel.error
        assert pixel.last_result.outputs["coverage"] == pytest.approx(26 / 161)
        assert plane.state == "pass", plane.error
        assert plane.last_result.outputs["coverage"] == 1.
        assert plane.last_result.outputs["mean_width_physical"] == pytest.approx(6., abs=.05)

        window.inspector.select(plane)
        window.inspector.open_plugin_dialog()
        dialog = window.inspector._plugin_dialog
        assert float(dialog.fields["nominal_width_physical"].text()) == 6.
        assert float(dialog.fields["edge_search_radius_physical"].text()) == 1.2
        assert not dialog.fields["nominal_width"].text()
        dialog.fields["edge_search_radius_physical"].setText("1.3")
        dialog.preview_button.click()
        assert not previews and "Attach a preview calibration" in dialog.status.text()
        dialog.wired_calibration_button.click()
        dialog.preview_button.click()
        assert len(previews) == 1 and previews[0].coverage == 1.
        dialog.stage_button.click()
        assert plane.parameters["recipe_path"] == original
        window.inspector.apply()
        assert not warnings and plane.parameters["recipe_path"] != original
        saved = recipe_io.load_recipe(moved / plane.parameters["recipe_path"])
        assert saved.settings.edge_search_radius_physical == 1.3
    finally:
        window.dirty = False
        window.close()
        app.processEvents()


def test_cross_scan_preview_tracks_measured_and_missing_stations(app, monkeypatch, tmp_path):
    trainer = pytest.importorskip("opencv_simple_path_width.qt_training")
    canvas = pytest.importorskip("opencv_simple_path_width.qt_canvas")
    if not hasattr(canvas.MeasurementPreview, "_number_changed"):
        pytest.skip("Cross-scan preview requires SimplePathWidth with station profiles")
    moved = tmp_path / "cross_scan"
    shutil.copytree(_EXAMPLE / "calibrated_curve", moved)
    project = moved / "bands-project.json"
    graph = Graph.load(project, builtin_registry())
    window = MainWindow()
    warnings, selections = [], []
    monkeypatch.setattr(window, "warn", warnings.append)

    def preview(view):
        view.show()
        app.processEvents()
        result = view.result
        measured = next(i for i, row in enumerate(result.stations) if row.edges is not None)
        missing = next(i for i, row in enumerate(result.stations) if row.status == "missing")
        view.station_number.setValue(measured + 1)
        assert view.selected_station == measured
        assert "width" in view.station_details.text()
        image = view.profile.image_item.pixmap().toImage()
        assert not image.isNull()
        view.tabs.setCurrentIndex(0)
        distance = result.stations[missing].distance_px
        x = canvas.PLOT_LEFT + distance / result.path_length_px * (canvas.PLOT_RIGHT - canvas.PLOT_LEFT)
        location = view.trend.mapFromScene(QPointF(x, 90.))
        QTest.mouseClick(view.trend.viewport(), Qt.MouseButton.LeftButton, pos=location)
        assert view.selected_station == missing and view.station_number.value() == missing + 1
        assert "missing" in view.station_details.text()
        assert view.profile.image_item.pixmap().toImage() != image
        line = view.scan_line.line()
        x, y = result.stations[missing].point
        assert line.center().x() == pytest.approx(x + .5)
        assert line.center().y() == pytest.approx(y + .5)
        assert line.length() == pytest.approx(2 * view.settings.scan_half_width)
        selections.extend((measured, missing))

    monkeypatch.setattr(trainer.MeasurementPreview, "exec", preview)
    try:
        window.replace_graph(graph, project)
        graph.run()
        window.inspector.select(graph.blocks["pixel-bands"])
        window.inspector.open_plugin_dialog()
        dialog = window.inspector._plugin_dialog
        dialog.wired_calibration_button.click()
        dialog.preview_button.click()
        assert len(selections) == 2 and not warnings
    finally:
        window.dirty = False
        window.close()
        app.processEvents()


def test_station_spacing_replay_and_calibrated_training_preview(app, monkeypatch, tmp_path):
    trainer = pytest.importorskip("opencv_simple_path_width.qt_training")
    recipe_io = pytest.importorskip("opencv_simple_path_width.recipe")
    if not hasattr(trainer.PathWidthSettings, "station_spacing_physical"):
        pytest.skip("Spacing comparison requires SimplePathWidth with physical station spacing")
    moved = tmp_path / "station_spacing"
    shutil.copytree(_EXAMPLE / "calibrated_bead", moved)
    project = moved / "spacing-project.json"
    graph = Graph.load(project, builtin_registry())
    pixel, plane = graph.blocks["pixel-grid"], graph.blocks["plane-grid"]
    original = plane.parameters["recipe_path"]
    window = MainWindow()
    warnings, previews = [], []
    monkeypatch.setattr(window, "warn", warnings.append)

    def preview(view):
        report = view.result.physical_measurements
        assert view.tabs.currentIndex() == 1
        assert report.station_spacing == pytest.approx(2.)
        view.select_station(.56, physical=True)
        index = view.selected_station
        assert report.stations[index].distance == pytest.approx(23.)
        x, y = view.result.stations[index].point
        assert view.marker.rect().center() == QPointF(x + .5, y + .5)
        previews.append(report)

    monkeypatch.setattr(trainer.MeasurementPreview, "exec", preview)
    try:
        window.replace_graph(graph, project)
        graph.run()
        for block in (pixel, plane):
            assert block.state == "pass", block.error
            outputs = block.last_result.outputs
            pixels = outputs["measurements"]
            physical = outputs["physical_measurements"]
            assert len(pixels["stations"]) == 16 and physical["path_length"] == pytest.approx(40.)
            intervals = np.diff([row["distance"] for row in physical["stations"]])
            if block is plane:
                assert pixels["station_spacing_px"] is None
                assert physical["station_spacing"] == pytest.approx(2.5)
                assert intervals == pytest.approx(np.full(15, 2.5))
            else:
                assert pixels["station_spacing_px"] == pytest.approx(9.09, abs=.01)
                assert physical["station_spacing"] is None and np.ptp(intervals) > .9

        window.inspector.select(plane)
        window.inspector.open_plugin_dialog()
        dialog = window.inspector._plugin_dialog
        assert float(dialog.fields["station_spacing_physical"].text()) == 2.5
        dialog.fields["station_spacing_physical"].setText("2")
        dialog.preview_button.click()
        assert not previews and "Attach a preview calibration" in dialog.status.text()
        dialog.wired_calibration_button.click()
        dialog.preview_button.click()
        assert len(previews) == 1 and plane.parameters["recipe_path"] == original
        dialog.stage_button.click()
        window.inspector.apply()
        assert not warnings and plane.parameters["recipe_path"] != original
        assert recipe_io.load_recipe(moved / plane.parameters["recipe_path"]).settings.station_spacing_physical == 2.
    finally:
        window.dirty = False
        window.close()
        app.processEvents()


def test_fixture_capture_stages_matching_pose_and_preserves_bead_offset(app, monkeypatch, tmp_path):
    trainer = pytest.importorskip("opencv_simple_path_width.qt_training")
    if not hasattr(trainer.PathWidthTrainingDialog, "_capture_fixture"):
        pytest.skip("Fixture training requires a SimplePathWidth build with fixture capture")
    moved = tmp_path / "fixture"
    shutil.copytree(_EXAMPLE / "fixture_capture", moved)
    project = moved / "project.json"
    graph = Graph.load(project, builtin_registry())
    block, fixture = graph.blocks["width"], graph.blocks["fixture"]
    original_path = block.parameters["recipe_path"]
    block.parameters["reference_fixture"] = []
    graph.save(project)
    window = MainWindow()
    warnings = []
    monkeypatch.setattr(window, "warn", warnings.append)
    try:
        window.replace_graph(graph, project)
        window.inspector.select(block)
        window.start_run(False)
        deadline = time.monotonic() + 10
        while window.runner is not None and time.monotonic() < deadline:
            app.processEvents()
            QTest.qWait(10)
        assert window.runner is None and fixture.state == "pass", fixture.error
        window.inspector.open_plugin_dialog()
        dialog = window.inspector._plugin_dialog
        original_reference = dialog.reference.copy()
        original_result = fixture.last_result
        stale_result = copy.deepcopy(original_result)
        stale_result.outputs["pose"]["target_payload"]["ticket"]["sequence"] += 1
        fixture.last_result = stale_result
        dialog.capture_fixture_button.click()
        assert "different acquisitions" in dialog.status.text()
        assert dialog.reference_fixture == []
        np.testing.assert_array_equal(dialog.reference, original_reference)

        fixture.last_result = original_result
        dialog.capture_fixture_button.click()
        expected = [[1, 0, 35], [0, 1, 49]]
        assert dialog.reference_fixture == expected
        assert block.parameters["reference_fixture"] == []
        dialog.reject()
        assert block.parameters["reference_fixture"] == []
        assert not (moved / "assets/path_width").exists()

        window.inspector.open_plugin_dialog()
        dialog = window.inspector._plugin_dialog
        dialog.capture_fixture_button.click()
        dialog.stage_button.click()
        assert block.parameters["recipe_path"] == original_path
        assert block.parameters["reference_fixture"] == []
        window.inspector.apply()
        assert not warnings
        assert block.parameters["reference_fixture"] == expected
        graph.save(project)
        window.inspector.open_plugin_dialog()
        restored = window.inspector._plugin_dialog
        assert restored.reference_fixture == expected
        assert "x 35.0, y 49.0" in restored.fixture_label.text()
        restored.clear_fixture_button.click()
        assert restored.reference_fixture == []
        restored.reject()
        assert block.parameters["reference_fixture"] == expected
    finally:
        window.dirty = False
        window.close()
        app.processEvents()

    replay = Graph.load(project, builtin_registry())
    try:
        inspection = replay.blocks["width"]
        for verdict, offset in (("pass", .5), ("pass", .5), ("fail", 4.5)):
            replay.run()
            assert inspection.state == verdict, inspection.error
            result = inspection.last_result.outputs["measurements"]
            assert result["coverage"] == 1
            assert result["stations"][0]["center_offset_px"] == pytest.approx(offset, abs=.05)
    finally:
        replay.close()


def test_train_preview_stage_apply_and_restore(app, monkeypatch, tmp_path):
    trainer = pytest.importorskip("opencv_simple_path_width.qt_training")
    original = pytest.importorskip("opencv_simple_path_width.recipe")
    moved = tmp_path / "training"
    shutil.copytree(_EXAMPLE, moved)
    project = moved / "project.json"
    graph = Graph.load(project, builtin_registry())
    block = graph.blocks["inspection"]
    original_path = block.parameters["recipe_path"]
    reference = original.load_reference(moved / original_path)
    window = MainWindow()
    warnings = []
    monkeypatch.setattr(window, "warn", warnings.append)
    previews = []

    def preview(self):
        self.show()
        app.processEvents()
        position = self.trend.mapFromScene(QPointF(343., 90.))
        QTest.mouseClick(self.trend.viewport(), Qt.MouseButton.LeftButton, pos=position)
        assert self.marker.rect().width() == 12
        assert self.cursor.isVisible()
        assert self.windowTitle().startswith("Station ")
        previews.append(self.result)

    monkeypatch.setattr(trainer.MeasurementPreview, "exec", preview)
    try:
        window.replace_graph(graph, project)
        window.inspector.select(block)
        window.start_run(False)
        deadline = time.monotonic() + 10
        while window.runner is not None and time.monotonic() < deadline:
            app.processEvents()
            QTest.qWait(10)
        assert window.runner is None
        window.inspector.open_plugin_dialog()
        assert not warnings
        dialog = window.inspector._plugin_dialog
        app.processEvents()
        np.testing.assert_array_equal(dialog.reference, reference)
        dialog._capture_reference()
        np.testing.assert_array_equal(dialog.reference, reference)
        canvas = dialog.canvas
        vertices = np.asarray(canvas.vertices)
        index = int(np.argmax(np.linalg.norm(np.diff(vertices, axis=0), axis=1)))
        midpoint = (vertices[index] + vertices[index + 1]) / 2

        def viewport_point(point):
            return canvas.mapFromScene(QPointF(*point + .5))

        QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.ShiftModifier, viewport_point(midpoint))
        assert len(canvas.vertices) == len(vertices) + 1
        inserted = canvas.selected
        adjusted = midpoint + [0., 2.]
        QTest.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton, pos=viewport_point(midpoint))
        QTest.mouseMove(canvas.viewport(), viewport_point(adjusted))
        QTest.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton, pos=viewport_point(adjusted))
        np.testing.assert_allclose(canvas.vertices[inserted], adjusted, atol=1.)
        QTest.keyClick(canvas, Qt.Key.Key_Delete)
        np.testing.assert_allclose(canvas.vertices, vertices)
        canvas.reverse()
        dialog.preview_button.click()
        assert len(previews) == 1
        dialog.stage_button.click()
        assert not warnings
        assert block.parameters["recipe_path"] == original_path
        window.inspector.apply()
        assert not warnings
        assert block.parameters["recipe_path"] != original_path
        graph.save(project)
        window.inspector.open_plugin_dialog()
        restored = window.inspector._plugin_dialog
        assert not warnings
        np.testing.assert_array_equal(restored.reference, reference)
        np.testing.assert_allclose(restored.canvas.vertices, np.vstack((vertices[0], vertices[:0:-1])))
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
