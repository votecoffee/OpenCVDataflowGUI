"""Paired pose capture, support retention, and runtime replay."""

import copy
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.app import MainWindow
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/Simple Plugins/reference_inspect/fixture_capture"


def test_pose_capture_rejects_stale_frames_and_reopens_with_retained_support(app, monkeypatch, tmp_path):
    trainer_module = pytest.importorskip("opencv_simple_reference_inspect.qt_training")
    if not hasattr(trainer_module.ReferenceTrainingDialog, "_capture_pose"):
        pytest.skip("Fixture capture requires SimpleReferenceInspect with paired pose capture")
    model_io = pytest.importorskip("opencv_simple_reference_inspect")
    moved = tmp_path / "fixture_training"
    shutil.copytree(EXAMPLE, moved)
    project = moved / "project.json"
    graph = Graph.load(project, builtin_registry())
    window = MainWindow()
    warnings = []
    monkeypatch.setattr(window, "warn", warnings.append)
    try:
        window.replace_graph(graph, project)
        graph.run()
        block, fixture = graph.blocks["inspection"], graph.blocks["fixture"]
        assert block.state == fixture.state == "pass"
        original = dict(block.parameters)
        window.inspector.select(block)
        window.inspector.open_plugin_dialog()
        trainer = window.inspector._plugin_dialog
        original_model = trainer.model
        assert len(trainer.samples) == 8
        np.testing.assert_array_equal(trainer.reference_fixture, [[1, 0, 17], [0, 1, 17]])
        saved_pose_result = fixture.last_result
        stale = copy.deepcopy(saved_pose_result)
        stale.outputs["pose"]["target_payload"]["ticket"]["sequence"] += 1
        fixture.last_result = stale
        trainer.capture_pose_button.click()
        assert "different acquisitions" in trainer.status.text()
        assert trainer.model is original_model and len(trainer.samples) == 8
        foreign = copy.deepcopy(saved_pose_result)
        foreign.outputs["pose"]["target_payload"]["spatial"]["root_id"] = "different-camera"
        fixture.last_result = foreign
        trainer.capture_pose_button.click()
        assert "different coordinate roots" in trainer.status.text()
        assert trainer.model is original_model and len(trainer.samples) == 8
        fixture.last_result = saved_pose_result

        graph.run()
        assert block.state == "pass", block.error
        trainer.capture_pose_button.click()
        assert len(trainer.samples) == 9 and trainer.model is None
        support = trainer.sample_support[-1].copy()
        assert not support.all() and support[trainer.inspection_mask].all()
        trainer.view_mode.setCurrentText("Common sample coverage")
        trainer._show_model()
        assert "supported by every training sample" in trainer.status.text()
        region = trainer.inspection_mask.copy()
        trainer.inspection_mask = np.ones_like(region)
        trainer._invalidate()
        trainer.train_button.click()
        assert trainer.model is None and "does not cover both training masks" in trainer.status.text()
        trainer.inspection_mask = region
        trainer.train_button.click()
        assert trainer.model is not None, trainer.status.text()
        trainer.wired_preview_button.click()
        assert trainer.inspection[0].passed
        graph.run()
        assert block.state == "fail" and block.last_result.outputs["defect_area"] == 754
        trainer.wired_preview_button.click()
        assert not trainer.inspection[0].passed
        assert trainer.inspection[0].report["defect_area"] == 754
        assert len(trainer.samples) == 9
        trainer.stage_button.click()
        assert block.parameters == original
        window.inspector.apply()
        assert not warnings and block.parameters["recipe_path"] != original["recipe_path"]
        graph.save(project)
        saved = model_io.load_model(moved / block.parameters["recipe_path"], sha256=block.parameters["sha256"])
        np.testing.assert_array_equal(saved.reference_fixture, [[1, 0, 17], [0, 1, 17]])
        np.testing.assert_array_equal(saved.sample_support[-1], support)
        window.inspector.open_plugin_dialog()
        restored = window.inspector._plugin_dialog
        assert len(restored.samples) == len(restored.sample_support) == 9
        np.testing.assert_array_equal(restored.sample_support[-1], support)
        assert "17.00" in restored.fixture_label.text()
    finally:
        window.dirty = False
        window.close()
        app.processEvents()

    replay = Graph.load(project, builtin_registry())
    try:
        for verdict, area in (("pass", 0), ("pass", 0), ("fail", 754)):
            replay.run()
            block = replay.blocks["inspection"]
            assert block.state == verdict, block.error
            assert block.last_result.outputs["defect_area"] == area
        image = replay.blocks["source"].last_result.outputs["image"]
        block.execute({"image": image})
        assert block.state == "error" and "requires a current image pose" in block.error
        assert block.last_result is None
    finally:
        replay.close()


def test_rotated_fixture_pose_maps_into_the_original_reference_grid():
    trainer_module = pytest.importorskip("opencv_simple_reference_inspect.qt_training")
    if not hasattr(trainer_module.ReferenceTrainingDialog, "_capture_pose"):
        pytest.skip("Fixture capture requires SimpleReferenceInspect with paired pose capture")
    model_io = pytest.importorskip("opencv_simple_reference_inspect")
    integration = pytest.importorskip("opencv_simple_reference_inspect.dataflow")
    read_image = pytest.importorskip("opencv_simple_reference_inspect.cli").read_image
    from opencv_dataflow.payloads import ValuePacket, frame, image_with_payload
    from opencv_dataflow.poses import make_pose
    from opencv_dataflow.tickets import TicketClock

    model = model_io.load_model(EXAMPLE / "assets/reference.sri")
    source = read_image(EXAMPLE / "assets/nominal.png")
    transform = cv2.getRotationMatrix2D((128, 96), 5., 1.)
    current = transform @ np.vstack((model.reference_fixture, [0., 0., 1.]))
    moved = cv2.warpAffine(source, transform, source.shape[::-1])
    payload = {"spatial": frame(list(source.shape[::-1]), "rotated-camera"), "ticket": TicketClock().issue("camera")}
    image = ValuePacket(image_with_payload(moved, payload), payload)
    pose = ValuePacket(make_pose(np.vstack((current, [0., 0., 1.])), list(source.shape[::-1]), payload), payload)
    fixture = integration.fixture_for_capture(image, pose)
    captured = integration.relative_fixture(fixture, model.reference_fixture)
    runtime = integration.reference_to_image(pose.value, image.value, model.reference_fixture)
    np.testing.assert_allclose(captured, runtime)
    assert model.inspect(moved, reference_to_image=runtime).passed

def test_empty_training_teaches_its_reference_pose_and_removal_clears_it(app, monkeypatch, tmp_path):
    trainer_module = pytest.importorskip("opencv_simple_reference_inspect.qt_training")
    if not hasattr(trainer_module.ReferenceTrainingDialog, "_capture_pose"):
        pytest.skip("Fixture capture requires SimpleReferenceInspect with paired pose capture")
    moved = tmp_path / "first_capture"
    shutil.copytree(EXAMPLE, moved)
    project = moved / "project.json"
    graph = Graph.load(project, builtin_registry())
    block = graph.blocks["inspection"]
    block.parameters.update(recipe_path="", sha256="")
    window = MainWindow()
    warnings = []
    monkeypatch.setattr(window, "warn", warnings.append)
    try:
        window.replace_graph(graph, project)
        graph.run()
        assert graph.blocks["fixture"].state == "pass"
        window.inspector.select(block)
        window.inspector.open_plugin_dialog()
        trainer = window.inspector._plugin_dialog
        assert not trainer.samples and trainer.reference_fixture is None
        trainer.capture_pose_button.click()
        assert len(trainer.samples) == 1 and trainer.sample_support[0].all()
        np.testing.assert_array_equal(trainer.reference_fixture, [[1, 0, 17], [0, 1, 17]])
        with pytest.raises(ValueError, match="Use Capture image"):
            trainer._capture()
        trainer.clear_fixture_button.click()
        assert trainer.reference_fixture is None and len(trainer.samples) == 1
        trainer.capture_pose_button.click()
        assert "empty training set" in trainer.status.text() and len(trainer.samples) == 1
        trainer._remove()
        assert not trainer.samples and not trainer.sample_support and trainer.reference_fixture is None
        assert trainer.inspection_mask is trainer.normalization_mask is None
        trainer.capture_pose_button.click()
        trainer.new_set_button.click()
        assert not trainer.samples and not trainer.sample_support and trainer.reference_fixture is None
        trainer.capture_pose_button.click()
        trainer.train_button.click()
        assert trainer.model is not None, trainer.status.text()
        original = dict(block.parameters)
        trainer.reject()
        assert block.parameters == original and not warnings
        assert not (moved / "assets/reference_inspect").exists()
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
