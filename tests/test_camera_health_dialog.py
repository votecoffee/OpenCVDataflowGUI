"""Offscreen capture/training/staging checks; no desktop or hardware interaction."""

import copy

import cv2
from test_camera_health import fixture, zones

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.camera_health import CameraHealth, load_reference
from opencv_dataflow.camera_health.dialog import CameraHealthTrainer
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import ValuePacket


class Context:
    def __init__(self, project_dir):
        self.project_dir = project_dir

    def refresh_images(self):
        return []

    def subscribe_outputs(self, callback):
        return lambda *args: None


def finish_training(dialog):
    dialog.start_training()
    assert dialog.future is not None, dialog.status.text()
    dialog.future.result(timeout=30)
    dialog.poll()
    assert dialog.candidate is not None, dialog.status.text()


def test_multi_capture_worker_and_staging_changes_only_the_draft(app, tmp_path):
    original = copy.deepcopy(CameraHealth.defaults)
    draft = copy.deepcopy(original)
    dialog = CameraHealthTrainer(None, draft, Context(tmp_path))
    try:
        dialog.add_capture(ValuePacket(fixture(), {}))
        dialog.add_capture(ValuePacket(fixture(), {}))
        dialog.regions = zones()
        dialog.limits["lighting_warning"].setValue(12)
        finish_training(dialog)
        assert dialog.report.rowCount() == 4
        assert dialog.stage_button.isEnabled()
        dialog.stage()
        assert dialog.result() == dialog.DialogCode.Accepted
        assert CameraHealth.defaults == original
        assert draft["lighting_warning"] == 12 and draft["reference_path"]
        model = load_reference(
            tmp_path / draft["reference_path"], draft["reference_sha256"], include_history=True
        )
        assert model.info["capture_count"] == 2
        assert len(model.info["regions"]["focus"]) == 2
    finally:
        dialog.close()
        app.processEvents()


def test_cancel_is_private_and_bad_limits_do_not_stage(app, tmp_path):
    draft = copy.deepcopy(CameraHealth.defaults)
    dialog = CameraHealthTrainer(None, draft, Context(tmp_path))
    dialog.add_capture(ValuePacket(fixture(), {}))
    finish_training(dialog)
    dialog.limits["focus_warning"].setValue(80)
    dialog.stage()
    assert "Warning must be below alarm" in dialog.status.text()
    assert not draft["reference_path"] and not (tmp_path / "models").exists()
    dialog.reject()
    assert draft == CameraHealth.defaults
    app.processEvents()


def test_restore_capture_history_and_remove_last_capture_safely(app, tmp_path):
    draft = copy.deepcopy(CameraHealth.defaults)
    first = CameraHealthTrainer(None, draft, Context(tmp_path))
    first.add_capture(ValuePacket(fixture(), {}))
    first.regions = zones()
    finish_training(first)
    first.stage()
    digest = draft["reference_sha256"]
    reopened = CameraHealthTrainer(None, draft, Context(tmp_path))
    try:
        assert len(reopened.frames) == 1
        assert reopened.candidate is not None
        assert reopened.candidate.info["regions"] == first.candidate.info["regions"]
        reopened.stage()
        assert draft["reference_sha256"] == digest
    finally:
        reopened.close()
    third = CameraHealthTrainer(None, copy.deepcopy(draft), Context(tmp_path))
    third.remove_capture()
    assert not third.frames and third.candidate is None and not third.train_button.isEnabled()
    third.reject()
    first.close()
    app.processEvents()


def test_host_untrained_block_captures_its_wired_input_and_requires_apply(app, tmp_path):
    from opencv_dataflow.app import MainWindow

    image = tmp_path / "camera.png"
    assert cv2.imwrite(str(image), fixture())
    registry = builtin_registry()
    graph = Graph()
    source = graph.add(
        registry.create("input.file", parameters={"path": str(image), "color_format": "gray"})
    )
    block = graph.add(registry.create("inspect.camera_health"))
    graph.connect(source.id, "image", block.id, "image")
    graph.save(tmp_path / "project.json")
    graph.run()  # Source publishes; monitor is still untrained.
    window = MainWindow()
    try:
        window.replace_graph(graph)
        app.processEvents()
        inspector = window.inspector
        inspector.select(block)
        inspector.open_plugin_dialog()
        dialog = inspector._plugin_dialog
        assert isinstance(dialog, CameraHealthTrainer)
        packet = dialog.picker.snapshot()
        assert packet.value.shape == (240, 320)
        dialog.add_capture(packet)
        finish_training(dialog)
        dialog.stage()
        assert not block.parameters["reference_path"]
        assert inspector.read_parameters()["reference_path"]
        assert inspector.apply()
        assert block.parameters["reference_path"]
        graph.run()
        assert block.state == "pass", block.error
    finally:
        window.dirty = False
        window.close()
        app.processEvents()


def test_staging_rejects_reference_trained_with_other_settings(app, tmp_path):
    from opencv_dataflow.camera_health import save_reference, train_reference

    model = train_reference([fixture()], zones())
    path = tmp_path / "reference.npz"
    digest = save_reference(model, path)
    draft = {
        **CameraHealth.defaults,
        "reference_path": "reference.npz",
        "reference_sha256": digest,
        "focus_aperture": "5",
    }
    dialog = CameraHealthTrainer(None, draft, Context(tmp_path))
    try:
        assert dialog.candidate is not None
        dialog.stage()
        assert "train again" in dialog.status.text()
        assert not (tmp_path / "models").exists()
        finish_training(dialog)
        dialog.stage()
        assert dialog.result() == dialog.DialogCode.Accepted
        updated = load_reference(tmp_path / draft["reference_path"], draft["reference_sha256"])
        assert updated.info["settings"]["focus_aperture"] == "5"
    finally:
        dialog.close()
        app.processEvents()
