"""Exercise native tracking, lost-target behavior, and editor state handoff."""

import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QThread

from opencv_dataflow.app import MainWindow
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.sequence import share_native_sequence_runtime

TRACKERS = ("sequence.track_csrt", "sequence.track_kcf")


@pytest.mark.parametrize("kind", TRACKERS)
def test_tracker_follows_target_and_reseeds_after_loss(kind):
    registry = builtin_registry()
    source = registry.create(
        "input.sequence_demo", parameters={"stationary_frames": 2, "step_pixels": 3}
    )
    block = registry.create(kind, parameters={"initial_bounds": [20, 45, 32, 32], "max_misses": 1})
    frames = [source.execute({}).outputs["image"] for _ in range(7)]
    first = block.execute({"image": frames[0]})
    assert first is not None, block.error
    assert first.outputs["bounds"] is NO_VALUE
    assert first.outputs["status"]["phase"] == "initialized"
    for image in frames[1:]:
        result = block.execute({"image": image})
        assert result is not None, block.error
        assert result.outputs["ready"]
    x, y, w, h = result.outputs["bounds"]
    np.testing.assert_allclose([x, y, w, h], [35, 45, 32, 32], atol=2)
    assert result.outputs["tracked_region"]["rectangle"] == result.outputs["bounds"]
    assert result.outputs["pose"]["origin"] == [x + (w - 1) / 2, y + (h - 1) / 2]
    first_id = result.outputs["pose"]["track_id"]
    lost = block.execute({"image": np.zeros_like(frames[-1])})
    assert lost is not None, block.error
    assert not lost.outputs["ready"]
    assert lost.outputs["pose"] is NO_VALUE
    assert lost.outputs["status"]["phase"] == "lost"
    expired = block.execute({"image": frames[-1]})
    assert expired.outputs["status"]["reason"] == "tracker_expired"
    reseeded = block.execute({"image": frames[-1], "seed": {"bounds": [35, 45, 32, 32]}})
    assert reseeded.outputs["status"]["phase"] == "initialized"
    assert reseeded.outputs["status"]["track_id"] != first_id
    again = block.execute({"image": frames[-1]})
    assert again.outputs["ready"]
    assert again.outputs["pose"]["track_id"] == reseeded.outputs["status"]["track_id"]


def test_csrt_can_initialize_from_mask_and_checks_wired_seed_frame():
    registry = builtin_registry()
    image = registry.create("input.sequence_demo").execute({}).outputs["image"]
    mask = np.zeros_like(image)
    mask[45:77, 20:52] = 255
    block = registry.create("sequence.track_csrt")
    result = block.execute({"image": image, "mask": mask})
    assert result is not None, block.error
    assert result.outputs["status"]["phase"] == "initialized"
    assert block.execute({"image": image}).outputs["ready"]
    size = [image.shape[1], image.shape[0]]
    source = ValuePacket(image, {"spatial": frame(size, "camera-a")})
    seed = ValuePacket([20, 45, 32, 32], {"spatial": frame(size, "camera-b")})
    assert block.execute({"image": source, "seed": seed}) is None
    assert "different coordinate frames" in block.error


def test_tracker_state_survives_editor_worker_close():
    registry = builtin_registry()
    root = Graph()
    demo = root.add(registry.create("input.sequence_demo"))
    tracker = root.add(
        registry.create("sequence.track_kcf", parameters={"initial_bounds": [20, 45, 32, 32]})
    )
    root.connect(demo.id, "image", tracker.id, "image")
    try:
        for index in range(3):
            worker = Graph.from_dict(root.to_dict(), registry)
            worker.copy_runtime_from(root)
            share_native_sequence_runtime(worker, root)
            try:
                worker.run()
                result = worker.blocks[tracker.id].last_result
                assert result is not None, worker.blocks[tracker.id].error
                assert result.outputs["activations"] == index + 1
                assert result.outputs["ready"] is (index > 0)
                root.copy_runtime_from(worker)
                share_native_sequence_runtime(root, worker)
            finally:
                worker.close()
    finally:
        root.close()


@pytest.mark.local_examples
def test_tracker_example_runs_both_trackers():
    project = (
        Path(__file__).resolve().parents[1]
        / "examples/opencv tools/native_box_tracking/native_box_tracking.json"
    )
    graph = Graph.load(project, builtin_registry())
    try:
        for _ in range(8):
            graph.run()
        for key in ("csrt", "kcf"):
            result = graph.blocks[key].last_result
            assert result is not None, graph.blocks[key].error
            assert result.outputs["ready"]
            assert 40 <= result.outputs["bounds"][0] <= 47
    finally:
        graph.close()


def test_kcf_initializes_opencl_on_desktop_thread_before_worker(app, monkeypatch):
    window = MainWindow()
    graph = Graph()
    graph.add(window.registry.create("sequence.track_kcf"))
    window.replace_graph(graph)
    queried = []
    monkeypatch.setattr(cv2.ocl, "useOpenCL", lambda: queried.append(QThread.currentThread()))
    try:
        window.start_run(continuous=False)
        assert queried == [app.thread()]
        deadline = time.monotonic() + 5
        while window.runner is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert window.runner is None
    finally:
        window.close()
