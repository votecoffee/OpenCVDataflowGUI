"""Exercise silhouette tracking using decoded video and independent metric truth."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.meshes import TriangleMesh
from opencv_dataflow.payloads import NO_VALUE, ValuePacket

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/silhouette_tracking"


@pytest.fixture
def replay():
    graph = Graph.load(EXAMPLE / "silhouette_tracking.json", builtin_registry())
    try:
        yield graph
    finally:
        graph.close()


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_replay_tracks_all_frames_in_metric_camera_coordinates(replay, profile):
    truth = json.loads((EXAMPLE / "assets/ground_truth.json").read_text(encoding="utf-8"))
    replay.set_run_profile(profile)
    tracker = replay.blocks["tracker"]
    if profile == "debug":
        tracker.exposed_outputs.append("result_graphics")
    for index, expected in enumerate(truth["metric_poses"][:12]):
        replay.run()
        result = tracker.last_result
        assert result and result.passed and result.outputs["tracked"], tracker.error
        pose, report = result.outputs["pose"], result.outputs["report"]
        assert pose["units"] == "m"
        assert pose["target_frame"] == truth["camera_frame"]
        assert pose["source_frame"] == truth["object_frame"]
        assert result.payloads["pose"]["ticket"]["value"] == f"part_motion.avi#{index:06d}"
        assert result.payloads["pose"]["video_frame"]["frame_index"] == index
        assert report["status"] == "tracked"
        assert report["seeded_this_frame"] == (index == 0)
        assert report["reprojection_rms_px"] < 0.25
        assert 0.9 <= result.score <= 1
        matrix, expected = np.asarray(pose["source_to_target"]), np.asarray(expected)
        assert 1000 * np.linalg.norm(matrix[:3, 3] - expected[:3, 3]) < 2
        delta = cv2.Rodrigues(matrix[:3, :3] @ expected[:3, :3].T)[0]
        assert np.rad2deg(np.linalg.norm(delta)) < 0.8
        if profile == "debug":
            assert set(result.drawings) == {"Tracked mesh", "Matched edges"}
            assert (
                result.outputs["result_graphics"]
                .layers[0]
                .payload["ticket"]["value"]
                .endswith(f"#{index:06d}")
            )
        else:
            assert not result.drawings
    assert replay.sources_exhausted()


@pytest.mark.local_examples
def test_blank_frame_disarms_pose_output_until_reset_and_worker_copy_preserves_loss(replay):
    tracker = replay.blocks["tracker"]
    replay.blocks["video"].parameters["end_frame"] = -1
    for _ in range(12):
        replay.run()
        assert tracker.last_result and tracker.last_result.passed
    replay.run()  # The saved video continues with a blank frame.
    assert tracker.last_result.outputs["report"]["status"] == "insufficient_edges"
    assert tracker.last_result.fail_reasons[0]["code"] == "silhouette_insufficient_edges"
    assert tracker.last_result.outputs["pose"] is NO_VALUE

    worker = builtin_registry().create(tracker.type_id, parameters=tracker.parameters)
    worker.copy_runtime_from(tracker)
    replay.run()  # The part is present again.
    assert tracker.last_result.outputs["report"]["status"] == "reset_required"
    assert tracker.last_result.outputs["pose"] is NO_VALUE
    assert tracker.last_result.outputs["tracked"] is False
    image_result = replay.blocks["video"].last_result
    record = replay.blocks["camera"].last_result.outputs["camera"]
    inputs = {
        "image": ValuePacket(image_result.outputs["image"], image_result.payloads["image"]),
        "camera": record,
    }
    assert worker.execute(inputs).outputs["pose"] is NO_VALUE
    inputs["reset"] = True
    revived = worker.execute(inputs)
    assert revived and revived.passed and revived.outputs["report"]["seeded_this_frame"]
    assert revived.payloads["pose"]["ticket"]["value"] == "part_motion.avi#000013"
    worker.close()


@pytest.mark.local_examples
def test_reprojection_gate_has_its_own_loss_reason(replay):
    tracker = replay.blocks["tracker"]
    tracker.parameters["maximum_reprojection_rms_px"] = 0.05
    replay.run()
    result = tracker.last_result
    assert result and not result.passed and result.outputs["pose"] is NO_VALUE
    assert result.outputs["report"]["reprojection_rms_px"] > 0.05
    assert result.fail_reasons[0]["code"] == "silhouette_reprojection_limit_exceeded"


@pytest.mark.local_examples
def test_typed_mesh_and_wired_seed_use_the_declared_camera_grid(replay):
    replay.run()
    saved = replay.blocks["tracker"]
    tracker = builtin_registry().create(saved.type_id)
    camera = replay.blocks["camera"].last_result.outputs["camera"]
    inputs = {
        "image": replay.blocks["video"].last_result.outputs["image"],
        "camera": camera,
        "mesh": TriangleMesh(**saved.parameters["model"]),
    }
    assert tracker.execute(inputs).outputs["report"]["status"] == "initial_pose_required"
    seed = saved.parameters["initial_pose"]
    assert tracker.execute({**inputs, "initial_pose": {**seed, "target_frame": "other"}}) is None
    assert "into the camera" in tracker.error
    assert tracker.execute({**inputs, "initial_pose": seed}).passed
    assert tracker.execute({**inputs, "camera": {**camera, "distortion": [0.1, 0, 0, 0]}}) is None
    assert "undistorted pinhole" in tracker.error
    tracker.close()
