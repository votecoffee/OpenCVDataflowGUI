"""Qualify bounded depth trajectory estimation, fused verification and replay recovery."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.depth_trajectory import TrackDepthFusion
from opencv_dataflow.images import DepthArray
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket
from opencv_dataflow.ptcloud3d import tsdf_views

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/depth_trajectory"


@pytest.fixture
def graph():
    graph = Graph.load(EXAMPLE / "depth_trajectory.json", builtin_registry())
    try:
        yield graph
    finally:
        graph.close()


def _run(graph, profile="debug"):
    graph.set_run_profile(profile)
    for _ in range(10):
        graph.run()
    tracker = graph.blocks["track"]
    assert tracker.last_result and tracker.last_result.passed, tracker.error
    group = graph.blocks["segment"].last_result.outputs["group"]
    camera = graph.blocks["camera"].last_result.outputs["camera"]
    return tracker, {"frames": group, "camera": camera}


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_known_recorded_trajectory_survives_loss_without_fusing_the_blank(graph, profile):
    original_opencl = cv2.ocl.useOpenCL()
    tracker, inputs = _run(graph, profile)
    assert cv2.ocl.useOpenCL() == original_opencl
    result = tracker.last_result
    report = result.outputs["trajectory"]
    truth = json.loads((EXAMPLE / "assets/ground_truth.json").read_text(encoding="utf-8"))
    assert report["complete"] and (report["accepted"], report["rejected"]) == (9, 1)
    assert report["tracking"][0]["status"] == "anchored"
    assert report["tracking"][4]["status"] == "insufficient_depth"
    assert "pose" not in report["tracking"][4]
    assert report["tracking"][5]["recovered"]
    assert not report["tracking"][6]["recovered"]
    poses = result.outputs["poses"]
    assert [r["source_index"] for r in poses] == [0, 1, 2, 3, 5, 6, 7, 8, 9]
    for pose in poses:
        expected = np.array(truth["camera_to_first_camera"][pose["source_index"]])
        got = np.array(pose["source_to_target"])
        assert np.linalg.norm(got[:3, 3] - expected[:3, 3]) < 0.003
        angle = cv2.Rodrigues(got[:3, :3] @ expected[:3, :3].T)[0]
        assert np.rad2deg(np.linalg.norm(angle)) < 0.35
        assert pose["target_frame"] == report["world_frame"] and pose["units"] == "m"
    np.testing.assert_array_equal(poses[0]["source_to_target"], np.eye(4))

    accepted, camera = result.outputs["frames"], inputs["camera"]
    last_depth = result.outputs["surface_depth"]
    assert np.isfinite(last_depth).mean() > 0.95
    assert result.payloads["surface_depth"]["camera_to_world"] == poses[-1]
    assert result.payloads["surface_depth"]["ticket"] == accepted.items[-1].payload["ticket"]
    assert report["tracking"][4]["ticket"] == inputs["frames"].items[4].payload["ticket"]
    assert [p.payload["recorded_rgbd"]["pair_index"] for p in accepted.items] == [
        pose["source_index"] for pose in poses
    ]
    _, _, same_world, _, _ = tsdf_views({"frames": accepted, "poses": poses}, camera)
    assert same_world == report["world_frame"]
    assert graph.sources_exhausted()


@pytest.mark.local_examples
def test_loss_limit_withholds_results_and_new_segment_has_a_fresh_origin(graph):
    tracker, inputs = _run(graph)
    saved_world = tracker.last_result.outputs["trajectory"]["world_frame"]
    block = TrackDepthFusion(parameters={**tracker.parameters, "maximum_consecutive_losses": 1})
    result = block.execute(inputs)
    assert result is not None and not result.passed
    assert all(result.outputs[name] is NO_VALUE for name in ("poses", "frames", "surface_depth"))
    rows = result.outputs["trajectory"]["tracking"]
    assert rows[4]["status"] == "insufficient_depth"
    assert all(r["status"] == "loss_limit_reached" and "pose" not in r for r in rows[5:])
    group = inputs["frames"]
    good = FlowGroup(group.items[5:], group.labels[5:])
    new = block.execute({**inputs, "frames": good})
    assert new and new.passed, block.error
    assert new.outputs["trajectory"]["world_frame"] != saved_world
    np.testing.assert_array_equal(new.outputs["poses"][0]["source_to_target"], np.eye(4))


@pytest.mark.local_examples
def test_fused_surface_can_reject_a_native_odometry_success(graph, monkeypatch):
    tracker, inputs = _run(graph)

    class BadOdometry:
        def compute(self, *_):
            return True, np.eye(4)

    monkeypatch.setattr("opencv_dataflow.depth_trajectory.make_odometry", lambda *_: BadOdometry())
    group = inputs["frames"]
    old = group.items[1]
    displaced = DepthArray(np.asarray(old.value) + np.float32(0.11), "m", 0, 2.0)
    mismatched = FlowGroup((group.items[0], ValuePacket(displaced, old.payload)), group.labels[:2])
    block = TrackDepthFusion(parameters=tracker.parameters)
    result = block.execute({**inputs, "frames": mismatched})
    assert result and not result.passed and result.outputs["poses"] is NO_VALUE
    rejected = result.outputs["trajectory"]["tracking"][1]
    assert rejected["status"] == "surface_mismatch" and rejected["model_support"] < 0.8
    assert "pose" not in rejected


@pytest.mark.local_examples
def test_recorded_camera_grid_and_order_are_checked(graph):
    tracker, inputs = _run(graph)
    block = TrackDepthFusion(parameters=tracker.parameters)
    group = inputs["frames"]
    reverse = FlowGroup(group.items[::-1], group.labels[::-1])
    assert block.execute({**inputs, "frames": reverse}) is None
    assert "strictly increasing frame order" in block.error
    frame = group.items[0]
    wrong = ValuePacket(frame.value, {**frame.payload, "optical_frame": "color_optical"})
    mixed = FlowGroup((wrong, *group.items[1:]), group.labels)
    assert block.execute({**inputs, "frames": mixed}) is None
    assert "optical frame differs" in block.error
