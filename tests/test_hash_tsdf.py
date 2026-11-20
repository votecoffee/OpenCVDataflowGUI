"""Qualify HashTSDF growth, atomic frame rejection and supported geometry."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import camera_record, pose_record
from opencv_dataflow.hash_tsdf import FuseHashedTSDF, admit_blocks, frame_block_box
from opencv_dataflow.payloads import NO_VALUE

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/hash_tsdf_growth"


def _sweep(profile="debug", limit=None):
    graph = Graph.load(EXAMPLE / "hash_tsdf_growth.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        if limit is not None:
            graph.blocks["growing"].parameters["block_limit"] = limit
        for _ in range(6):
            graph.run()
        return graph.blocks["growing"].last_result, graph.blocks["seeded"].last_result
    finally:
        graph.close()


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_sweep_extends_past_fixed_seed_bounds(profile):
    opencl = cv2.ocl.useOpenCL()
    growing, seeded = _sweep(profile)
    assert cv2.ocl.useOpenCL() == opencl
    assert growing.passed and seeded.passed
    grow, fixed = (result.outputs["report"] for result in (growing, seeded))
    assert grow["integrated_frames"] == 6 and fixed["integrated_frames"] == 3
    assert [row["reserved_blocks"] for row in grow["frames"]] == [60, 90, 110, 130, 160, 180]
    assert {frame["reserved_blocks"] for frame in fixed["frames"]} == {60}
    assert [frame["growth_enabled"] for frame in fixed["frames"]] == [
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert fixed["frames"][-1]["status"] == "outside_seed_reservation"
    assert grow["bounds_world_m"][1][0] > 2.30 and fixed["bounds_world_m"][1][0] < 0.50
    cloud, seed_cloud = (result.outputs["cloud"] for result in (growing, seeded))
    assert cloud.points[:, 0].max() > 2.15 and seed_cloud.points[:, 0].max() < 0.50
    assert cloud.frame == "plane" and cloud.units == "m"
    assert grow["extracted_points"] > grow["supported_points"] == len(cloud.points)
    assert np.max(np.abs(cloud.points[:, 2] - 1.009)) < 0.026
    np.testing.assert_allclose(np.linalg.norm(cloud.normals, axis=1), 1, atol=1e-6)


@pytest.mark.local_examples
def test_growth_cap_rejects_the_frame_before_it_can_change_the_volume():
    capped, _ = _sweep(limit=160)
    assert not capped.passed and capped.fail_reasons[0]["code"] == "hash_tsdf_growth_limit"
    report = capped.outputs["report"]
    assert report["integrated_frames"] == 5 and report["rejected_frames"] == 1
    assert report["reserved_blocks"] == report["block_limit"] == 160
    last = report["frames"][-1]
    assert last["status"] == "rejected_before_growth" and last["reserved_blocks"] == 160
    assert "native_visible_blocks" not in last
    assert 1.79 < capped.outputs["cloud"].points[:, 0].max() < 1.89
    assert 2.0 < report["bounds_world_m"][1][0] < 2.1


def test_conservative_box_covers_rotated_depth_truncation_and_negative_blocks():
    rng = np.random.default_rng(251)
    depth = rng.uniform(0.5, 2, (24, 29)).astype(np.float32)
    depth[4::9, :] = 0
    k = np.array([[60, 0, 12.1], [0, 55, 11.2], [0, 0, 1]], np.float32)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = cv2.Rodrigues(np.array([0.19, 0.27, 0.06]))[0]
    transform[:3, 3] = [-1.2, -1.5, 0.2]
    origin = np.array([0.1, -0.3, 0.6])
    side, truncation = 0.16, 0.06
    lower, upper = frame_block_box(depth, transform, k, origin, side, truncation)
    y, x = np.nonzero(depth)
    z = depth[y, x]
    points = np.c_[z * (x - k[0, 2]) / k[0, 0], z * (y - k[1, 2]) / k[1, 1], z]
    points = points @ transform[:3, :3].T + transform[:3, 3] - origin
    assert np.all(np.floor((points - truncation) / side) >= lower)
    assert np.all(np.floor((points + truncation) / side) <= upper)
    reserved = {(-2, 0, 0)}
    assert admit_blocks(reserved, ((-2, 0, 0), (0, 0, 0)), 2) is None
    assert reserved == {(-2, 0, 0)}


def test_workspace_can_reject_before_native_pool_and_small_cap_withholds_cloud(monkeypatch):
    camera = camera_record([[80, 0, 10], [0, 80, 10], [0, 0, 1]], [], (20, 20))
    inputs = {
        "camera": camera,
        "poses": pose_record(np.eye(4), "camera", "world"),
        "depth": np.full((20, 20), 1.009, np.float32),
    }
    result = FuseHashedTSDF(parameters={"block_limit": 1}).execute(inputs)
    assert not result.passed and result.outputs["cloud"] is NO_VALUE
    assert result.outputs["report"]["reserved_blocks"] == 0

    def native_pool(*args, **kwargs):
        pytest.fail("Workspace admission must precede native pool allocation")

    monkeypatch.setattr("opencv_dataflow.hash_tsdf.cpu_tsdf_volume", native_pool)
    tool = FuseHashedTSDF(parameters={"maximum_workspace_mib": 8})
    assert tool.execute(inputs) is None and "native pool" in tool.error
