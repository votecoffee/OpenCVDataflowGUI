"""Compare an independently posed TSDF raycast to camera rays on a known plane."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import camera_record, pose_record
from opencv_dataflow.images import DepthArray, image_format
from opencv_dataflow.payloads import NO_VALUE

PROJECT = (
    Path(__file__).resolve().parents[1]
    / "examples/opencv tools/colored_tsdf_fusion"
    / "tsdf_target_raycast.json"
)


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_target_rgb_depth_normals_and_validity_against_analytic_plane(profile):
    graph = Graph.load(PROJECT, builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        block = graph.blocks["surface"]
        result = block.last_result
        assert result and result.passed, block.error
        output = result.outputs
        report = output["report"]
        raycast = report["raycast"]
        depth, normal, color, mask = (
            output[key] for key in ("depth", "normals", "color", "valid_mask")
        )
        assert depth.shape == mask.shape == (68, 112)
        assert normal.shape == color.shape == (68, 112, 3)
        assert image_format(color) == "rgb" and depth.units == "m"
        hit = mask == 255
        assert 5600 < hit.sum() < 6200
        assert raycast["valid_pixels"] == hit.sum()
        assert raycast["world_frame"] == "panel" and raycast["optical_frame"] == "inspection_camera"
        assert report["workspace_estimate_bytes"] == 44482560 + 96 * 68 * 112
        assert np.array_equal(np.isfinite(depth), hit)
        assert np.array_equal(np.isfinite(normal).all(axis=2), hit)
        assert np.all(color[~hit] == 0)
        assert {
            entry["spatial"]["root_id"]
            for key, entry in result.payloads.items()
            if key in {"depth", "normals", "color", "valid_mask"}
        } == {"surface:target_raycast"}
        assert result.payloads["depth"]["optical_frame"] == "inspection_camera"
        assert result.payloads["normals"]["normal_components"] == ["x", "y", "z"]
        assert result.payloads["color"]["ticket"] == result.payloads["cloud"]["ticket"]

        transform = np.asarray(
            graph.blocks["target_pose"].last_result.outputs["transform"]["source_to_target"]
        )
        rotation = transform[:3, :3]
        translation = transform[:3, 3] / 1000
        np.testing.assert_allclose(
            raycast["camera_to_world_m"],
            np.c_[rotation, translation].tolist() + [[0, 0, 0, 1]],
            atol=1e-7,
        )
        yy, xx = np.indices((68, 112))
        camera_rays = np.stack(((xx - 51.7) / 128, (yy - 29.4) / 125, np.ones_like(xx)), axis=2)
        world_rays = camera_rays @ rotation.T
        plane = np.array([-0.06, 0.03, 1])
        expected = (1.023 - translation @ plane) / (world_rays @ plane)
        assert np.quantile(np.abs(depth[hit] - expected[hit]), 0.99) < 0.015
        expected_normal = (-plane / np.linalg.norm(plane)) @ rotation
        np.testing.assert_allclose(np.median(normal[hit], axis=0), expected_normal, atol=0.002)
        np.testing.assert_allclose(np.linalg.norm(normal[hit], axis=1), 1, atol=1e-6)
        xyz = translation + expected[..., None] * world_rays
        x, y = xyz[..., 0], xyz[..., 1]
        for region, rgb in (
            (x < -0.08, [225, 35, 18]),
            ((x > 0.08) & (y < 0.04), [22, 50, 221]),
            ((x > 0.18) & (y > 0.16), [0, 0, 0]),
        ):
            check = hit & region
            assert check.sum() > 100
            np.testing.assert_array_equal(color[check], np.broadcast_to(rgb, (check.sum(), 3)))
        assert not np.any(hit & (np.abs(x) < 0.015) & (y < -0.18))
    finally:
        graph.close()


def _depth_inputs():
    k = [[80, 0, 31.5], [0, 80, 23.5], [0, 0, 1]]
    pose = np.eye(4)
    pose[2, 3] = 0.15
    return {
        "depth": DepthArray(np.full((48, 64), 1.009), "m"),
        "camera": camera_record(k, [], (64, 48), frame="capture"),
        "poses": pose_record(np.eye(4), "capture", "world"),
        "target_camera": camera_record(
            [[94, 0, 29], [0, 84, 22], [0, 0, 1]], [], (66, 41), frame="simulated"
        ),
        "target_pose": pose_record(pose, "simulated", "world"),
    }


def test_depth_only_target_retains_legacy_entry_and_reports_empty_raycasts():
    registry = builtin_registry()
    block = registry.create(
        "3d.tsdf_fuse",
        parameters={
            "resolution": [48, 40, 32],
            "volume_origin_m": [-0.30, -0.25, 0.78],
            "voxel_size_m": 0.0125,
            "truncation_m": 0.04,
        },
    )
    inputs = _depth_inputs()
    result = block.execute(inputs)
    assert result and result.passed, block.error
    mask = result.outputs["valid_mask"] == 255
    assert mask.sum() > 1200 and mask.shape == (41, 66)
    np.testing.assert_allclose(result.outputs["depth"][mask], 1.009 - 0.15, atol=0.02)
    assert result.outputs["raycast_report"]["has_color"] is False

    inputs["target_pose"]["source_to_target"][2][3] = 3.0
    missed = block.execute(inputs)
    assert missed and not missed.passed and len(missed.outputs["cloud"].points)
    assert missed.fail_reasons[0]["code"] == "tsdf_raycast_empty"
    assert missed.outputs["raycast_report"]["valid_pixels"] == 0
    assert np.isnan(missed.outputs["depth"]).all()

    del inputs["target_pose"]
    assert block.execute(inputs) is None and "Connect both" in block.error
    del inputs["target_camera"]
    legacy = block.execute(inputs)
    assert legacy and legacy.passed and legacy.outputs["depth"].shape == (48, 64)
    assert legacy.outputs["normals"] is NO_VALUE


def test_target_frame_and_raster_budgets_are_validated_before_native_volume(monkeypatch):
    def reject_volume(*args, **kwargs):
        pytest.fail("Invalid raycasts should be rejected before volume allocation")

    monkeypatch.setattr("opencv_dataflow.ptcloud3d.cpu_tsdf_volume", reject_volume)
    block = builtin_registry().create("3d.tsdf_fuse")
    inputs = _depth_inputs()
    inputs["target_pose"]["target_frame"] = "wrong"
    assert block.execute(inputs) is None and "volume world frame" in block.error
    inputs["target_pose"]["target_frame"] = "world"
    inputs["target_camera"]["image_size"] = [4096, 4096]
    assert block.execute(inputs) is None and "2,097,152 pixels" in block.error
