"""Qualify fixed-view reconstruction, recovery and surface provenance."""

from pathlib import Path
from runpy import run_path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.depth_evaluation import EvaluateDepth
from opencv_dataflow.depth_surface_tracking import TrackDeformingDepthSurface, _surface_samples
from opencv_dataflow.geometry3d_common import camera_record
from opencv_dataflow.images import DepthArray
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/deforming_depth_surface"


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_recorded_sheet_reconstructs_nonrigid_and_recovers_after_tracking_loss(profile):
    fixture = run_path(str(EXAMPLE / "build_example.py"))
    graph = Graph.load(EXAMPLE / "deforming_depth_surface.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.set_branch_workers(2)
        tracker = graph.blocks["track"]
        generation, material_grid, reference_ticket = None, None, None
        measured = 0
        for index in range(fixture["FRAME_COUNT"]):
            graph.run()
            result = tracker.last_result
            assert result is not None, tracker.error
            outputs, status = result.outputs, result.outputs["status"]
            assert status["current_recording"]["pair_index"] == index
            if index == 0:
                assert status["surface_state"] == "anchored"
                reference_ticket = status["reference_ticket"]
                generation = status["generation"]
            assert status["generation"] == generation
            assert status["reference_ticket"] == reference_ticket
            assert status["reference_recording"]["pair_index"] == 0

            if index in (0, fixture["DROPPED"]):
                assert not outputs["ready"] and outputs["cloud"] is NO_VALUE
                assert not result.emissions["cloud"]
                if index:
                    assert not result.passed
                    assert status["surface_state"] == "tracking_lost"
                    assert result.fail_reasons[0]["code"] == "deforming_surface_tracking_lost"
                continue

            assert result.passed and outputs["ready"] and status["surface_state"] == "tracked"
            cloud = outputs["cloud"]
            indices = cloud.attributes["source_index"]
            assert len(np.unique(indices)) == len(indices) == status["accepted_points"] > 590
            assert cloud.units == "m" and cloud.frame == fixture["CAMERA"]
            assert cloud.metadata["recorded_rgbd"]["pair_index"] == index
            assert cloud.metadata["ticket"] == status["current_ticket"]
            assert cloud.metadata["sequence"]["reference_ticket"] == reference_ticket
            xyz = fixture["truth_xyz"](index, indices)
            errors = np.linalg.norm(cloud.points - xyz, axis=1)
            assert np.sqrt(np.mean(errors**2)) < 0.00065
            assert np.max(errors) < 0.003
            y, x = np.divmod(indices, fixture["SIZE"][0])
            expected_uv = np.column_stack((x + fixture["shift"](index, y), y))
            np.testing.assert_allclose(cloud.pixels, expected_uv, atol=1.0)
            truth_motion = np.linalg.norm(xyz - fixture["truth_xyz"](0, indices), axis=1)
            np.testing.assert_allclose(
                outputs["displacement"].flat[indices], truth_motion, atol=0.003
            )
            reference = np.full(fixture["SIZE"][::-1], np.nan, np.float32)
            reference.flat[indices] = xyz[:, 2]
            judge = EvaluateDepth(parameters={"alignment": "pixel_grid"})
            tested = judge.execute(
                {"estimate": outputs["surface_depth"], "reference": DepthArray(reference, "m")}
            )
            assert tested and tested.passed, judge.error
            assert tested.outputs["report"]["rmse_mm"] < 0.35
            assert tested.outputs["report"]["bad_pixels"] == 0

            surface_payload = result.payloads["surface_depth"]
            if material_grid is None:
                material_grid = surface_payload["spatial"]
            assert surface_payload["spatial"] == material_grid
            assert material_grid != cloud.metadata["spatial"]
            assert surface_payload["surface_layout"]["domain"] == "reference_material_pixels"
            assert "optical_frame" not in surface_payload
            assert ("Reference-grid displacement" in result.graphics) == (profile == "debug")
            measured += 1
        assert measured == 3
    finally:
        graph.close()


def _camera(size=(12, 10), **overrides):
    values = camera_record([[35.0, 0, 5.5], [0, 34.0, 4.5], [0, 0, 1]], [], size, "mm", "optical")
    return {**values, **overrides}


def _fixed():
    return TrackDeformingDepthSurface(
        block_id="surface",
        parameters={"correspondence": "fixed_pixel", "sample_step": 2, "minimum_points": 4},
    )


def test_fixed_pixel_depth_only_preserves_axial_displacement_and_reset_reanchors():
    block = _fixed()
    baseline = np.full((10, 12), 900, np.uint16)
    bent = baseline.copy()
    bent[:, :6] += 14
    bent[2, 2] = 0
    clock = TicketClock("sample")
    spatial = frame((12, 10), "fixture:depth")
    camera = _camera()
    region = {
        "model": "image_region_v1",
        "shape": "rectangle",
        "image_size": [12, 10],
        "rectangle": [0, 0, 8, 10],
    }

    def run(depth, reset=False):
        ticket = clock.issue("replay")
        packet = ValuePacket(
            depth, {"spatial": spatial, "optical_frame": "optical", "ticket": ticket}
        )
        return block.execute(
            {"camera": camera, "image": packet, "region": region, "reset": reset}
        ), ticket

    first, original = run(baseline)
    assert first and first.outputs["status"]["surface_state"] == "anchored"
    result, current = run(bent)
    assert result and result.passed and result.outputs["status"]["ready"], block.error
    cloud = result.outputs["cloud"]
    assert cloud.metadata["ticket"] == current
    assert len(cloud.points) == 19
    assert 2 + 2 * 12 not in cloud.attributes["source_index"]
    assert np.all(cloud.pixels[:, 0] < 8)
    assert len(set(cloud.points[:, 2].round(3))) == 2
    np.testing.assert_allclose(
        cloud.points[:, 2], bent[cloud.pixels[:, 1], cloud.pixels[:, 0]] / 1000, atol=1e-7
    )
    old_root = result.payloads["surface_depth"]["spatial"]["root_id"]
    rejected, _ = run(np.full_like(bent, 1500))
    assert rejected and not rejected.passed
    assert rejected.outputs["status"]["surface_state"] == "tracking_lost"
    assert rejected.outputs["status"]["depth_supported_points"] == 20
    assert rejected.outputs["status"]["accepted_points"] == 0
    assert rejected.outputs["cloud"] is NO_VALUE

    reset, replacement = run(bent, reset=True)
    assert reset.outputs["status"]["surface_state"] == "anchored"
    assert reset.outputs["status"]["reference_ticket"] == replacement != original
    assert reset.outputs["cloud"] is NO_VALUE
    reobserved, _ = run(bent)
    assert reobserved.outputs["ready"]
    assert reobserved.payloads["surface_depth"]["spatial"]["root_id"] != old_root
    np.testing.assert_array_equal(
        reobserved.outputs["displacement"][::2, :8:2][
            np.isfinite(reobserved.outputs["displacement"][::2, :8:2])
        ],
        0,
    )
    block.reset_runtime()
    after_closeout, _ = run(baseline)
    assert after_closeout.outputs["status"]["activations"] == 1


def test_bilinear_sampling_rejects_invalid_neighbors_and_cross_surface_blending():
    depth = np.array([[1, 1.02, 2], [1, 1.02, np.nan], [1, 1.02, 2]], np.float32)
    points = np.array([[0.5, 0.5], [1.5, 0.5], [1.5, 2], [2.0, 2]], np.float32)
    keep, samples = _surface_samples(depth, np.isfinite(depth), points, 0.05)
    np.testing.assert_array_equal(keep, [True, False, False, True])
    np.testing.assert_allclose(samples, [1.01, 2], atol=1e-7)


def test_unaligned_or_different_pair_is_rejected():
    model = _camera()
    block = TrackDeformingDepthSurface(parameters={"minimum_points": 2})
    depth = DepthArray(np.ones((10, 12), np.float32), "m")
    guide = np.full(depth.shape, 100, np.uint8)
    stamp = {"clock": "sample", "timestamp_ns": 22, "pair_index": 1}
    packet = ValuePacket(depth, {"recorded_rgbd": stamp, "optical_frame": "optical"})
    invalid_guide = ValuePacket(
        guide, {"optical_frame": "optical", "recorded_rgbd": {**stamp, "pair_index": 2}}
    )
    assert block.execute({"image": packet, "camera": model, "guide": invalid_guide}) is None
    assert "same recorded RGB-D pair" in block.error
    assert block.execute({"image": depth, "camera": model}) is None
    assert "registered guide" in block.error
    assert block.execute({"image": depth, "camera": model, "guide": guide[:6]}) is None
    assert "pixel dimensions" in block.error


def test_invalid_camera_and_oversize_reference_terminate_before_tracking():
    block = _fixed()
    depth = DepthArray(np.ones((10, 12), np.float32), "m")
    assert block.execute({"image": depth, "camera": _camera(distortion=[0.1, 0, 0, 0, 0])}) is None
    assert "undistorted" in block.error
    high = DepthArray(np.ones((260, 260), np.float32), "m")
    other = TrackDeformingDepthSurface(
        parameters={"correspondence": "fixed_pixel", "sample_step": 1}
    )
    assert other.execute({"image": high, "camera": _camera((260, 260))}) is None
    assert "65,536" in other.error
