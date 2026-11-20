"""Qualify calibrated multi-baseline reprojection, fusion and disagreement reporting."""

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow import depth_fusion
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.depth_fusion import FuseRegisteredDepth
from opencv_dataflow.geometry3d_common import camera_record, pose_record
from opencv_dataflow.images import DepthArray
from opencv_dataflow.payloads import ValuePacket
from opencv_dataflow.stereo3d import RectifyStereo, stereo_rig_record
from opencv_dataflow.stereo_rectified_camera import StereoRectifiedCamera

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/multi_baseline_depth"


def _box(bounds):
    x, y, width, height = bounds
    return np.s_[y : y + height, x : x + width]


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_three_baselines_detect_occlusion_reject_nearer_outlier_and_preserve_ties(profile):
    truth = json.loads((EXAMPLE / "assets/fixture.json").read_bytes())
    graph = Graph.load(EXAMPLE / "multi_baseline_depth.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        for block in graph.processing_blocks():
            assert block.last_result and block.last_result.passed, (block.id, block.error)
        result = graph.blocks["consensus"].last_result
        fused = result.outputs
        shape = (truth["height"], truth["width"])
        known = np.full(shape, truth["background_z_m"])
        known[_box(truth["foreground_rectangle"])] = truth["foreground_z_m"]
        mask = np.ones(shape, bool)
        mask[:, :8] = False  # Fewer than two overlapping right camera views.
        x, y, _, height = truth["foreground_rectangle"]
        mask[y : y + height, x - 4 : x] = False  # Geometric left of foreground occlusion.
        ties = np.zeros(shape, bool)
        ties[_box(truth["tie_rectangle"])] = True
        mask[ties] = False
        outliers = np.zeros(shape, bool)
        outliers[_box(truth["near_outlier_rectangle"])] = True
        np.testing.assert_array_equal(np.isfinite(fused["depth"]), mask)
        np.testing.assert_array_equal(fused["depth"][mask], known[mask])
        np.testing.assert_array_equal(fused["ambiguous_mask"], ties * 255)
        np.testing.assert_array_equal(fused["rejected_views"], outliers * 4)
        np.testing.assert_array_equal(fused["conflict_mask"], (ties | outliers) * 255)
        assert fused["report"]["ambiguous_pixels"] == 36
        assert fused["report"]["views"][2]["rejected_samples"] == 72
        assert fused["report"]["fused_pixels"] == mask.sum() == 15944
        compare = graph.blocks["evaluate_consensus"].last_result.outputs["report"]
        assert compare["bad_pixels"] == 0 and compare["mae_mm"] == 0
        assert compare["evaluated_pixels"] == 15944
        baseline = graph.blocks["evaluate_nearest"].last_result.outputs["report"]
        assert baseline["bad_pixels"] == 0 and baseline["evaluated_pixels"] == 15872
        for name in ("a", "b", "c"):
            projected = graph.blocks[f"registered_{name}"].last_result
            assert result.payloads["depth"]["spatial"] == projected.payloads["depth"]["spatial"]
            assert result.payloads["depth"]["ticket"] == projected.payloads["depth"]["ticket"]
        assert bool(result.graphics) == (profile == "debug")
    finally:
        graph.close()


def _oracle(depths, weights, tolerance):
    eligible = {
        i for i, (z, w) in enumerate(zip(depths, weights, strict=True)) if z > 0 and w >= 0.25
    }
    if not eligible:
        return None, False, 0, 0
    groups = {
        frozenset(j for j in eligible if depths[i] <= depths[j] <= depths[i] + tolerance)
        for i in eligible
    }
    maximum = max(map(len, groups))
    best = [group for group in groups if len(group) == maximum]
    if len(best) != 1:
        return None, True, 0, 0
    group = best[0]
    depth = sum(depths[i] * weights[i] for i in group) / sum(weights[i] for i in group)
    bits = sum(1 << i for i in eligible - group)
    return depth, False, bits, len(group)


def test_consensus_matches_scalar_oracle_on_adversarial_four_view_pixels(monkeypatch):
    monkeypatch.setattr(depth_fusion, "_BAND_ROWS", 1)
    pixels = np.array(
        [
            [0.75, 1.5, 1.5, 1.5625],  # Unambiguous farther consensus.
            [1, 2, 0, 0],  # Two incompatible singleton groups.
            [1, 1, 2, 2],  # Equal-sized disjoint groups.
            [1, 1.09375, 1.1875, 0],  # Overlapping pairs still tied.
            [0, 0, 0, 0],  # No support.
            [2, 2.125, 2, 2],  # Inclusive tolerance.
            [1, 1.75, 1.75, 1.75],  # Initially tied candidates, later clear winner.
            [1.5, 1.625, 1.5, 0.5],  # Final slot has low confidence.
        ],
        np.float32,
    ).reshape(2, 4, 4)
    weights = np.broadcast_to([1, 0.5, 0.875, 1], pixels.shape).astype(np.float32).copy()
    weights[1, 3, 3] = 0
    data = {
        f"{kind}_{key}": DepthArray(pixels[..., i], "m") if kind == "depth" else weights[..., i]
        for i, key in enumerate("abcd")
        for kind in ("depth", "confidence")
    }
    block = FuseRegisteredDepth(
        parameters={
            "surface_policy": "consensus",
            "surface_tolerance_m": 0.125,
            "minimum_confidence": 0.25,
            "minimum_support": 2,
        }
    )
    result = block.execute(data)
    assert result and result.passed, block.error
    out = result.outputs
    for index in np.ndindex(pixels.shape[:2]):
        depth, tied, bits, count = _oracle(pixels[index], weights[index], 0.125)
        assert bool(out["ambiguous_mask"][index]) == tied
        assert out["rejected_views"][index] == bits
        assert out["support_count"][index] == count
        if depth is None or count < 2:
            assert np.isnan(out["depth"][index])
        else:
            assert out["depth"][index] == pytest.approx(depth, abs=2e-7)
    assert out["report"]["no_eligible_depth_pixels"] == 1
    assert out["report"]["ambiguous_pixels"] == 3
    assert out["report"]["insufficient_support_pixels"] == 3


def _rig():
    size = (88, 68)
    k = np.array([[86, 0, 43.5], [0, 91, 33.5], [0, 0, 1]])
    left = camera_record(k, [], size, frame="left")
    right = camera_record(k, [], size, frame="right")
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(np.array([0.03, -0.09, 0.03]))[0]
    transform[:3, 3] = [-0.07, 0.014, 0.01]
    rig = stereo_rig_record(left, right, pose_record(transform, "left", "right", "m"))
    image = np.zeros(size[::-1], np.uint8)
    rectify = RectifyStereo()
    result = rectify.execute({"left": image, "right": image, "rig": rig})
    assert result and result.passed, rectify.error
    return left, image, result


def test_nonidentity_rectification_is_undone_before_projecting_to_the_original_grid():
    camera, image, rectified = _rig()
    rec = rectified.outputs["rectification"]
    rotation = np.asarray(rec["R1"])
    assert np.linalg.norm(rotation - np.eye(3)) > 0.05
    block = StereoRectifiedCamera()
    result = block.execute({"rectification": rec, "left_camera": camera})
    assert result and result.passed, block.error
    outputs = result.outputs
    np.testing.assert_allclose(outputs["camera"]["camera_matrix"], np.asarray(rec["P1"])[:, :3])
    np.testing.assert_allclose(
        np.asarray(outputs["transform"]["source_to_target"])[:3, :3] @ rotation,
        np.eye(3),
        atol=1e-10,
    )
    y, x = np.mgrid[: image.shape[0], : image.shape[1]]
    k = np.asarray(rec["P1"])
    rays = np.stack(((x - k[0, 2]) / k[0, 0], (y - k[1, 2]) / k[1, 1], np.ones(x.shape)), axis=2)
    expected_z = 1.125
    depth_on_rectified_grid = expected_z / (rays @ rotation)[..., 2]
    source = ValuePacket(DepthArray(depth_on_rectified_grid, "m"), rectified.payloads["left"])
    assert rectified.payloads["left"]["optical_frame"] == "left:rectified"
    assert rectified.payloads["right"]["optical_frame"] == "right:rectified"
    register = builtin_registry().create("3d.register_rgbd")
    outcome = register.execute(
        {
            "color": image,
            "depth": source,
            "depth_camera": outputs["camera"],
            "color_camera": camera,
            "transform": outputs["transform"],
        }
    )
    assert outcome and outcome.passed, register.error
    projected = (rays @ rotation) @ np.asarray(camera["camera_matrix"]).T
    cells = np.floor(projected[..., :2] / projected[..., 2, None] + 0.5).astype(int)
    width, height = camera["image_size"]
    inside = ((cells >= 0) & (cells < [width, height])).all(axis=2)
    expected_coverage = np.zeros(image.shape, bool)
    xc, yc = cells[inside].T
    expected_coverage[yc, xc] = True
    z = np.asarray(outcome.outputs["depth"])
    valid = np.isfinite(z)
    assert valid.any() and (~valid).any()
    np.testing.assert_array_equal(valid, expected_coverage)
    np.testing.assert_allclose(z[valid], expected_z, atol=2e-7, rtol=0)


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("frame", "original metric left camera"),
        ("units", "original metric left camera"),
        ("translated_p1", "camera origin"),
        ("rotation", "orthonormal rotation"),
    ],
)
def test_rectified_camera_rejects_inconsistent_transforms(fault, message):
    camera, _, rectified = _rig()
    rec = copy.deepcopy(rectified.outputs["rectification"])
    if fault == "frame":
        rec["frame"] = "other:rectified"
    elif fault == "units":
        rec["units"] = "mm"
    elif fault == "translated_p1":
        rec["P1"][0][3] = -12.0
    else:
        rec["R1"][0][0] = 1.5
    block = StereoRectifiedCamera()
    assert block.execute({"rectification": rec, "left_camera": camera}) is None
    assert message in block.error
