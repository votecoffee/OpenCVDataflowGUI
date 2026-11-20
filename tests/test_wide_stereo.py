"""Independent rig, epipolar and depth checks for native wide-angle stereo."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.calibration_revisions import REVISION
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE, packet_for

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/wide_angle_stereo"
TRUTH = json.loads((BASE / "truth.json").read_bytes())
R = cv2.Rodrigues(np.asarray(TRUTH["rvec"]))[0]
T = np.array(TRUTH["translation"])


def raw_pixels(xyz, k, xi):
    if xi is None:
        radial = xyz[:, :2] / xyz[:, 2, None]
        magnitude = np.linalg.norm(radial, axis=1)
        scale = np.divide(
            np.arctan(magnitude), magnitude, out=np.ones_like(magnitude), where=magnitude > 0
        )
        normalized = radial * scale[:, None]
    else:
        normalized = xyz[:, :2] / (xyz[:, 2] + xi * np.linalg.norm(xyz, axis=1))[:, None]
    return (normalized * np.diag(k)[:2] + k[:2, 2])[:, None]


def rectified_points(point_list, lens, model, side, record):
    k = np.asarray(model["camera_matrix"])
    d = np.asarray(model["distortion"])
    rot = np.asarray(record[f"R{side}"])
    proj = np.asarray(record[f"P{side}"])[:, :3].copy()
    if lens == "fisheye":
        return cv2.fisheye.undistortPoints(point_list, k, d, R=rot, P=proj).reshape(-1, 2)
    rays = cv2.omnidir.undistortPoints(point_list, k, d, np.array([model["xi"]]), rot)
    return cv2.perspectiveTransform(rays, proj).reshape(-1, 2)


@pytest.mark.parametrize("lens", ["fisheye", "omnidir"])
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_fit_rectifies_held_out_points_and_depth(lens, profile):
    graph = Graph.load(BASE / f"{lens}.json", builtin_registry())
    try:
        check = validate_graph(graph)
        assert check["valid"], check["issues"]
        graph.set_run_profile(profile)
        graph.run()
        assert all(b.state == "pass" for b in graph.processing_blocks()), {
            b.id: b.error for b in graph.processing_blocks()
        }
        result = graph.blocks["fit"].last_result.outputs
        rig = result["rig"]
        report = result["fit_report"]
        np.testing.assert_allclose(rig["transform"]["translation"], T, atol=2e-5)
        np.testing.assert_allclose(rig["transform"]["rvec"], TRUTH["rvec"], atol=2e-5)
        assert result["rms"] < 1e-4 and report["worst_view_rms_pixels"] < 1e-4
        assert rig["left"]["lens"] == rig["right"]["lens"] == lens
        assert sorted(report["used_view_indices"] + report["excluded_view_indices"]) == list(
            range(14)
        )
        assert [view["index"] for view in report["views"]] == report["used_view_indices"]
        if lens == "omnidir":
            assert [rig[side]["xi"] for side in ("left", "right")] == pytest.approx(
                [cam["xi"] for cam in TRUTH[lens]], abs=0.001
            )
        rec = graph.blocks["rectify"].last_result.outputs["rectification"]
        assert rec["source_lens"] == lens
        a, b = np.meshgrid(np.linspace(-1.6, 1.6, 7), np.linspace(-0.9, 0.9, 5))
        left_xyz = np.c_[a.ravel(), b.ravel(), np.full(a.size, 3.0)]
        right_xyz = left_xyz @ R.T + T
        raw = [
            raw_pixels(xyz, np.asarray(cam["matrix"]), cam["xi"])
            for xyz, cam in zip((left_xyz, right_xyz), TRUTH[lens], strict=True)
        ]
        left, right = (
            rectified_points(pts, lens, rig[side], i, rec)
            for i, (pts, side) in enumerate(zip(raw, ("left", "right"), strict=True), start=1)
        )
        assert np.max(np.abs(left[:, 1] - right[:, 1])) < 0.005
        disparity = left[:, 0] - right[:, 0]
        xyz_h = np.c_[left, disparity, np.ones(len(left))] @ np.asarray(rec["Q"]).T
        xyz = xyz_h[:, :3] / xyz_h[:, 3, None]
        np.testing.assert_allclose(xyz, left_xyz @ np.asarray(rec["R1"]).T, atol=0.003)
        cloud = graph.blocks["cloud"].last_result.outputs["cloud"]
        assert cloud.units == "m" and len(cloud.points) > 1000
        assert np.median(cloud.points[:, 2]) == pytest.approx(TRUTH["plane_depth_m"], abs=0.09)
    finally:
        graph.close()


def test_bad_view_rms_withholds_rig_and_mixed_lenses_are_rejected():
    registry = builtin_registry()
    graph = Graph.load(BASE / "fisheye.json", registry)
    try:
        graph.run()
        inputs = {
            key: graph.blocks[f"{side}_model"].last_result.outputs["camera"]
            for key, side in (("left_camera", "left"), ("right_camera", "right"))
        }
        groups = json.loads((BASE / "assets/fisheye-views.json").read_bytes())
        groups["left_views"][4][12][0] += 0.6
        fit = registry.create("3d.wide_stereo_calibrate", parameters={"max_view_rms": 0.02})
        out = fit.execute({**inputs, **groups})
        assert out and not out.passed and out.outputs["rig"] is NO_VALUE
        assert out.outputs["fit_report"]["worst_view_rms_pixels"] > 0.02
        assert {reason["code"] for reason in out.fail_reasons} == {"wide_stereo.view_rms"}
        for pair in groups.values():
            pair[3].pop()
        assert fit.execute({**inputs, **groups}) is None and "equal point counts" in fit.error
        inputs["right_camera"] = {**inputs["right_camera"], "lens": "omnidir", "xi": 0.8}
        assert fit.execute({**inputs, **groups}) is None and "lens family" in fit.error
    finally:
        graph.close()


def test_versioned_wide_angle_rig_still_checks_capture_metadata(tmp_path):
    lens = "omnidir"
    registry = builtin_registry()
    graph = Graph.load(BASE / f"{lens}.json", registry)
    try:
        graph.run()
        rig = graph.blocks["fit"].last_result.outputs["rig"]
        selector = {"path": str(tmp_path), "profile_id": "wide-rig", "revision": "r1"}
        assert (
            registry.create("output.calibration", parameters=selector)
            .execute({"profile": rig})
            .passed
        )
        loaded = (
            registry.create("input.calibration", parameters=selector).execute({}).outputs["profile"]
        )
        rectify = registry.create("3d.stereo_rectify")
        frames = {
            side: packet_for(graph.blocks[side].last_result, "image") for side in ("left", "right")
        }
        assert rectify.execute({**frames, "rig": loaded}) is None
        assert "matching capture metadata" in rectify.error
        for side, packet in frames.items():
            declare = registry.create(
                "calibration.declare_capture",
                parameters={
                    "profile_id": selector["profile_id"],
                    "revision": selector["revision"],
                    "camera_frame": rig[side]["frame"],
                    "sha256": loaded[REVISION]["sha256"],
                },
            )
            frames[side] = packet_for(declare.execute({"image": packet}), "image")
        outcome = rectify.execute({**frames, "rig": loaded})
        assert outcome and outcome.passed, rectify.error
        assert outcome.outputs["rectification"][REVISION] == loaded[REVISION]
    finally:
        graph.close()
