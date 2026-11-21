"""Qualify lens-aware pose, source-grid acceptance and the forward ray domain."""

import copy
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import camera_record, pose_matrix, pose_record
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.wide_pose import SolveWideAnglePose

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/wide_angle_pose"


def _angle_degrees(a, b):
    return np.degrees(np.arccos(np.clip((np.trace(a @ b.T) - 1) / 2, -1, 1)))


@pytest.mark.local_examples
@pytest.mark.parametrize("lens", ("fisheye", "omnidir"))
@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_saved_project_recovers_wide_angle_pose_and_reprojects_held_out_points(lens, profile):
    truth = json.loads((EXAMPLE / "assets/truth.json").read_text(encoding="utf-8"))[lens]
    graph = Graph.load(EXAMPLE / f"{lens}_object_pose.json", builtin_registry())
    graph.set_run_profile(profile)
    try:
        graph.run()
        fit = graph.blocks["fit"]
        pose = fit.last_result
        assert pose and pose.passed, fit.error
        out = pose.outputs
        report = out["report"]
        assert out["pose"]["source_frame"] == "fixture"
        assert out["pose"]["target_frame"] == "wide_camera" and out["pose"]["units"] == "mm"
        assert out["inliers"] == truth["accepted"]
        assert report["unsupported_ray_indices"] == [truth["out_of_domain_row"]]
        assert report["projected_pixels"][truth["out_of_domain_row"]] is None
        assert report["refinement_adopted"] and report["refine_evaluations"] > 0
        if lens == "fisheye":
            assert report["undistort_criteria"] == [50, 1e-10]
            assert set(report["native_inlier_indices"]) - set(out["inliers"])
        else:
            assert report["undistort_criteria"] is None
        t = np.asarray(out["pose"]["source_to_target"])
        np.testing.assert_allclose(t[:3, 3], truth["translation"], atol=0.03, rtol=0)
        assert _angle_degrees(t[:3, :3], np.asarray(truth["rotation"])) < 0.03
        wired_pixels = np.asarray(graph.blocks["xy"].last_result.outputs["vectors"])
        np.testing.assert_array_equal(wired_pixels, np.asarray(truth["measured"], dtype=np.float32))
        inlier_pixels = wired_pixels[out["inliers"]]
        radius = np.linalg.norm(inlier_pixels - [319.5, 239.5], axis=1)
        assert radius.min() < 23 <= 210 < radius.max()
        projected = np.asarray([report["projected_pixels"][i] for i in out["inliers"]])
        error = np.linalg.norm(projected - inlier_pixels, axis=1)
        assert out["rms"] == pytest.approx(np.sqrt(np.mean(error**2)))
        assert out["rms"] < 0.10
        held = graph.blocks["project_check"].last_result
        assert held and held.passed
        delta = np.asarray(held.outputs["points"]) - truth["held_pixels"]
        assert np.linalg.norm(delta, axis=1).max() < 0.06
        assert bool(pose.drawings) == (profile == "debug")
        assert (
            pose.payloads["pose"]["ticket"]
            == graph.blocks["view"].last_result.payloads["image"]["ticket"]
        )
    finally:
        graph.close()


@pytest.fixture(scope="module")
def models():
    code = runpy.run_path(str(EXAMPLE / "build_example.py"))
    result = {}
    for lens in ("fisheye", "omnidir"):
        d = code["LENSES"][lens]
        k = [[d["fx"], 0, 319.5], [0, d["fy"], 239.5], [0, 0, 1]]
        record = camera_record(
            k,
            d["distortion"],
            (640, 480),
            "mm",
            "wide_camera",
            lens,
            **({"xi": d["xi"]} if lens == "omnidir" else {}),
        )
        truth = code["test_scene"](lens)
        result[lens] = (record, truth)
    return result


@pytest.mark.local_examples
@pytest.mark.parametrize("lens", ("fisheye", "omnidir"))
@pytest.mark.parametrize("method", ("iterative", "epnp", "sqpnp"))
def test_uncontaminated_direct_native_route_is_exact_without_refinement(models, lens, method):
    camera, data = models[lens]
    tool = SolveWideAnglePose(
        parameters={
            "ransac": False,
            "refine_evaluations": 0,
            "source_frame": "fixture",
            "method": method,
        }
    )
    result = tool.execute(
        {
            "camera": camera,
            "object_points": data["xyz"],
            "image_points": data["pixels_truth"],
        }
    )
    assert result and result.passed, tool.error
    assert result.outputs["inliers"] == list(range(len(data["xyz"])))
    assert result.outputs["report"]["refine_evaluations"] == 0
    assert result.outputs["rms"] < 1e-7
    np.testing.assert_allclose(
        result.outputs["pose"]["translation"], data["translation"], atol=1e-7
    )


@pytest.mark.local_examples
@pytest.mark.parametrize("lens", ("fisheye", "omnidir"))
def test_pose_guess_and_source_pixel_gate(models, lens):
    camera, data = models[lens]
    tool = SolveWideAnglePose(
        parameters={"source_frame": "fixture", "max_rms_px": 0.001, "inlier_threshold_px": 0.65},
    )
    guess = pose_record(
        pose_matrix(data["rotation"], np.array(data["translation"]) + [2, -1, 3]),
        "fixture",
        "wide_camera",
        "mm",
    )
    inputs = {
        "camera": camera,
        "object_points": data["xyz"],
        "image_points": data["measured"],
        "initial_pose": guess,
    }
    out = tool.execute(inputs)
    assert out and not out.passed and out.outputs["pose"] is NO_VALUE
    assert out.fail_reasons[0]["code"] == "wide_pose_rms"
    assert out.outputs["inliers"] == data["accepted"]
    assert out.outputs["rms"] > 0.06

    wrong = {**guess, "source_frame": "different_object"}
    assert tool.execute({**inputs, "initial_pose": wrong}) is None
    assert "frames" in tool.error or "object frame" in tool.error


@pytest.mark.local_examples
def test_nonrobust_path_rejects_unsupported_rays_and_wrong_metadata(models):
    k, truth = models["omnidir"]
    base = {"camera": k, "object_points": truth["xyz"], "image_points": truth["measured"]}
    tool = SolveWideAnglePose(parameters={"ransac": False})
    result = tool.execute(base)
    assert result and not result.passed
    assert result.fail_reasons[0]["code"] == "wide_pose_ray_domain"
    assert result.outputs["report"]["unsupported_ray_indices"] == [truth["out_of_domain_row"]]

    assert tool.execute({**base, "camera": {**k, "xi": -0.15}}) is None
    assert "nonnegative xi" in tool.error
    assert (
        tool.execute(
            {**base, "object_points": ValuePacket(truth["xyz"], {"object_point_units": "m"})}
        )
        is None
    )
    assert "units" in tool.error
    tagged = {"spatial": frame([320, 240], "thumb")}
    assert tool.execute({**base, "image_points": ValuePacket(truth["measured"], tagged)}) is None
    assert "grid" in tool.error


@pytest.mark.local_examples
def test_pose_fails_on_underconstrained_geometry_and_loses_no_ticket(models):
    camera, data = models["fisheye"]
    ticket = TicketClock("wide-pose-test").issue("camera")
    tagged = ValuePacket(data["measured"], {"ticket": ticket})
    tool = SolveWideAnglePose()
    out = tool.execute({"camera": camera, "object_points": data["xyz"], "image_points": tagged})
    assert out and out.passed, tool.error
    assert out.payloads["pose"]["ticket"] == ticket

    same_point = copy.deepcopy(data["xyz"][0])
    out = tool.execute(
        {
            "camera": camera,
            "object_points": [same_point] * len(data["xyz"]),
            "image_points": tagged,
        }
    )
    assert out and not out.passed
    assert out.fail_reasons[0]["code"] == "wide_pose_degenerate"


@pytest.mark.local_examples
def test_native_proposal_behind_camera_cannot_publish_pose(models, monkeypatch):
    camera, truth = models["fisheye"]

    def behind(xyz, *args):
        return np.array([0, 0, 0, 0, 0, -10.0]), np.arange(len(xyz))

    monkeypatch.setattr("opencv_dataflow.wide_pose._solve", behind)
    result = SolveWideAnglePose().execute(
        {
            "camera": camera,
            "object_points": truth["xyz"],
            "image_points": truth["pixels_truth"],
        }
    )
    assert result and not result.passed
    assert result.outputs["pose"] is NO_VALUE and result.outputs["rms"] is NO_VALUE
    assert result.fail_reasons[0]["code"] == "wide_pose_insufficient_support"
    assert not result.outputs["inliers"]


@pytest.mark.local_examples
@pytest.mark.parametrize("lens", ("fisheye", "omnidir"))
def test_planar_ippe_recovers_tilted_fixture(models, lens):
    camera, data = models[lens]
    build = runpy.run_path(str(EXAMPLE / "build_example.py"))
    xyz = np.asarray(data["xyz"])
    xyz[:, 2] = 0
    xy = build["project"](lens, xyz, np.asarray(data["rotation"]), data["translation"])
    result = SolveWideAnglePose(parameters={"method": "ippe", "ransac": False}).execute(
        {
            "camera": camera,
            "object_points": xyz.tolist(),
            "image_points": xy.tolist(),
        }
    )
    assert result and result.passed
    assert result.outputs["rms"] < 1e-7
    np.testing.assert_allclose(
        result.outputs["pose"]["translation"], data["translation"], atol=1e-7
    )
