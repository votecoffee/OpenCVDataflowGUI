"""Qualify independent camera frames, cheirality, lens residuals, and baseline scale."""

import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import camera_record
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.relative_pose import IndependentRelativePose

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/independent_relative_pose"


def _cos_degrees(a, b):
    return np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)))


def _camera(model, name, units="m"):
    k = [[model["fx"], 0, model["cx"]], [0, model["fy"], model["cy"]], [0, 0, 1]]
    optional = {"xi": model["xi"]} if model["lens"] == "omnidir" else {}
    return camera_record(
        k,
        model["distortion"],
        (model["width"], model["height"]),
        units,
        name,
        model["lens"],
        **optional,
    )


@pytest.fixture(scope="module")
def fixture():
    return runpy.run_path(str(EXAMPLE / "build_example.py"))


def _inputs(fixture, lens, scenario, noise=True):
    return {
        "points_left": scenario["left" if noise else "left_truth"],
        "points_right": scenario["right" if noise else "right_truth"],
        "camera_left": _camera(fixture["LEFT"], "left"),
        "camera_right": _camera(fixture["right_camera"](lens), "right", "mm"),
    }


@pytest.mark.local_examples
@pytest.mark.parametrize("lens", ("pinhole", "fisheye"))
@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_saved_projects_recover_direction_and_source_pixels(fixture, lens, profile):
    truth = json.loads((EXAMPLE / "assets/truth.json").read_text(encoding="utf-8"))[lens]
    graph = Graph.load(EXAMPLE / f"relative_pose_{lens}.json", builtin_registry())
    graph.set_run_profile(profile)
    try:
        graph.run()
        block = graph.blocks["pose"]
        result = block.last_result
        assert result and result.passed, block.error
        data = result.outputs
        pose, report = data["pose"], data["report"]
        assert pose["source_frame"] == "left_camera" and pose["target_frame"] == "right_camera"
        assert report["left_image_size"] == [800, 540]
        assert report["right_image_size"] == [930, 670]
        assert report["right_lens"] == lens
        assert report["invalid_right_ray_indices"] == truth["invalid_ray_indices"]
        assert data["inlier_indices"] == truth["inliers"]
        assert report["median_parallax_deg"] > 3
        assert report["rms_left_px"] < 0.12 and report["rms_right_px"] < 0.08
        assert report["refit_adopted"]
        transform = np.asarray(pose["source_to_target"])
        r, t = transform[:3, :3], transform[:3, 3]
        target = np.asarray(truth["right_center_in_left"])
        assert _cos_degrees(-r.T @ t, target) < 0.05
        assert _cos_degrees(t, np.asarray(truth["translation_m"])) < 0.08
        assert abs(np.linalg.det(r) - 1) < 1e-9
        actual = np.asarray(truth["rotation"])
        angle = np.degrees(np.arccos(np.clip((np.trace(r @ actual.T) - 1) / 2, -1, 1)))
        assert angle < 0.06
        scale = truth["baseline_m"]
        cloud = data["cloud"]
        assert cloud.frame == "left_camera" and cloud.units == pose["units"]
        assert cloud.attributes["source_index"].tolist() == truth["inliers"]
        assert np.all(cloud.points[:, 2] > 0)
        assert np.all((cloud.points @ r.T + t)[:, 2] > 0)
        if lens == "fisheye":
            assert pose["scale"] == "measured_baseline" and pose["units"] == "m"
            assert np.linalg.norm(t) == pytest.approx(scale)
        else:
            assert pose["scale"] == "unknown" and pose["units"] == "unitless"
            assert np.linalg.norm(t) == pytest.approx(1)
            assert report["baseline_length"] is None
            # Synthetic truth supplies this scale for the test only.
            t = t * scale

        held = np.asarray(truth["held_xyz_m"]) @ r.T + t
        source_right = fixture["pixels"](held, fixture["right_camera"](lens))
        error = np.linalg.norm(source_right - truth["held_right"], axis=1)
        assert error.max() < 0.38
        for side in ("left", "right"):
            image = graph.blocks[f"{side}_image"].last_result.outputs["image"]
            np.testing.assert_array_equal(data[f"{side}_image"], image)
            upstream = graph.blocks[f"{side}_image"].last_result.payloads["image"]["ticket"]
            assert result.payloads[f"{side}_image"]["ticket"] == upstream
        assert bool(result.drawings) == (profile == "debug")
    finally:
        graph.close()


@pytest.mark.local_examples
def test_omnidir_lmeds_and_metric_baseline_in_millimeters(fixture):
    data = fixture["scene"]("omnidir")
    tool = IndependentRelativePose(
        parameters={
            "method": "lmeds",
            "baseline_length": 1000 * data["baseline_m"],
            "baseline_units": "mm",
        }
    )
    result = tool.execute(_inputs(fixture, "omnidir", data))
    assert result and result.passed, tool.error
    out = result.outputs
    assert out["inlier_indices"] == data["inliers"]
    assert out["pose"]["units"] == "mm" and out["cloud"].units == "mm"
    assert np.linalg.norm(out["pose"]["translation"]) == pytest.approx(1000 * data["baseline_m"])
    r = np.array(out["pose"]["source_to_target"])[:3, :3]
    assert (
        _cos_degrees(-r.T @ out["pose"]["translation"], np.asarray(data["right_center_in_left"]))
        < 0.16
    )


@pytest.mark.local_examples
@pytest.mark.parametrize("baseline", (0.0, 0.0001))
def test_pure_rotation_and_tiny_parallax_do_not_publish_a_baseline(fixture, baseline):
    data = fixture["scene"]("fisheye", np.array([baseline, 0, 0]))
    block = IndependentRelativePose()
    result = block.execute(_inputs(fixture, "fisheye", data, noise=False))
    assert result and not result.passed, block.error
    assert result.outputs["pose"] is NO_VALUE
    assert result.outputs["essential"] is NO_VALUE
    assert result.fail_reasons[0]["code"] in {
        "relative_pose_cheirality",
        "relative_pose_parallax",
        "relative_pose_no_essential",
    }


@pytest.mark.local_examples
def test_native_proposal_is_still_gated_in_original_pixels(fixture):
    data = fixture["scene"]("fisheye")
    tool = IndependentRelativePose(parameters={"maximum_reprojection_px": 0.0001})
    result = tool.execute(_inputs(fixture, "fisheye", data))
    assert result and not result.passed, tool.error
    assert result.outputs["pose"] is NO_VALUE
    assert len(result.outputs["report"]["native_inlier_indices"]) >= 80
    assert result.fail_reasons[0]["code"] == "relative_pose_reprojection"


@pytest.mark.local_examples
def test_independent_view_domains_and_metadata_are_validated(fixture):
    scenario = fixture["scene"]("omnidir")
    inputs = _inputs(fixture, "omnidir", scenario)
    tool = IndependentRelativePose()
    assert (
        tool.execute({**inputs, "camera_right": {**inputs["camera_right"], "frame": "left"}})
        is None
    )
    assert "distinct camera frame" in tool.error
    assert tool.execute({**inputs, "camera_right": {**inputs["camera_right"], "xi": -1}}) is None
    assert "nonnegative xi" in tool.error
    mismatched_grid = ValuePacket(scenario["right"], {"spatial": frame([800, 540], "left")})
    assert tool.execute({**inputs, "points_right": mismatched_grid}) is None
    assert "Right points use a different" in tool.error
    assert tool.execute({**inputs, "points_right": scenario["right"][:-1]}) is None
    assert "counts differ" in tool.error


@pytest.mark.local_examples
def test_insufficient_and_collinear_observations_fail_explicitly(fixture):
    scenario = fixture["scene"]("pinhole")
    inputs = _inputs(fixture, "pinhole", scenario)
    tool = IndependentRelativePose()
    few = tool.execute(
        {**inputs, "points_left": scenario["left"][:8], "points_right": scenario["right"][:8]}
    )
    assert few and not few.passed
    assert few.fail_reasons[0]["code"] == "relative_pose_insufficient_pairs"
    # Identical points remain collinear even after nonlinear inverse projection.
    same = tool.execute({**inputs, "points_left": [scenario["left"][0]] * 96})
    assert same and not same.passed
    assert same.fail_reasons[0]["code"] == "relative_pose_degenerate"
