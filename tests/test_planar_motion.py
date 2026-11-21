"""Verify planar camera motion alternatives, visibility and metric scale."""

from pathlib import Path
from runpy import run_path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import camera_record
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.planar_motion import _physical_candidate

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/planar_camera_motion"
TYPE = "3d.planar_camera_motion"


@pytest.fixture(scope="module")
def fixture():
    return run_path(str(EXAMPLE / "build_example.py"))


def _inputs(fixture, *, distance_only=False, translation=None):
    options = {} if translation is None else {"translation": translation}
    left, right = fixture["analytic_pairs"](**options)
    k1, k2 = fixture["K1"], fixture["K2"]
    optics = {
        f"{side}_camera": camera_record(k, d, fixture["SIZE"], frame=side)
        for side, k, d in (("source", k1, fixture["D1"]), ("target", k2, fixture["D2"]))
    }
    if distance_only:
        return {
            **optics,
            "matches": {
                "query_points": left.tolist(),
                "train_points": right.tolist(),
                "query_size": list(fixture["SIZE"]),
                "train_size": list(fixture["SIZE"]),
            },
        }
    return {**optics, "source_points": left.tolist(), "target_points": right.tolist()}


def _angle(a, b):
    cosine = (np.trace(a @ b.T) - 1) / 2
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_images_keep_ambiguity_and_resolve_metric_motion(fixture, profile):
    graph = Graph.load(EXAMPLE / "planar_camera_motion.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        ambiguous, metric = (graph.blocks[name].last_result for name in ("ambiguous", "metric"))
        assert ambiguous and metric and ambiguous.passed and metric.passed
        a, m = ambiguous.outputs, metric.outputs
        assert a["ambiguous"] and len(a["hypotheses"]) == 2 and a["pose"] is NO_VALUE
        assert a["report"]["scale_status"] == "unknown"
        assert all(candidate["metric_pose"] is None for candidate in a["hypotheses"])
        assert not m["ambiguous"] and len(m["hypotheses"]) == 1
        assert m["report"]["fit_inliers"] > 300
        assert m["report"]["fit_inliers"] == a["report"]["fit_inliers"]
        assert m["hypotheses"][0]["rms_pixels"] < 0.12
        transform = np.asarray(m["pose"]["source_to_target"])
        assert np.linalg.norm(transform[:3, 3] - fixture["TRANSLATION"]) < 0.00025
        assert _angle(transform[:3, :3], fixture["rotation_matrix"]()) < 0.01
        assert ("Source Image matches" in metric.drawings) == (profile == "debug")
        assert m["report"]["scale_status"] == "measured_plane_distance"
        assert (
            metric.payloads["source_image"]["spatial"] != metric.payloads["target_image"]["spatial"]
        )
    finally:
        graph.close()


def test_unequal_distorted_cameras_and_outliers_preserve_both_physical_solutions(fixture):
    inputs = _inputs(fixture)
    n = len(inputs["target_points"])
    # Associate real source rows with unrelated, valid in-grid target points.
    rng = np.random.default_rng(48)
    outliers = np.arange(0, n, 5)
    for i, replacement in zip(
        outliers, rng.uniform([5, 5], [635, 475], (len(outliers), 2)), strict=True
    ):
        inputs["target_points"][i] = replacement.tolist()
    tool = builtin_registry().create(TYPE, parameters={"ransac_pixels": 0.6})
    result = tool.execute(inputs)
    assert result and result.passed, tool.error
    out = result.outputs
    assert out["ambiguous"] and out["pose"] is NO_VALUE
    assert out["report"]["native_candidates"] == 4
    assert len(out["report"]["native_visible_indices"]) == 2
    assert (
        sum(item["status"] == "nonpositive_source_depth" for item in out["report"]["candidates"])
        == 2
    )
    keep = np.asarray(out["inliers"])
    expected = np.ones(n, bool)
    expected[outliers] = False
    np.testing.assert_array_equal(keep, expected)
    for candidate in out["hypotheses"]:
        normal = candidate["normal_source"]
        h = np.asarray(candidate["rotation"]) + np.outer(
            candidate["translation_over_plane_distance"], normal
        )
        np.testing.assert_allclose(h, out["report"]["normalized_homography"], atol=1e-7, rtol=0)
        assert candidate["rms_pixels"] < 0.0001
    angles = [
        _angle(np.asarray(candidate["rotation"]), fixture["rotation_matrix"]())
        for candidate in out["hypotheses"]
    ]
    assert min(angles) < 0.001 and max(angles) > 1
    best = out["hypotheses"][np.argmin(angles)]
    np.testing.assert_allclose(
        best["translation_over_plane_distance"],
        fixture["TRANSLATION"] / fixture["DISTANCE"],
        atol=1e-6,
    )


@pytest.mark.parametrize("model", ["depth", "cloud"])
def test_measured_plane_sets_scale_and_disambiguates_normal(fixture, model):
    normal = fixture["NORMAL"]
    d = fixture["DISTANCE"]
    if model == "depth":
        plane = {
            "model": "camera_depth_plane_v1",
            "frame": "source",
            "units": "m",
            "normal": (-normal).tolist(),
            "offset": d,
        }
    else:
        plane = {
            "schema": "point_cloud_model_v1",
            "kind": "plane",
            "frame": "source",
            "units": "mm",
            "coefficients": [*(normal * 3), -d * 3000],
        }
    tool = builtin_registry().create(TYPE)
    result = tool.execute({**_inputs(fixture), "source_plane": plane})
    assert result and result.passed, tool.error
    out = result.outputs
    assert not out["ambiguous"]
    assert out["report"]["source_plane_distance_m"] == pytest.approx(d)
    assert out["pose"]["units"] == "m"
    assert (out["pose"]["source_frame"], out["pose"]["target_frame"]) == ("source", "target")
    np.testing.assert_allclose(out["pose"]["translation"], fixture["TRANSLATION"], atol=1e-6)


def test_distance_alone_keeps_multiple_metric_poses(fixture):
    tool = builtin_registry().create(TYPE, parameters={"plane_distance_m": fixture["DISTANCE"]})
    result = tool.execute(_inputs(fixture, distance_only=True))
    assert result and result.passed, tool.error
    assert result.outputs["ambiguous"] and result.outputs["pose"] is NO_VALUE
    assert all(
        candidate["metric_pose"]["units"] == "m" for candidate in result.outputs["hypotheses"]
    )


def test_rotation_only_keeps_rotation_with_no_observed_plane_or_metric_pose(fixture):
    tool = builtin_registry().create(TYPE, parameters={"plane_distance_m": 1.0})
    result = tool.execute(_inputs(fixture, translation=np.zeros(3)))
    assert result and result.passed, tool.error
    out = result.outputs
    assert not out["ambiguous"] and out["pose"] is NO_VALUE
    assert out["report"]["scale_status"] == "rotation_only_model"
    assert len(out["hypotheses"]) == 1
    candidate = out["hypotheses"][0]
    assert (
        candidate["normal_source"] is None and candidate["translation_over_plane_distance"] is None
    )
    assert _angle(np.asarray(candidate["rotation"]), fixture["rotation_matrix"]()) < 0.005


def test_independent_target_depth_check_rejects_native_candidate():
    # Point source rays through a plane at normalized depth one, then move them behind camera two.
    rays = np.array([[0, 0, 1], [0.1, 0.2, 1], [-0.1, 0.15, 1], [0.13, -0.24, 1]])
    info, pose = _physical_candidate(
        np.eye(3),
        [0, 0, -2],
        [0, 0, 1],
        rays,
        rays[:, :2],
        np.eye(3),
        True,
        False,
        None,
        1.0,
        ("source", "target"),
        {"maximum_rms_px": 1.0},
    )
    assert pose is None
    assert info["status"] == "nonpositive_target_depth" and info["native_visible"]


def test_degenerate_foreign_grid_and_conflicting_anchor_fail_closed(fixture):
    registry = builtin_registry()
    tool = registry.create(TYPE)
    data = _inputs(fixture)
    line = [[x, 50] for x in range(20, 60)]
    failed = tool.execute({**data, "source_points": line, "target_points": line})
    assert failed and not failed.passed
    assert failed.fail_reasons[0]["code"] == "planar_motion_support"

    packet = ValuePacket(data["source_points"], {"spatial": frame(fixture["SIZE"], "foreign")})
    image = ValuePacket(
        np.zeros(fixture["SIZE"][::-1], np.uint8), {"spatial": frame(fixture["SIZE"], "source")}
    )
    assert tool.execute({**data, "source_points": packet, "source_image": image}) is None
    assert "image and points use different frames" in tool.error
    tool.parameters["plane_distance_m"] = 1.2
    plane = {
        "model": "camera_depth_plane_v1",
        "frame": "source",
        "units": "m",
        "normal": [0, 0, 1],
        "offset": -1.2,
    }
    assert tool.execute({**data, "source_plane": plane}) is None
    assert "either the connected plane or the configured plane prior" in tool.error
