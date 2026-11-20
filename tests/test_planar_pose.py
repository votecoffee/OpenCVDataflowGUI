"""Verify IPPE alternatives, calibrated scores and the original object frame."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import camera_record, rigid
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame

TYPE_ID = "3d.planar_pose_hypotheses"
EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/planar_pose_hypotheses"


def rotation_error(found, expected):
    return np.degrees(Rotation.from_matrix(found @ expected.T).magnitude())


def project(xyz, rot, translation, camera):
    return cv2.projectPoints(
        np.asarray(xyz, float),
        cv2.Rodrigues(rot)[0],
        translation,
        np.asarray(camera["camera_matrix"]),
        np.asarray(camera["distortion"]),
    )[0].reshape(-1, 2)


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_graph_distinguishes_near_frontal_and_oblique_views(profile):
    graph = Graph.load(EXAMPLE / "planar_pose_hypotheses.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        truth = json.loads((EXAMPLE / "assets/truth.json").read_bytes())["poses"]
        camera = graph.blocks["camera"].last_result.outputs["camera"]
        for view in ("near_frontal", "oblique"):
            block = graph.blocks[view]
            result = block.last_result
            assert result and result.passed, block.error
            out = result.outputs
            alternatives, report = out["hypotheses"], out["report"]
            assert report["native_count"] == report["plausible_count"] == len(alternatives) == 2
            assert alternatives[0]["rms_pixels"] <= alternatives[1]["rms_pixels"]
            observed = graph.blocks[f"detect_{view}"].last_result.outputs
            for item, diag in zip(alternatives, report["candidates"], strict=True):
                assert (item["source_frame"], item["target_frame"], item["units"]) == (
                    "board",
                    "camera",
                    "mm",
                )
                pose = rigid(item)
                pred = project(observed["object_points"], pose[:3, :3], pose[:3, 3], camera)
                error = pred - np.asarray(observed["image_points"])
                assert item["rms_pixels"] == pytest.approx(
                    np.sqrt(np.mean(np.sum(error**2, axis=1))), abs=1e-8
                )
                assert diag["status"] == "accepted" and diag["min_depth"] > 0
            src = graph.blocks[f"load_{view}"].last_result.outputs["image"]
            np.testing.assert_array_equal(out["image"], src)
            assert len(result.drawings) == (2 if profile == "debug" else 0)
            best = rigid(alternatives[0])
            known = np.asarray(truth[view])
            if view == "near_frontal":
                assert 0.02 < report["rms_gap_px"] < 0.2
                assert out["ambiguous"] is True and out["pose"] is NO_VALUE
                assert rotation_error(best[:3, :3], rigid(alternatives[1])[:3, :3]) > 2
            else:
                assert report["rms_gap_px"] > 4
                assert out["ambiguous"] is False and out["pose"] == alternatives[0]
                assert rotation_error(best[:3, :3], known[:3, :3]) < 0.1
                assert np.linalg.norm(best[:3, 3] - known[:3, 3]) < 0.3
    finally:
        graph.close()


@pytest.fixture
def tilted_square():
    cam = camera_record(
        [[730, 0, 343], [0, 710, 258], [0, 0, 1]],
        [0.08, -0.02, 0.01, -0.007, 0.005],
        (640, 480),
        "mm",
        "lens",
    )
    xyz = np.array([[-56, 56, 0], [56, 56, 0], [56, -56, 0], [-56, -56, 0]], float)
    rotation = Rotation.from_rotvec([0.28, -0.35, 0.08]).as_matrix()
    translation = np.array([8.0, 15.0, 570.0])
    return cam, xyz, rotation, translation


@pytest.mark.parametrize("method", ["ippe", "ippe_square", "transformed_plane"])
def test_distortion_and_supplied_object_frame(tilted_square, method):
    cam, xyz, rot, translation = tilted_square
    xy = project(xyz, rot, translation, cam)
    if method == "transformed_plane":
        basis = Rotation.from_rotvec([0.26, 0.44, -0.21]).as_matrix()
        origin = np.array([223.0, -100.0, 78.0])
        xyz = xyz @ basis.T + origin
        rot = rot @ basis.T
        translation = translation - rot @ origin
        method = "ippe"
    block = builtin_registry().create(
        TYPE_ID, parameters={"method": method, "source_frame": "fixture"}
    )
    result = block.execute(
        {"camera": cam, "object_points": xyz.tolist(), "image_points": xy.tolist()}
    )
    assert result and result.passed and not result.outputs["ambiguous"], block.error
    chosen = result.outputs["pose"]
    assert (chosen["source_frame"], chosen["target_frame"], chosen["units"]) == (
        "fixture",
        "lens",
        "mm",
    )
    found = rigid(chosen)
    np.testing.assert_allclose(found[:3, 3], translation, atol=1e-5)
    np.testing.assert_allclose(found[:3, :3], rot, atol=1e-7)


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("plane", "coplanar"),
        ("square_order", "centered Z=0"),
        ("units", "units must match"),
        ("grid", "must match the calibrated camera"),
        ("collinear", "non-collinear image points"),
    ],
)
def test_invalid_geometry_units_and_grid(tilted_square, case, error):
    cam, xyz, rot, translation = tilted_square
    xy = project(xyz, rot, translation, cam)
    config = {}
    if case == "plane":
        xyz[0, 2] = 1
    elif case == "square_order":
        xyz, xy = xyz[::-1], xy[::-1]
        config["method"] = "ippe_square"
    elif case == "units":
        xyz = ValuePacket(xyz.tolist(), {"object_point_units": "m"})
    elif case == "grid":
        xy = ValuePacket(xy.tolist(), {"spatial": frame([320, 240], "wrong-grid")})
    elif case == "collinear":
        xy[:, 1] = 220
    block = builtin_registry().create(TYPE_ID, parameters=config)
    values = {"camera": cam, "object_points": xyz, "image_points": xy}
    values = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in values.items()}
    assert block.execute(values) is None
    assert error in block.error


def test_limits_candidate_reprojection_and_reports_failed_fit(tilted_square):
    cam, xyz, rot, translation = tilted_square
    xy = project(xyz, rot, translation, cam)
    reg = builtin_registry()
    inputs = {"camera": cam, "object_points": xyz.tolist(), "image_points": xy.tolist()}
    gated = reg.create(TYPE_ID, parameters={"max_rms_px": 0.1}).execute(inputs)
    assert gated and gated.passed
    assert [row["status"] for row in gated.outputs["report"]["candidates"]] == [
        "accepted",
        "reprojection_limit",
    ]
    assert gated.outputs["pose"] == gated.outputs["hypotheses"][0]
    xy[0] += [5, 4]
    inputs["image_points"] = xy.tolist()
    failed = reg.create(TYPE_ID, parameters={"max_rms_px": 0.05}).execute(inputs)
    assert failed and not failed.passed
    assert failed.outputs["pose"] is NO_VALUE and failed.outputs["hypotheses"] == []
    assert failed.fail_reasons[0]["code"] == "planar_pose_no_candidate"
    assert all(
        row["status"] == "reprojection_limit" for row in failed.outputs["report"]["candidates"]
    )


@pytest.mark.parametrize("mode", ["duplicate", "behind", "empty"])
def test_rejects_duplicate_or_behind_camera_candidates(tilted_square, monkeypatch, mode):
    cam, xyz, rotation, translation = tilted_square
    rv = cv2.Rodrigues(rotation)[0]
    tv = (translation / 56).reshape(3, 1)

    def fake_solver(*args, **kwargs):
        if mode == "empty":
            return 0, (), (), None
        alternative = tv if mode == "duplicate" else -tv
        return 2, (rv, rv), (alternative, tv), None

    monkeypatch.setattr(cv2, "solvePnPGeneric", fake_solver)
    block = builtin_registry().create(TYPE_ID, parameters={"method": "ippe_square"})
    pixels = project(xyz, rotation, translation, cam)
    result = block.execute(
        {"camera": cam, "object_points": xyz.tolist(), "image_points": pixels.tolist()}
    )
    assert result is not None, block.error
    assert result.outputs["ambiguous"] is False
    if mode == "empty":
        assert not result.passed and result.outputs["hypotheses"] == []
        assert result.outputs["report"]["candidates"] == []
    else:
        assert result.passed and len(result.outputs["hypotheses"]) == 1
        reasons = {"duplicate": "duplicate_solution", "behind": "nonpositive_depth"}
        assert {item["status"] for item in result.outputs["report"]["candidates"]} == {
            "accepted",
            reasons[mode],
        }
