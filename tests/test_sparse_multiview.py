"""Verify calibrated sparse reconstruction against independent images and 3D points."""

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import camera_record, pose_record, rigid
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.sparse_multiview import VIEW_MODEL

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/sparse_multiview"
RECONSTRUCT = "3d.reconstruct_sparse_views"


def rms(errors):
    return np.sqrt(np.mean(np.sum(errors**2, axis=1)))


def project(xyz, pose, intrinsics, distortion=None):
    return cv2.projectPoints(
        np.asarray(xyz, float), cv2.Rodrigues(pose[:3, :3])[0], pose[:3, 3], intrinsics, distortion
    )[0].reshape(-1, 2)


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_graph_recovers_metric_landmarks_and_predicts_a_heldout_image(profile):
    registry = builtin_registry()
    graph = Graph.load(EXAMPLE / "sparse_multiview.json", registry)
    try:
        graph.set_run_profile(profile)
        graph.run()
        assert all(
            block.last_result and block.last_result.passed
            for block in graph.blocks.values()
            if block.id != "credit"
        ), {block.id: block.error for block in graph.blocks.values() if block.error}
        fixed = graph.blocks["fixed"].last_result.outputs
        joint = graph.blocks["joint"].last_result.outputs
        report = joint["report"]
        assert report["converged"] and report["anchor_ids"] == ["A", "B"]
        assert report["accepted_count"] == 35
        assert sum(len(row["view_ids"]) == 3 for row in report["landmarks"]) == 19
        assert report["rms_px"] < 0.14
        assert report["rms_px"] < fixed["report"]["rms_px"] * 0.3

        assets = EXAMPLE / "assets"
        truth = json.loads((assets / "truth.json").read_bytes())
        target = np.array(truth["landmarks_mm"])
        ids = joint["cloud"].metadata["landmark_ids"]
        assert joint["cloud"].frame == "board" and joint["cloud"].units == "mm"
        assert joint["cloud"].points.shape == (35, 3)
        assert sorted(ids) == list(range(35))
        assert rms(joint["cloud"].points - target[ids]) < 0.4
        assert rms(fixed["cloud"].points - target[fixed["cloud"].metadata["landmark_ids"]]) > 1.5
        original = {view["id"]: rigid(view["world_to_camera"]) for view in fixed["views"]}
        for view in joint["views"]:
            key = view["id"]
            recovered = rigid(view["world_to_camera"])
            if key in ("A", "B"):
                np.testing.assert_array_equal(recovered, original[key])
            else:
                expected = np.asarray(truth["world_to_camera"][key])
                center = recovered[:3, :3].T @ recovered[:3, 3]
                expected_center = expected[:3, :3].T @ expected[:3, 3]
                assert np.linalg.norm(center - expected_center) < 5
                relative = recovered[:3, :3] @ expected[:3, :3].T
                assert np.degrees(Rotation.from_matrix(relative).magnitude()) < 0.4

        camera = graph.blocks["intrinsics"].last_result.outputs["camera"]
        holdout = registry.create(
            "calibration.charuco", parameters={**truth["board"], "units": "mm"}
        )
        image = cv2.imdecode(
            np.frombuffer((assets / "E.png").read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE
        )
        detection = holdout.execute({"image": image, "camera": camera})
        assert detection and detection.passed
        observed = dict(zip(detection.outputs["corner_ids"], detection.outputs["image_points"]))
        for output, limit in ((joint, 0.24), (fixed, 0.8)):
            cloud = output["cloud"]
            prediction = project(
                cloud.points,
                np.array(truth["world_to_camera"]["E"]),
                np.array(truth["intrinsics"]),
            )
            assert (
                rms(prediction - [observed[key] for key in cloud.metadata["landmark_ids"]]) < limit
            )
    finally:
        graph.close()


@pytest.fixture
def warped_views():
    rng = np.random.default_rng(92345)
    xyz = rng.uniform([-0.35, -0.36, 1.3], [0.32, 0.35, 2.0], size=(20, 3))
    views = []
    for i, shift in enumerate((0, 0.22, 0.1, -0.2)):
        k = np.array([[340 + 10 * i, 0, 301], [0, 336 - 5 * i, 227], [0, 0, 1]], float)
        distortion = np.array([0.18, -0.05, 0.02, -0.01, 0.005])
        pose = np.eye(4)
        pose[:3, :3] = cv2.Rodrigues(np.array([shift / 5, 0.0, 0.0]))[0]
        pose[:3, 3] = [shift, shift * 0.3, 0]
        ids = [j for j in range(len(xyz) - 1) if (i + j) % 7 != 0]
        if i == 0:
            ids.append(len(xyz) - 1)
        ids.reverse()
        views.append(
            {
                "model": VIEW_MODEL,
                "id": i,
                "camera": camera_record(k, distortion, (640, 480)),
                "world_to_camera": pose_record(pose, "fixture"),
                "landmark_ids": [f"feat-{j}" for j in ids],
                "image_points": project(xyz[ids], pose, k, distortion).tolist(),
            }
        )
    return views, {f"feat-{j}": point for j, point in enumerate(xyz)}


@pytest.mark.parametrize("units", ["m", "mm", "unitless"])
@pytest.mark.parametrize("mode", ["points", "poses_and_points"])
def test_distorted_independent_views_preserve_units_and_landmark_id_alignment(
    warped_views, units, mode
):
    views, expected = warped_views
    count_scale = 1000 if units == "mm" else 1
    for view in views:
        transform = rigid(view["world_to_camera"])
        transform[:3, 3] *= count_scale
        view["world_to_camera"] = pose_record(transform, "fixture", units=units)
        view["camera"]["units"] = units
    block = builtin_registry().create(
        RECONSTRUCT,
        parameters={"maximum_error_px": 0.05, "refinement": mode, "anchor_view_ids": [1, 0]},
    )
    result = block.execute({"views": list(reversed(views))})
    assert result and result.passed, block.error
    if mode == "poses_and_points":
        assert result.outputs["report"]["anchor_ids"] == [1, 0]
        for key in (0, 1):
            recovered = next(view for view in result.outputs["views"] if view["id"] == key)
            assert recovered["world_to_camera"] == views[key]["world_to_camera"]
    cloud = result.outputs["cloud"]
    assert cloud.units == units and cloud.frame == "fixture"
    assert len(cloud.points) == 19
    np.testing.assert_allclose(
        cloud.points / count_scale,
        [expected[key] for key in cloud.metadata["landmark_ids"]],
        atol=1e-5,
    )
    entries = result.outputs["report"]["landmarks"]
    single = next(entry for entry in entries if entry["id"] == "feat-19")
    assert single["reason"] == "too_few_views"
    assert "feat-19" not in cloud.metadata["landmark_ids"]


def test_wire_view_packer_checks_grid_metadata_and_duplicate_ids(warped_views):
    views, _ = warped_views
    values = {key: val for key, val in views[0].items() if key not in ("model", "id")}
    block = builtin_registry().create("3d.landmark_view", parameters={"view_id": "packed"})
    assert block.execute(values).outputs["view"]["id"] == "packed"
    pixels = values["image_points"]
    mismatch = ValuePacket(pixels, {"spatial": frame([580, 400])})
    assert block.execute({**values, "image_points": mismatch}) is None
    assert "same grid" in block.error
    ids = values["landmark_ids"]
    assert block.execute({**values, "landmark_ids": [*ids[:-1], ids[0]]}) is None
    assert "unique" in block.error


def test_reprojection_checks_and_weak_parallax_flag_bad_landmarks(warped_views):
    views, _ = warped_views
    pixels = views[1]["image_points"][0]
    pixels[0] += 15
    corrupt = views[1]["landmark_ids"][0]
    reg = builtin_registry()
    checked = reg.create(RECONSTRUCT, parameters={"maximum_error_px": 1.0})
    result = checked.execute({"views": views})
    assert result and result.passed, checked.error
    row = next(row for row in result.outputs["report"]["landmarks"] if row["id"] == corrupt)
    assert row["reason"] == "reprojection_limit"
    assert result.outputs["report"]["accepted_count"] == 18
    weak = reg.create(RECONSTRUCT, parameters={"minimum_ray_angle_deg": 85.0})
    failed = weak.execute({"views": views})
    assert failed and not failed.passed
    assert failed.outputs["cloud"] is failed.outputs["views"] is NO_VALUE
    assert {row["reason"] for row in failed.outputs["report"]["landmarks"]} == {
        "weak_parallax",
        "too_few_views",
    }


def test_joint_fit_rejects_unanchored_camera_or_unconverged_output(warped_views):
    views, _ = warped_views
    views[3]["world_to_camera"]["source_to_target"][0][3] += 0.025
    registry = builtin_registry()
    block = registry.create(
        RECONSTRUCT,
        parameters={
            "refinement": "poses_and_points",
            "anchor_view_ids": [0, 1],
            "maximum_evaluations": 1,
        },
    )
    result = block.execute({"views": views})
    assert result and not result.passed
    assert result.outputs["report"]["converged"] is False
    assert result.outputs["cloud"] is result.outputs["views"] is NO_VALUE
    assert result.fail_reasons[0]["code"] == "multiview_refinement_did_not_converge"
    sparse = copy.deepcopy(views)
    sparse[3]["landmark_ids"] = sparse[3]["landmark_ids"][:3]
    sparse[3]["image_points"] = sparse[3]["image_points"][:3]
    assert block.execute({"views": sparse}) is None
    assert "sufficient parallax in both fixed anchors" in block.error


def test_file_replay_validates_view_identity_and_world_frame(tmp_path, warped_views):
    views, _ = warped_views
    path = tmp_path / "views.json"
    path.write_text(json.dumps(views), encoding="utf-8")
    registry = builtin_registry()
    block = registry.create(RECONSTRUCT, parameters={"path": path.name, "refinement": "none"})
    with_graph = Graph()
    with_graph.add(block)
    project_path = tmp_path / "replay.json"
    with_graph.save(project_path)
    with_graph.close()
    graph = Graph.load(project_path, registry)
    try:
        assert graph.blocks[block.id].execute({}).passed
    finally:
        graph.close()
    direct = registry.create(RECONSTRUCT)
    views[3]["world_to_camera"]["source_frame"] = "another-world"
    assert direct.execute({"views": views}) is None
    assert "share a world frame" in direct.error
    views[3]["world_to_camera"]["source_frame"] = "fixture"
    views[3]["id"] = views[2]["id"]
    assert direct.execute({"views": views}) is None
    assert "View IDs must be unique" in direct.error
