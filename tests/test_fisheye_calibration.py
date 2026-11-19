"""Qualify fisheye fitting, independent held-out validation, and remap coverage."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import camera, camera_record
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/fisheye_calibration"
TRUTH = json.loads((EXAMPLE / "truth.json").read_bytes())
SIZE = tuple(TRUTH["size"])
K = np.array(TRUTH["camera_matrix"])
D = np.array(TRUTH["distortion"])


def independent_projection(xy, k=K, d=D):
    radius = np.linalg.norm(xy, axis=-1)
    theta = np.arctan(radius)
    distorted = theta * (1 + sum(term * theta ** (2 * i) for i, term in enumerate(d, 1)))
    scale = np.divide(distorted, radius, out=np.ones_like(radius), where=radius > 0)
    return xy * scale[..., None] * np.diag(k)[:2] + k[:2, 2]


@pytest.mark.parametrize("run_profile", ["debug", "lean"])
def test_saved_fit_validates_separate_image_and_rectifies_with_known_coverage(run_profile):
    graph = Graph.load(EXAMPLE / "fisheye_calibration.json", builtin_registry())
    try:
        check = validate_graph(graph)
        assert check["valid"], check["issues"]
        graph.set_run_profile(run_profile)
        graph.run()
        assert all(block.state == "pass" for block in graph.processing_blocks()), {
            block.id: block.error for block in graph.processing_blocks()
        }
        fit = graph.blocks["fit"].last_result.outputs
        report = fit["fit_report"]
        profile = fit["camera"]
        k, d, size = camera(profile)
        assert size == SIZE and profile["lens"] == "fisheye" and profile["units"] == "mm"
        np.testing.assert_allclose(k, K, atol=1.6)
        np.testing.assert_allclose(d, D, atol=0.004)
        assert 0 < fit["rms"] < 0.25 and 0.37 < report["coverage"] < 0.41
        assert len(report["views"]) == TRUTH["training"]
        assert all(view["name"].startswith("training-") for view in report["views"])
        weights = [view["point_count"] for view in report["views"]]
        vals = [view["rms_pixels"] ** 2 for view in report["views"]]
        assert fit["rms"] == pytest.approx(np.sqrt(np.average(vals, weights=weights)), abs=1e-8)

        check = graph.blocks["validate"].last_result
        assert check.outputs["report"]["model"] == "fisheye_validation_v1"
        assert check.outputs["rms"] < 0.22 and check.outputs["max_error"] < 0.45
        assert bool(check.drawings) is (run_profile == "debug")

        rectified = graph.blocks["rectify"].last_result
        assert rectified.outputs["camera"]["lens"] == "pinhole"
        new_k, _, _ = camera(rectified.outputs["camera"])
        yy, xx = np.indices(SIZE[::-1], dtype=np.float64)
        rays = (np.stack((xx, yy), axis=-1) - new_k[:2, 2]) / np.diag(new_k)[:2]
        truth_map = independent_projection(rays)
        native = cv2.fisheye.initUndistortRectifyMap(k, d, None, new_k, SIZE, cv2.CV_32FC1)
        pixel_map = np.stack(native, axis=-1)
        assert np.max(np.linalg.norm(pixel_map - truth_map, axis=-1)) < 2.0
        assert np.mean(np.all((pixel_map >= 0) & (pixel_map <= [639, 479]), axis=-1)) > 0.999
        image = graph.blocks["heldout"].last_result.outputs["image"]
        np.testing.assert_array_equal(
            rectified.outputs["image"], cv2.remap(image, *native, cv2.INTER_LINEAR)
        )
        assert rectified.payloads["image"]["spatial"]["root_id"] == "rectify:rectified"
    finally:
        graph.close()


def test_rejected_fit_retains_diagnostics_and_withholds_camera():
    graph = Graph.load(EXAMPLE / "fisheye_calibration.json", builtin_registry())
    try:
        fit = graph.blocks["fit"]
        fit.parameters.update(minimum_coverage=0.6, max_rms_pixels=0.1, max_view_rms_pixels=0.15)
        graph.run()
        result = fit.last_result
        assert fit.state == "fail" and result is not None
        assert result.outputs["camera"] is NO_VALUE and result.outputs["rms"] > 0.1
        assert result.outputs["fit_report"]["worst_view_rms_pixels"] > 0.15
        assert {r["code"] for r in result.fail_reasons} == {
            "fisheye.overall_rms",
            "fisheye.view_rms",
            "fisheye.coverage",
        }
        assert graph.blocks["validate"].last_result is None
    finally:
        graph.close()


def synthetic_observations():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((10, 7), 10.0, 7.0, dictionary)
    spec = {
        "columns": 10,
        "rows": 7,
        "square_length": 10.0,
        "marker_length": 7.0,
        "units": "mm",
        "dictionary": "4x4_50",
        "marker_ids": board.getIds().ravel().tolist(),
        "legacy_pattern": False,
    }
    entries = []
    xyz = board.getChessboardCorners().astype(np.float64)
    for i, (rvec, t) in enumerate(
        (
            ([0.15, 0.26, -0.06], [-42, -28, 105]),
            ([-0.26, 0.28, 0.14], [-83, -45, 102]),
            ([0.35, -0.23, -0.19], [4, -54, 105]),
            ([-0.18, -0.34, 0.23], [-91, 3, 112]),
            ([0.21, 0.16, 0.10], [4, 3, 116]),
            ([-0.42, 0.12, -0.28], [-25, -54, 124]),
        )
    ):
        ids = np.flatnonzero(np.arange(len(xyz)) % (i + 3) != 0)
        coords = xyz[ids]
        r = cv2.Rodrigues(np.array(rvec))[0]
        transformed = coords @ r.T + t
        xy = independent_projection(transformed[:, :2] / transformed[:, 2, None])
        inside = np.all((xy >= 0) & (xy < np.array(SIZE) - 1), axis=1)
        entries.append(
            ValuePacket(
                {
                    "model": "charuco_observation_v1",
                    "board": spec,
                    "image_size": list(SIZE),
                    "corner_ids": ids[inside].tolist(),
                    "image_points": xy[inside].tolist(),
                    "object_points": coords[inside].tolist(),
                },
                {},
            )
        )
    return FlowGroup(tuple(entries), tuple(str(i) for i in range(len(entries))))


def test_seeded_charuco_fit_accepts_ragged_views_and_saves_camera_profile(tmp_path):
    registry = builtin_registry()
    fit = registry.create(
        "3d.fisheye_calibrate",
        parameters={
            "fix_focal_length": True,
            "fix_principal_point": True,
            "fix_k3": True,
            "fix_k4": True,
            "camera_frame": "test-camera",
        },
    )
    seed = camera_record(K, [0, 0, 0, 0], SIZE, "mm", "approximate", "fisheye")
    result = fit.execute({"observations": synthetic_observations(), "initial_camera": seed})
    assert result and result.passed, fit.error
    profile = result.outputs["camera"]
    k, d, _ = camera(profile)
    np.testing.assert_array_equal(k, K)
    np.testing.assert_allclose(d, D, atol=1e-5)
    assert result.outputs["rms"] < 3e-5
    views = result.outputs["fit_report"]["views"]
    assert len({row["point_count"] for row in views}) > 1
    assert all(row["pose"]["target_frame"] == "test-camera" for row in views)
    path = tmp_path / "profile.json"
    writer = registry.create("output.calibration", parameters={"path": str(path)})
    assert writer.execute({"profile": profile}).passed
    reader = registry.create("input.calibration", parameters={"path": str(path)})
    assert reader.execute({}).outputs["profile"] == profile


def test_mismatched_seed_and_missing_required_seed_do_not_fit():
    registry = builtin_registry()
    fit = registry.create("3d.fisheye_calibrate", parameters={"fix_focal_length": True})
    group = synthetic_observations()
    assert fit.execute({"observations": group}) is None
    assert "initial camera" in fit.error and not fit.current_emissions
    seed = camera_record(K, D, [320, 240], "mm", "approximate", "fisheye")
    assert fit.execute({"observations": group, "initial_camera": seed}) is None
    assert "same pixel grid" in fit.error and not fit.current_emissions
