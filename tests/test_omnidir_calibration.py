"""Validate the omnidirectional fit with rendered targets and native view exclusions."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import camera
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/omnidir_calibration"
TRUTH = json.loads((BASE / "truth.json").read_bytes())
SIZE = tuple(TRUTH["size"])
K = np.array(TRUTH["camera_matrix"])
XI = TRUTH["xi"]
LOCK_DISTORTION = dict.fromkeys(("fix_k1", "fix_k2", "fix_p1", "fix_p2"), True)


def mei_projection(xyz, k=K, xi=XI):
    scale = xyz[..., 2] + xi * np.linalg.norm(xyz, axis=-1)
    return xyz[..., :2] / scale[..., None] * np.diag(k)[:2] + k[:2, 2]


@pytest.mark.parametrize("run_profile", ["debug", "lean"])
def test_saved_camera_validates_edge_target_and_rectifies(run_profile):
    graph = Graph.load(BASE / "omnidir_calibration.json", builtin_registry())
    try:
        validation = validate_graph(graph)
        assert validation["valid"], validation["issues"]
        graph.set_run_profile(run_profile)
        graph.run()
        blocks = {block.id: block for block in graph.processing_blocks()}
        assert all(b.state == "pass" for b in blocks.values()), {
            b.id: b.error for b in blocks.values()
        }
        outputs = blocks["fit"].last_result.outputs
        model, report = outputs["camera"], outputs["fit_report"]
        k, d, size = camera(model)
        assert size == SIZE and model["lens"] == "omnidir" and model["units"] == "mm"
        assert model["xi"] == report["xi"] == pytest.approx(XI, abs=0.025)
        assert np.allclose(k, K, atol=8.0) and not d.any()
        assert report["used_view_indices"] == list(range(len(TRUTH["training"])))
        assert [v["name"] for v in report["views"]] == TRUTH["training"]
        assert not report["excluded_views"] and report["coverage"] > 0.55
        assert report["rms_pixels"] < 0.35
        squared = [view["rms_pixels"] ** 2 for view in report["views"]]
        weights = [view["point_count"] for view in report["views"]]
        assert outputs["rms"] == pytest.approx(np.sqrt(np.average(squared, weights=weights)))

        review = blocks["validate"].last_result
        assert review.outputs["report"]["model"] == "omnidir_validation_v1"
        assert review.outputs["rms"] < 0.25 and review.outputs["max_error"] < 0.5
        assert min(p["observed"][0] for p in review.outputs["report"]["points"]) < SIZE[0] / 16
        assert bool(review.drawings) is (run_profile == "debug")

        rectified = blocks["rectify"].last_result
        assert rectified.outputs["camera"]["lens"] == "pinhole"
        new_k, _, _ = camera(rectified.outputs["camera"])
        mx, my = cv2.omnidir.initUndistortRectifyMap(
            k,
            d,
            np.array([model["xi"]]),
            np.eye(3),
            new_k,
            SIZE,
            cv2.CV_32FC1,
            cv2.omnidir.RECTIFY_PERSPECTIVE,
        )
        y, x = np.indices(SIZE[::-1], dtype=np.float64)
        rays = np.stack(
            ((x - new_k[0, 2]) / new_k[0, 0], (y - new_k[1, 2]) / new_k[1, 1], np.ones_like(x)),
            axis=-1,
        )
        error = np.linalg.norm(np.stack((mx, my), axis=-1) - mei_projection(rays), axis=-1)
        assert error.max() < 2.4
        assert np.mean((mx >= 0) & (mx <= 639) & (my >= 0) & (my <= 479)) > 0.999
        image = blocks["heldout"].last_result.outputs["image"]
        np.testing.assert_array_equal(
            rectified.outputs["image"], cv2.remap(image, mx, my, cv2.INTER_LINEAR)
        )
        assert rectified.payloads["image"]["spatial"]["root_id"] == "rectify:rectified"
    finally:
        graph.close()


def observations_with_outlier_view():
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
    corners = board.getChessboardCorners().astype(np.float64)
    rng = np.random.default_rng(424)
    entries = []
    for i in range(15):
        rotation = cv2.Rodrigues(rng.uniform(-0.5, 0.5, 3))[0]
        translation = np.r_[rng.uniform([-135, -89], [45, 41]), rng.uniform(93, 160)]
        ids = np.flatnonzero(np.arange(len(corners)) % (i + 3) != 0)
        xyz = corners[ids]
        camera_xyz = (xyz - [10.0, 10.0, 0.0]) @ rotation.T + translation
        pixels = mei_projection(camera_xyz)
        valid = np.all((pixels > [5, 5]) & (pixels < [635, 475]), axis=1)
        view = {
            "model": "charuco_observation_v1",
            "board": spec,
            "image_size": list(SIZE),
            "corner_ids": ids[valid].tolist(),
            "image_points": pixels[valid].tolist(),
            "object_points": xyz[valid].tolist(),
        }
        entries.append(ValuePacket(view, {}))
    # A distinctly shuffled view has plausible individual corners and no coherent pose.
    broken = {**entries[2].value}
    pairs = np.array(broken["image_points"])
    broken["image_points"] = pairs[np.random.default_rng(381).permutation(len(pairs))].tolist()
    entries.insert(3, ValuePacket(broken, {}))
    return FlowGroup(tuple(entries), tuple(str(i) for i in range(len(entries))))


def test_native_view_selection_keeps_original_indices_and_requires_explicit_acceptance(tmp_path):
    registry = builtin_registry()
    fit = registry.create(
        "3d.omnidir_calibrate", parameters={**LOCK_DISTORTION, "minimum_views": 10}
    )
    inputs = {"observations": observations_with_outlier_view()}
    strict = fit.execute(inputs)
    assert strict is not None, fit.error
    assert not strict.passed and strict.outputs["camera"] is NO_VALUE
    assert {r["code"] for r in strict.fail_reasons} == {"omnidir.excluded_views"}
    report = strict.outputs["fit_report"]
    assert {"index": 3, "name": "view-4"} in report["excluded_views"]
    assert [v["name"] for v in report["views"]] == [
        f"view-{i + 1}" for i in report["used_view_indices"]
    ]
    assert len({v["point_count"] for v in report["views"]}) > 1
    assert report["worst_view_index"] in report["used_view_indices"]

    fit.parameters["view_policy"] = "allow_subset"
    accepted = fit.execute(inputs)
    assert accepted and accepted.passed, fit.error
    assert accepted.outputs["fit_report"] == report
    model = accepted.outputs["camera"]
    assert model["xi"] == pytest.approx(XI, abs=2e-4)
    assert model["rms_pixels"] < 0.0001 and model["views"] == len(report["views"])
    assert all(v["pose"]["source_frame"] == "calibration_target" for v in report["views"])

    path = tmp_path / "omni.json"
    writer = registry.create("output.calibration", parameters={"path": str(path)})
    assert writer.execute({"profile": model}).passed
    reader = registry.create("input.calibration", parameters={"path": str(path)})
    assert reader.execute({}).outputs["profile"] == model

    fit.parameters["minimum_views"] = len(report["views"]) + 1
    insufficient = fit.execute(inputs)
    assert insufficient and not insufficient.passed
    assert insufficient.outputs["camera"] is NO_VALUE
    assert {r["code"] for r in insufficient.fail_reasons} == {"omnidir.insufficient_views"}
