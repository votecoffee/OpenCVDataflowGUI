"""Qualify fixed intrinsics, disjoint target regions and held-out frame closure."""

import copy
import json
from pathlib import Path
from runpy import run_path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import camera_record, pose_record
from opencv_dataflow.payloads import NO_VALUE

EXAMPLE = (
    Path(__file__).resolve().parents[1] / "examples/opencv tools/nonoverlap_camera_registration"
)


@pytest.fixture
def fixture():
    return run_path(str(EXAMPLE / "build_example.py"))


def _angle(left, right):
    cos = (np.trace(left @ right.T) - 1) / 2
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_pair_fits_known_metric_geometry_and_independent_closure(profile, fixture):
    graph = Graph.load(EXAMPLE / "nonoverlap_camera_registration.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        outputs = {}
        for key, second_lens in (
            ("pinhole", "pinhole"),
            ("mixed", "fisheye"),
            ("drift", "fisheye"),
        ):
            outcome = graph.blocks[key].last_result
            assert outcome is not None, graph.blocks[key].error
            assert outcome.passed == (key != "drift")
            out = outputs[key] = outcome.outputs
            report = out["report"]
            assert report["lens_models"] == ["pinhole", second_lens]
            assert report["initial_pose_used"] == (key != "pinhole")
            known = np.array(out["pose"]["source_to_target"])
            assert out["pose"]["units"] == "mm"
            assert (out["pose"]["source_frame"], out["pose"]["target_frame"]) == fixture["FRAMES"]
            assert np.linalg.norm(known[:3, 3] - fixture["SHIFT"]) < 0.1
            assert _angle(known[:3, :3], fixture["ROTATION"]) < 0.005
            assert report["fit_rms_px"] < 0.028
            assert len(report["fit_views"]) == 8
            assert all(view["common_points"] == 0 for view in report["fit_views"])
            assert any(
                view["point_counts"][0] != view["point_counts"][1] for view in report["fit_views"]
            )
            assert len(report["holdout"]["views"]) == 4
            assert all(view["common_points"] == 0 for view in report["holdout"]["views"])
            assert (out["fundamental"] is NO_VALUE) == (second_lens == "fisheye")
            assert ("Held-out cross-projection" in outcome.graphics) == (profile == "debug")

            # Native E must reproduce the known normalized-ray epipolar geometry.
            x, y, z = known[:3, 3]
            cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            expected = cross @ known[:3, :3]
            actual = np.array(out["essential"])
            np.testing.assert_allclose(
                actual / np.linalg.norm(actual), expected / np.linalg.norm(expected), atol=1e-9
            )
            if second_lens == "pinhole":
                expected = np.linalg.inv(fixture["K2"]).T @ expected @ np.linalg.inv(fixture["K1"])
                actual = np.array(out["fundamental"])
                pivot = np.unravel_index(np.argmax(np.abs(expected)), expected.shape)
                np.testing.assert_allclose(
                    actual / actual[pivot], expected / expected[pivot], atol=1e-9
                )
            if key != "drift":
                assert report["holdout"]["rms_px"] < 0.06
                assert max(v["frame_closure_translation"] for v in report["holdout"]["views"]) < 0.5
                assert (
                    max(v["frame_closure_rotation_deg"] for v in report["holdout"]["views"]) < 0.05
                )
        assert outputs["drift"]["report"]["holdout"]["rms_px"] > 8
        np.testing.assert_allclose(
            outputs["drift"]["pose"]["source_to_target"],
            outputs["mixed"]["pose"]["source_to_target"],
            atol=1e-6,
        )
    finally:
        graph.close()


def _inputs(fixture):
    def read(file):
        return json.loads((EXAMPLE / "assets" / file).read_text())

    a, b = fixture["FRAMES"]
    return {
        "observations": read("fit_mixed.json"),
        "holdout": read("held_mixed.json"),
        "first_camera": camera_record(
            fixture["K1"],
            fixture["BROWN"],
            fixture["SIZES"][0],
            "mm",
            a,
            "pinhole",
        ),
        "second_camera": camera_record(
            fixture["K2"],
            fixture["FISHEYE"],
            fixture["SIZES"][1],
            "mm",
            b,
            "fisheye",
        ),
    }


@pytest.mark.local_examples
def test_native_call_receives_only_paired_fit_points_and_column_seed(fixture, monkeypatch):
    inputs = _inputs(fixture)
    data = inputs["observations"]
    data["visibility"][1][-1] = 0
    data["image_points"][1][-1] = None
    seed = np.eye(4)
    seed[:3, 3] = [-1.18, 0.055, 0.018]
    inputs["initial_pose"] = pose_record(seed, *fixture["FRAMES"], "m")
    original = copy.deepcopy(inputs)
    real = cv2.registerCamerasExtended

    def check(*args, **kwargs):
        first, second = args[:2]
        assert len(first) == len(second) == 7
        assert all(x.shape[1] == 3 for x in (*first, *second))
        assert all(x.max(axis=0)[0] < y.min(axis=0)[0] for x, y in zip(first, second, strict=True))
        assert args[11].shape == (3, 1)
        np.testing.assert_allclose(args[11].ravel(), [-1180, 55, 18])
        assert args[6] == cv2.CALIB_MODEL_PINHOLE and args[9] == cv2.CALIB_MODEL_FISHEYE
        assert kwargs["flags"] == cv2.CALIB_USE_EXTRINSIC_GUESS
        return real(*args, **kwargs)

    monkeypatch.setattr(cv2, "registerCamerasExtended", check)
    block = builtin_registry().create("3d.register_cameras")
    result = block.execute(inputs)
    assert result and result.passed, block.error
    assert result.outputs["report"]["unpaired_fit_capture_ids"] == [data["capture_ids"][-1]]
    assert inputs == original


@pytest.mark.local_examples
@pytest.mark.parametrize("problem", ["duplicate", "renamed_duplicate", "target", "camera"])
def test_bad_calibration_provenance_stops_before_native_fit(problem, fixture, monkeypatch):
    inputs = _inputs(fixture)
    error = "must be disjoint"
    if problem == "duplicate":
        inputs["holdout"]["capture_ids"][0] = inputs["observations"]["capture_ids"][0]
    elif problem == "renamed_duplicate":
        inputs["holdout"]["image_points"][0][0] = inputs["observations"]["image_points"][0][0]
        error = "repeats a fit observation"
    elif problem == "target":
        inputs["holdout"]["object_points"][0][0] += 0.5
        error = "same optics frames and target geometry"
    else:
        inputs["second_camera"]["frame"] = "wrong_camera"
        error = "must match observation frames"
    monkeypatch.setattr(cv2, "registerCamerasExtended", lambda *a, **kw: pytest.fail("native fit"))
    block = builtin_registry().create("3d.register_cameras")
    assert block.execute(inputs) is None
    assert error in block.error
