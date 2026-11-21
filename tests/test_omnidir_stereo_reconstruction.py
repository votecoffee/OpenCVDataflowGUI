"""Verify native wide-angle disparity, metric geometry and per-pixel color."""

import copy
from pathlib import Path
from runpy import run_path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.calibration_revisions import APPLIED, CAPTURE, REVISION
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import ValuePacket, frame, packet_for
from opencv_dataflow.stereo_omnidir_reconstruction import ReconstructOmnidirStereo, _triangulate

EXAMPLE = (
    Path(__file__).resolve().parents[1] / "examples/opencv tools/omnidir_stereo_reconstruction"
)
MODES = ("perspective", "longitude_latitude")


@pytest.fixture(scope="module")
def fixture():
    return run_path(str(EXAMPLE / "build_example.py"))


def _run_graph(profile="debug"):
    graph = Graph.load(EXAMPLE / "omnidir_stereo_reconstruction.json", builtin_registry())
    graph.set_run_profile(profile)
    graph.set_branch_workers(2)
    graph.run()
    return graph


def _native(input_pair, rig, report, settings):
    cameras = [rig[side] for side in ("left", "right")]
    arguments = [
        item
        for cam in cameras
        for item in (
            np.array(cam["camera_matrix"]),
            np.array(cam["distortion"], dtype=np.float64),
            np.array([cam["xi"]]),
        )
    ]
    transform = np.array(rig["transform"]["source_to_target"])
    flag = (
        cv2.omnidir.RECTIFY_PERSPECTIVE
        if report["projection"] == "perspective"
        else cv2.omnidir.RECTIFY_LONGLATI
    )
    return cv2.omnidir.stereoReconstruct(
        *input_pair,
        *arguments,
        transform[:3, :3],
        transform[:3, 3],
        flag,
        settings["num_disparities"],
        settings["block_size"],
        newSize=tuple(report["image_size"]),
        Knew=np.array(report["virtual_matrix"]),
        pointType=cv2.omnidir.XYZ,
    )


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_stereo_pair_retains_supported_disparity_and_true_colored_xyz(fixture, profile):
    graph = _run_graph(profile)
    try:
        rig = graph.blocks["rig"].last_result.outputs["rig"]
        original = [graph.blocks[side].last_result.outputs["image"] for side in ("left", "right")]
        for mode in MODES:
            block = graph.blocks[mode]
            result = block.last_result
            assert result and result.passed, block.error
            o = result.outputs
            report = o["report"]
            assert report["projection"] == mode and report["native_disparity_scale"] == 1
            assert report["baseline_m"] == pytest.approx(0.19)
            np.testing.assert_allclose(report["left_rotation"], np.eye(3), atol=1e-12)
            native, left, right, _ = _native(original, rig, report, block.parameters)
            valid = o["valid_mask"] != 0
            assert (
                valid.sum() == report["valid_pixels"] and report["valid_at_or_below_15_px"] > 70_000
            )
            np.testing.assert_array_equal(o["disparity"][valid], native[valid])
            np.testing.assert_array_equal(o["left"], left)
            np.testing.assert_array_equal(o["right"], right)
            assert np.isnan(o["disparity"][~valid]).all()
            assert np.isnan(o["depth"][~valid]).all()
            assert not valid[:, : block.parameters["num_disparities"]].any()
            assert (native[np.where(~valid)] > 0).sum() > 1000  # Mask or rectification footprint.

            cloud = o["cloud"]
            x, y = cloud.pixels.T
            np.testing.assert_array_equal(cloud.colors, left[y, x, ::-1])
            assert np.any(cloud.colors[:, 0] != cloud.colors[:, 2])
            np.testing.assert_array_equal(cloud.points[:, 2], o["depth"][y, x])
            assert cloud.units == "m" and cloud.frame == "omni_left:rectified"
            left_payload, right_payload = (result.payloads[side] for side in ("left", "right"))
            assert left_payload["spatial"] != right_payload["spatial"]
            assert (
                left_payload["omnidir_stereo_grid"]["sha256"]
                == right_payload["omnidir_stereo_grid"]["sha256"]
            )
            assert (
                left_payload["omnidir_stereo_grid"]["source"]
                == graph.blocks["left"].last_result.payloads["image"]["spatial"]
            )
            assert (
                left_payload["spatial"]
                == cloud.metadata["spatial"]
                == result.payloads["depth"]["spatial"]
            )
            assert o["depth"].units == "m"

            rays = fixture["ideal_rays"](fixture["SIZE"], np.array(report["virtual_matrix"]), mode)
            near, far = (rays * (z / rays[..., 2:3]) for z in (fixture["NEAR"], fixture["FAR"]))
            foreground = (near[..., 0] > -0.22) & (near[..., 0] < 0.24)
            foreground &= (near[..., 1] > -0.20) & (near[..., 1] < 0.12)
            background = (far[..., 0] > 0.72) & (far[..., 0] < 1.35) & (abs(far[..., 1]) < 0.30)
            for known, region, tolerance in ((near, foreground, 0.023), (far, background, 0.095)):
                inspect = region[y, x]
                assert inspect.sum() > 160
                independently_rendered = known[y[inspect], x[inspect]]
                errs = np.linalg.norm(cloud.points[inspect] - independently_rendered, axis=1)
                assert np.percentile(errs, 90) < tolerance
            # Forward-project the reconstructed points to the original right capture.
            right_xyz = cloud.points @ fixture["ROTATION"].T + fixture["TRANSLATION"]
            uv = right_xyz[:, :2] / (
                right_xyz[:, 2:]
                + fixture["XI"][1] * np.linalg.norm(right_xyz, axis=1, keepdims=True)
            )
            raw_right = uv * np.diag(fixture["K2"])[:2] + fixture["K2"][:2, 2]
            hx, hy, w, h = fixture["RIGHT_HOLE"]
            inside = (
                (raw_right[:, 0] > hx + 1)
                & (raw_right[:, 0] < hx + w - 2)
                & (raw_right[:, 1] > hy + 1)
                & (raw_right[:, 1] < hy + h - 2)
            )
            assert not inside.any()
            assert ("point cloud" in result.graphics) == (profile == "debug")
    finally:
        graph.close()


@pytest.mark.parametrize("mode", MODES)
def test_metric_lifting_recovers_independent_epipolar_ray_intersections(fixture, mode):
    x, y = np.meshgrid([84, 112, 157, 192, 239, 274, 301], [46, 92, 124, 168, 191])
    x, y = x.ravel(), y.ravel()
    width, height = fixture["SIZE"]
    cx, cy = (width - 1) / 2, (height - 1) / 2
    fx, fy = 204.0, 180.0
    baseline = 0.19
    if mode == "longitude_latitude":
        cx, cy = cx - fx * np.pi / 2, cy - fy * np.pi / 2
    virtual = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    rays = fixture["ideal_rays"]((width, height), virtual, mode)[y, x]
    expected = rays * (-0.002 * x + 2.8)[:, None]
    from_right = expected - np.array([baseline, 0, 0])
    if mode == "perspective":
        right_x = fx * from_right[:, 0] / from_right[:, 2] + cx
    else:
        right_x = fx * np.arccos(-from_right[:, 0] / np.linalg.norm(from_right, axis=1)) + cx
    actual = _triangulate(x, y, x - right_x, virtual, baseline, mode)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def _inputs(graph):
    return {
        side: packet_for(graph.blocks[side].last_result, "image") for side in ("left", "right")
    } | {"rig": graph.blocks["rig"].last_result.outputs["rig"]}


@pytest.mark.local_examples
def test_millimeter_rig_smaller_grid_and_monochrome_normalize_metric_geometry():
    graph = _run_graph()
    try:
        inputs = _inputs(graph)
        metric, mm = inputs["rig"], copy.deepcopy(inputs["rig"])
        mm["units"] = mm["transform"]["units"] = mm["left"]["units"] = mm["right"]["units"] = "mm"
        scaled = np.array(mm["transform"]["source_to_target"])
        scaled[:3, 3] *= 1000
        mm["transform"]["source_to_target"] = scaled.tolist()
        mm["transform"]["translation"] = scaled[:3, 3].tolist()
        params = {
            "width": 320,
            "height": 198,
            "num_disparities": 48,
            "sample_step": 4,
            "point_type": "xyz",
        }
        tool = ReconstructOmnidirStereo(parameters=params)
        a = tool.execute({**inputs, "rig": metric})
        b = tool.execute({**inputs, "rig": mm})
        assert a and b and a.passed and b.passed, tool.error
        assert b.outputs["report"]["input_rig_units"] == "mm"
        assert b.outputs["depth"].shape == (198, 320)
        assert b.outputs["cloud"].colors is None and b.outputs["cloud"].units == "m"
        np.testing.assert_allclose(a.outputs["cloud"].points, b.outputs["cloud"].points, atol=1e-7)
        both_gray = {}
        for side in ("left", "right"):
            source = inputs[side]
            gray = cv2.cvtColor(source.value, cv2.COLOR_BGR2GRAY)
            both_gray[side] = ValuePacket(gray, source.payload)
        gray_tool = ReconstructOmnidirStereo(parameters={**params, "point_type": "xyzrgb"})
        res = gray_tool.execute({**inputs, **both_gray})
        assert res and res.passed, gray_tool.error
        colors = res.outputs["cloud"].colors
        assert res.outputs["left"].ndim == 2
        np.testing.assert_array_equal(colors[:, 0], colors[:, 2])
    finally:
        graph.close()


@pytest.mark.local_examples
def test_sealed_calibration_and_empty_support_preserve_capture_contracts(tmp_path):
    graph = _run_graph()
    try:
        registry = builtin_registry()
        inputs = _inputs(graph)
        spec = {"path": str(tmp_path), "profile_id": "omni-fixture", "revision": "verified"}
        assert (
            registry.create("output.calibration", parameters=spec)
            .execute({"profile": inputs["rig"]})
            .passed
        )
        sealed = (
            registry.create("input.calibration", parameters=spec).execute({}).outputs["profile"]
        )
        tool = ReconstructOmnidirStereo(parameters={"num_disparities": 48})
        assert tool.execute({**inputs, "rig": sealed}) is None
        assert "matching capture metadata" in tool.error
        for side in ("left", "right"):
            declaration = registry.create(
                "calibration.declare_capture",
                parameters={
                    "profile_id": spec["profile_id"],
                    "revision": spec["revision"],
                    "camera_frame": sealed[side]["frame"],
                    "sha256": sealed[REVISION]["sha256"],
                },
            )
            inputs[side] = packet_for(declaration.execute({"image": inputs[side]}), "image")
        result = tool.execute({**inputs, "rig": sealed})
        assert result and result.passed, tool.error
        for key in ("left", "right", "disparity", "cloud"):
            assert APPLIED in result.payloads[key] and CAPTURE not in result.payloads[key]
        all_bad = np.zeros_like(inputs["right"].value[..., 0])
        invalid = tool.execute({**inputs, "rig": sealed, "right_mask": all_bad})
        assert invalid and not invalid.passed, tool.error
        assert invalid.fail_reasons[0]["code"] == "omnidir_stereo_no_supported_points"
        assert not invalid.outputs["valid_mask"].any()
        assert np.isnan(invalid.outputs["disparity"]).all()
        assert len(invalid.outputs["cloud"].points) == 0
        crop = inputs["left"]
        metadata = {**crop.payload, "spatial": frame(crop.value.shape[:2][::-1], "scaled")}
        assert (
            tool.execute({**inputs, "rig": sealed, "left": ValuePacket(crop.value, metadata)})
            is None
        )
        assert "pixel grid" in tool.error
    finally:
        graph.close()
