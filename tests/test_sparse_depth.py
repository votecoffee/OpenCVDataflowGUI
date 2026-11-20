"""Qualify sparse camera XYZ, per-request diagnostics and project replay."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import camera_record
from opencv_dataflow.images import DepthArray
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.sparse_depth import BackprojectDepthPixels
from opencv_dataflow.tickets import TicketClock

EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples/opencv tools/sparse_depth_backprojection/sparse_depth_backprojection.json"
)
K = np.array([[5.0, 0, 2], [0, 4.0, 1], [0, 0, 1]])


def _model(**extra):
    return camera_record(K, extra.get("distortion", []), (6, 4), frame="depth-camera")


def test_uint16_uses_native_with_only_eligible_pixels_and_preserves_request_order(monkeypatch):
    depth = np.full((4, 6), 2000, np.uint16)
    depth[1, 1], depth[1, 2] = 0, 65535
    mask = np.full(depth.shape, 255, np.uint8)
    mask[0, 0] = 0
    inputs = [[5, 3], [1, 1], [2, 1], [3.6, 2.1], [6, 2], [5, 3], [0, 0]]
    clock = TicketClock()
    payload = {
        "spatial": frame([6, 4], "depth-pixels"),
        "ticket": clock.issue("frame-1"),
        "optical_frame": "depth-camera",
    }
    native = cv2.depthTo3dSparse

    def counted(source, matrix, points):
        np.testing.assert_array_equal(source, depth)
        np.testing.assert_array_equal(matrix, K)
        np.testing.assert_array_equal(points.reshape(-1, 2), [[5, 3], [4, 2], [5, 3]])
        return native(source, matrix, points)

    monkeypatch.setattr(cv2, "depthTo3dSparse", counted)
    block = BackprojectDepthPixels(
        parameters={"units_per_sample": 0.002, "output_units": "mm", "minimum_valid_fraction": 0.5}
    )
    result = block.execute(
        {
            "depth": ValuePacket(depth, payload),
            "camera": _model(),
            "points": ValuePacket(inputs, payload),
            "mask": ValuePacket(mask, payload),
        }
    )
    assert result is not None and not result.passed, block.error
    assert result.fail_reasons[0]["code"] == "insufficient_valid_depth_points"
    cloud = result.outputs["cloud"]
    assert (result.outputs["count"], cloud.units, cloud.frame, cloud.source_size) == (
        3,
        "mm",
        "depth-camera",
        (6, 4),
    )
    np.testing.assert_allclose(
        cloud.points,
        [[2400, 2000, 4000], [1600, 1000, 4000], [2400, 2000, 4000]],
        atol=0.001,
    )
    np.testing.assert_array_equal(cloud.pixels, [[5, 3], [4, 2], [5, 3]])
    rows = result.outputs["report"]["samples"]
    assert [row["status"] for row in rows] == [
        "ok",
        "invalid_depth",
        "invalid_depth",
        "ok",
        "outside_image",
        "ok",
        "outside_selection",
    ]
    assert [row.get("pixel_index") for row in rows] == [23, 7, 8, 16, None, 23, 0]
    assert [row.get("cloud_index") for row in rows] == [0, None, None, 1, None, 2, None]
    assert result.payloads["cloud"]["spatial"] == payload["spatial"]
    assert cloud.metadata["ticket"] == payload["ticket"]
    assert result.drawings["Depth samples"].source == "depth"


def test_float64_and_tagged_millimeters_and_empty_requests(monkeypatch):
    depth = np.full((4, 6), 1200, np.float64)
    depth[0, 0] = np.inf
    block = BackprojectDepthPixels(
        parameters={"pixel_points": [[2, 1], [0, 0], [3.2, 1]], "pixel_sampling": "integer"}
    )
    with monkeypatch.context() as spy:

        def denied(*args):
            pytest.fail("float64 should not require a full-image conversion to OpenCV")

        spy.setattr(cv2, "depthTo3dSparse", denied)
        result = block.execute({"depth": depth, "camera": _model()})
    assert result and result.passed, block.error
    np.testing.assert_allclose(result.outputs["cloud"].points, [[0, 0, 1200]])
    assert [row["status"] for row in result.outputs["report"]["samples"]] == [
        "ok",
        "invalid_depth",
        "fractional_pixel",
    ]
    tagged = DepthArray(depth, "mm")
    result = block.execute({"depth": tagged, "camera": _model()})
    assert result and result.passed, block.error
    np.testing.assert_allclose(result.outputs["cloud"].points, [[0, 0, 1.2]])
    empty = block.execute({"depth": tagged, "camera": _model(), "points": []})
    assert empty is not None and not empty.passed
    assert empty.outputs["cloud"].points.shape == (0, 3)
    assert empty.fail_reasons[0]["code"] == "no_valid_depth_points"


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("grid", "different coordinate frames"),
        ("ticket", "same ticket"),
        ("camera", "optical frame"),
        ("distortion", "undistorted pinhole"),
        ("size", "size differs"),
        ("points", "finite numeric [x, y] pairs"),
    ],
)
def test_invalid_camera_or_point_contract_is_a_validation_error(mode, message):
    clock = TicketClock()
    payload = {"spatial": frame([6, 4], "depth"), "ticket": clock.issue("depth")}
    point_payload = dict(payload)
    points, camera = [[0, 0]], _model()
    if mode == "grid":
        point_payload["spatial"] = frame([6, 4], "other")
    if mode == "ticket":
        point_payload["ticket"] = clock.issue("point")
    if mode == "camera":
        payload["optical_frame"] = "other"
    if mode == "distortion":
        camera = _model(distortion=[0.1, 0, 0, 0, 0])
    if mode == "size":
        camera = camera_record(K, [], (5, 4))
    if mode == "points":
        points = [["0", 0]]
    block = BackprojectDepthPixels()
    result = block.execute(
        {
            "depth": ValuePacket(np.ones((4, 6), np.float32), payload),
            "camera": camera,
            "points": ValuePacket(points, point_payload),
        }
    )
    assert result is None and message in block.error


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_depth_step_and_independent_pinhole_equations(profile):
    graph = Graph.load(EXAMPLE, builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        block = graph.blocks["sparse"]
        result = block.last_result
        assert result is not None and result.passed, block.error
        cloud = result.outputs["cloud"]
        pixels = np.array([[25, 14], [55, 14], [25, 41], [0, 55], [25, 14]])
        z = np.array([1, 1.6, 1, 1, 1])
        expected = np.column_stack(((pixels[:, 0] - 40) / 100 * z, (pixels[:, 1] - 28) / 80 * z, z))
        np.testing.assert_array_equal(cloud.pixels, pixels)
        np.testing.assert_allclose(cloud.points, expected, atol=2e-7)
        assert result.outputs["report"]["status_counts"] == {
            "ok": 5,
            "invalid_depth": 1,
            "outside_image": 1,
        }
        assert bool(result.drawings) == (profile == "debug")
    finally:
        graph.close()
