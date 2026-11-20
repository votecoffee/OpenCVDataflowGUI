"""Qualify the OpenCV RGB-D warp against a geometric destination-depth oracle."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import camera_record, pose_record
from opencv_dataflow.images import DepthArray, ImageArray
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.rgbd_warp import WarpRGBDByPose
from opencv_dataflow.tickets import TicketClock

PROJECT = Path(__file__).resolve().parents[1] / "examples/opencv tools/rgbd_pose_warp"


def _oracle(raw):
    """Independent exact arithmetic for the saved project's two integer-meter planes."""
    height, width = raw.shape
    warped = np.full(raw.shape, np.nan, np.float32)
    owners = np.full(raw.shape, -1, np.int32)
    hits = {}
    for sy, sx in np.argwhere(raw != 0):
        z = int(raw[sy, sx]) // 1000
        depth = z + 1
        tx = (28 * depth + (sx - 28) * z + 12) // depth
        ty = (20 * depth + (sy - 20) * z) // depth
        if 0 <= tx < width and 0 <= ty < height:
            hits.setdefault((ty, tx), set()).add(depth)
            if np.isnan(warped[ty, tx]) or depth < warped[ty, tx]:
                owners[ty, tx] = sy * width + sx
                warped[ty, tx] = depth
    return warped, owners, [point for point, depths in hits.items() if len(depths) > 1]


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_warp_lifts_depth_and_keeps_near_colors_at_occlusions(profile):
    graph = Graph.load(PROJECT / "rgbd_pose_warp.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        block = graph.blocks["warp"]
        result = block.last_result
        assert result and result.passed, block.error
        raw = cv2.imread(str(PROJECT / "assets/source_depth.png"), cv2.IMREAD_UNCHANGED)
        original = cv2.imread(str(PROJECT / "assets/source_color.png"), cv2.IMREAD_UNCHANGED)
        expected, ids, mixed = _oracle(raw)
        tested = tuple(np.array(mixed).T)
        assert len(mixed) == 20
        output = result.outputs
        np.testing.assert_array_equal(output["depth"], expected)
        np.testing.assert_array_equal(output["source_index"], ids)
        np.testing.assert_array_equal(output["valid_mask"], (ids >= 0).astype(np.uint8) * 255)
        np.testing.assert_array_equal(output["depth"][tested], 2)
        assert np.all(output["color"][tested][:, 2] == 235)
        expected_color = np.zeros_like(original)
        expected_color[ids >= 0] = original.reshape(-1, 3)[ids[ids >= 0]]
        np.testing.assert_array_equal(output["color"], expected_color)
        assert output["camera"]["frame"] == "moved_camera"
        assert output["report"]["visible_destination_samples"] == np.count_nonzero(ids >= 0)
        source = graph.blocks["source"].last_result.payloads["depth"]
        for name in ("depth", "color", "source_index", "valid_mask"):
            payload = result.payloads[name]
            assert payload["ticket"] == source["ticket"]
            assert payload["optical_frame"] == "moved_camera"
            assert payload["spatial"]["root_id"] != source["spatial"]["root_id"]
            assert payload["rgbd_warp"]["source_spatial"] == source["spatial"]
        source_color = graph.blocks["source"].last_result.payloads["image"]["recorded_rgbd"]
        assert result.payloads["color"]["recorded_rgbd"] == source_color
        assert result.payloads["depth"]["recorded_rgbd"] == source["recorded_rgbd"]
    finally:
        graph.close()


def _camera():
    return camera_record([[4, 0, 2], [0, 4, 2], [0, 0, 1]], [], (5, 5), frame="source")


def _pose(**options):
    transform = np.eye(4)
    transform[:3, 3] = options.get("translation", (0, 0, 0))
    if "rotation" in options:
        transform[:3, :3] = options["rotation"]
    return pose_record(
        transform, options.get("source", "source"), "target", options.get("units", "m")
    )


def test_rotation_carries_color_bytes_and_source_selection_without_converting_dtype():
    depth = DepthArray(np.full((5, 5), 1250), "mm")
    color = ImageArray(np.arange(100, dtype=np.uint16).reshape(5, 5, 4) * 607, "rgba")
    mask = np.zeros((5, 5), np.uint8)
    mask[1, 1] = mask[2, 3] = mask[3, 3] = 255
    region = {
        "model": "image_region_v1",
        "image_size": [5, 5],
        "shape": "rectangle",
        "rectangle": [0, 0, 4, 3],
    }
    box = WarpRGBDByPose()
    rotation = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    result = box.execute(
        {
            "depth": depth,
            "color": color,
            "camera": _camera(),
            "transform": _pose(rotation=rotation),
            "mask": mask,
            "region": region,
        }
    )
    assert result and result.passed, box.error
    output = result.outputs
    assert output["color"].dtype == np.uint16 and output["color"].color_format == "rgba"
    good = output["valid_mask"] != 0
    assert good.sum() == 2
    np.testing.assert_array_equal(output["color"][1, 3], color[1, 1])
    np.testing.assert_array_equal(output["color"][3, 2], color[2, 3])
    np.testing.assert_array_equal(output["depth"][good], [1.25, 1.25])
    assert np.isnan(output["depth"][~good]).all()
    assert output["source_index"][1, 3] == 6
    assert output["source_index"][3, 2] == 13


def test_no_color_passthrough_identity_and_no_visible_depth():
    depth = np.full((5, 5), 1500, np.uint16)
    depth[0, 0], depth[0, 1] = 0, 65535
    box = WarpRGBDByPose(parameters={"minimum_coverage": 1.0})
    inputs = {"depth": depth, "camera": _camera()}
    result = box.execute({**inputs, "transform": _pose()})
    assert result is not None and not result.passed
    assert result.fail_reasons[0]["code"] == "rgbd_warp_coverage"
    assert result.outputs["color"] is NO_VALUE
    assert result.outputs["report"]["visible_destination_samples"] == 23
    np.testing.assert_allclose(result.outputs["depth"].ravel()[2:], 1.5, atol=2e-7, rtol=0)
    assert np.isnan(result.outputs["depth"][0, :2]).all()
    invisible = box.execute({**inputs, "transform": _pose(translation=(0, 0, -3))})
    assert invisible is not None and not invisible.passed
    assert invisible.fail_reasons[0]["code"] == "rgbd_warp_no_visible_depth"
    assert not invisible.outputs["valid_mask"].any()


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("ticket", "same ticket"),
        ("pixels", "different pixel grids"),
        ("exposure", "different recorded pairs"),
        ("pose", "source optical frame"),
        ("distortion", "undistorted pinhole"),
        ("zero_principal", "positive focal lengths and principal point"),
    ],
)
def test_incompatible_color_depth_or_camera_contract(kind, message):
    box = WarpRGBDByPose()
    depth = np.ones((5, 5), np.float32)
    color = np.ones((5, 5, 3), np.uint8)
    a, b = [{"spatial": frame([5, 5], "same")} for _ in range(2)]
    pose, camera = _pose(), _camera()
    if kind == "ticket":
        clock = TicketClock()
        a["ticket"], b["ticket"] = clock.issue("a"), clock.issue("b")
    if kind == "pixels":
        b["spatial"] = frame([5, 5], "different")
    if kind == "exposure":
        a["recorded_rgbd"] = {"clock": "test", "pair_index": 0}
        b["recorded_rgbd"] = {"clock": "test", "pair_index": 1}
    if kind == "pose":
        pose = _pose(source="other")
    if kind == "distortion":
        camera["distortion"] = [0.1, 0, 0, 0]
    if kind == "zero_principal":
        camera["camera_matrix"][0][2] = 0
    result = box.execute(
        {
            "depth": ValuePacket(depth, a),
            "camera": camera,
            "transform": pose,
            "color": ValuePacket(color, b),
        }
    )
    assert result is None and message in box.error
