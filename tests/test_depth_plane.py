"""Depth plane recovery, orthogonal residuals, and foreground pixel provenance."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import camera_record
from opencv_dataflow.images import DepthArray
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock

TYPE_ID = "3d.separate_depth_plane"
EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/depth_plane"
K = np.array([[80, 0, 17.4], [0, 73, 11.6], [0, 0, 1]])


def test_saved_graph_matches_independent_tilted_plane_truth():
    graph = Graph.load(EXAMPLE / "depth_plane.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["separate"]
        result = block.last_result
        assert result and result.passed, block.error
        output = result.outputs
        labels = cv2.imread(str(EXAMPLE / "assets/truth_labels.png"), cv2.IMREAD_GRAYSCALE)
        np.testing.assert_array_equal(output["valid_mask"] > 0, labels > 0)
        np.testing.assert_array_equal(output["plane_mask"] > 0, labels == 1)
        np.testing.assert_array_equal(output["foreground_mask"] > 0, labels == 2)
        signed = output["signed_distance_m"]
        for label, expected in [(1, 0.0), (2, 0.065), (3, -0.07), (4, 0.008)]:
            np.testing.assert_allclose(signed[labels == label], expected, atol=0.002)
        assert np.isnan(signed[labels == 0]).all()
        obj_depth = output["foreground_depth"]
        assert obj_depth.units == "mm"
        np.testing.assert_array_equal(np.isfinite(obj_depth), labels == 2)
        plane = output["plane"]
        normal = np.array([0.22, -0.16, -1.0])
        np.testing.assert_allclose(plane["normal"], normal / np.linalg.norm(normal), atol=0.0002)
        assert plane["offset"] == pytest.approx(1.44, abs=0.00015)
        assert plane["units"] == "m" and plane["frame"] == "depth_camera"
        report = output["report"]
        assert report["fit_samples"] == 4096 < report["fitting_pixels"]
        assert report["fit_inliers"] == np.count_nonzero(labels == 1)
        assert report["fit_rms_m"] < 0.001
        assert len(result.drawings["Foreground"].objects) == 2
        cloud = graph.blocks["objects"].last_result.outputs["cloud"]
        x, y = cloud.pixels.T
        assert len(x) == np.count_nonzero(labels == 2)
        assert (labels[y, x] == 2).all()
        assert cloud.frame == "depth_camera" and cloud.units == "m"

        block.parameters["minimum_inlier_fraction"] = 1.0
        graph.run()
        assert block.last_result.fail_reasons[0]["code"] == "insufficient_plane_support"
        assert graph.blocks["objects"].last_result is None
    finally:
        graph.close()


def test_fit_selection_preserves_full_frame_and_wire_context():
    size = (36, 24)
    values = np.full(size[::-1], 800, np.float32)
    values[:, 22:] = 1200
    values[:2, 16:20] = np.nan
    model = camera_record(K, [], size, frame="station")
    coord = frame(size, "camera-grid")
    metadata = {"spatial": coord, "ticket": TicketClock().issue("camera")}
    mask = np.zeros_like(values, dtype=np.uint8)
    mask[:, 20:] = 255
    region = {
        "model": "image_region_v1",
        "image_size": list(size),
        "shape": "rectangle",
        "rectangle": [23, 4, 10, 12],
    }
    block = builtin_registry().create(TYPE_ID, parameters={"maximum_fit_samples": 80})
    result = block.execute(
        {
            "depth": ValuePacket(DepthArray(values, "mm"), metadata),
            "camera": model,
            "mask": ValuePacket(mask, {"spatial": coord}),
            "region": region,
        }
    )
    assert result and result.passed, block.error
    report = result.outputs["report"]
    assert report["fit_inliers"] == report["fitting_pixels"] == 120
    assert report["fit_samples"] == 80
    assert result.outputs["plane"]["offset"] == pytest.approx(1.2)
    assert result.outputs["foreground_mask"][8, 5] == 255
    assert result.outputs["plane_mask"][8, -1] == 255
    assert result.outputs["signed_distance_m"][8, 5] == pytest.approx(0.4)
    assert result.outputs["foreground_mask"][0, 16] == 0
    for output in (
        "signed_distance_m",
        "foreground_mask",
        "foreground_depth",
        "plane_mask",
        "valid_mask",
    ):
        assert result.payloads[output]["spatial"] == coord
        assert result.payloads[output]["ticket"] == metadata["ticket"]


@pytest.mark.parametrize("mode", ("empty", "collinear", "below_consensus"))
def test_failed_fit_withholds_plane_and_classification(mode):
    size = (36, 24)
    parameters = {"minimum_inliers": 12, "minimum_inlier_fraction": 0.9}
    depth = np.zeros(size[::-1], np.float32)
    if mode == "collinear":
        depth[12, :] = 1
        reason = "degenerate_plane"
    elif mode == "below_consensus":
        depth[:, :18], depth[:, 18:] = 1, 1.5
        reason = "insufficient_plane_support"
    else:
        reason = "insufficient_fit_pixels"
    block = builtin_registry().create(TYPE_ID, parameters=parameters)
    result = block.execute({"depth": depth, "camera": camera_record(K, [], size)})
    assert result is not None and not result.passed
    assert result.fail_reasons[0]["code"] == reason
    for name in ("plane", "plane_mask", "foreground_mask", "foreground_depth", "signed_distance_m"):
        assert result.outputs[name] is NO_VALUE
    np.testing.assert_array_equal(result.outputs["valid_mask"] > 0, depth > 0)


@pytest.mark.parametrize("wrong_size", (False, True))
def test_invalid_intrinsics_reject(wrong_size):
    depth = np.ones((24, 36), np.float32)
    if wrong_size:
        model = camera_record(K, [], [36, 25])
        message = "Camera image size differs"
    else:
        model = camera_record(K, [0.03, 0, 0, 0], [36, 24])
        message = "undistorted pinhole"
    block = builtin_registry().create(TYPE_ID)
    assert block.execute({"depth": depth, "camera": model}) is None
    assert message in block.error


def test_foreground_and_plane_cutoffs_cannot_overlap():
    block = builtin_registry().create(TYPE_ID)
    with pytest.raises(ValueError, match="at least the plane tolerance"):
        block.validate_parameters({**block.parameters, "foreground_min_m": 0.003})
