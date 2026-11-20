"""Check native multi-plane detection, metric residuals and label semantics."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import camera_record
from opencv_dataflow.images import DepthArray
from opencv_dataflow.organized_planes import FindOrganizedPlanes
from opencv_dataflow.payloads import ValuePacket, frame

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/organized_plane_detection"


def _scene():
    h, w, focal = 100, 168, 145
    y, x = np.mgrid[:h, :w]
    cx, cy = (w - 1) / 2, (h - 1) / 2
    model = camera_record([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], [], (w, h), frame="optic")
    u, v = (x - cx) / focal, (y - cy) / focal
    left = x < w // 2
    depth = np.where(left, 0.9, 1.12 / (1 - 0.29 * u + 0.12 * v))
    radius = ((x - 83) / 12) ** 2 + ((y - 42) / 9) ** 2
    foreground = radius < 1
    depth[foreground] = 0.64 + 0.07 * radius[foreground]
    invalid = np.zeros((h, w), bool)
    invalid[38:47, 24:31] = invalid[66:75, 138:145] = True
    depth[invalid] = np.nan
    return depth, model, left, foreground, invalid, np.stack((u * depth, v * depth, depth), axis=-1)


def _tool(**parameters):
    settings = {
        "block_size": 16,
        "minimum_plane_pixels": 900,
        "threshold_m": 0.005,
        "minimum_planes": 2,
        **parameters,
    }
    return FindOrganizedPlanes(parameters=settings)


@pytest.mark.parametrize("method", ["none", "fals", "cross_product", "sri"])
def test_two_planes_have_metric_equations_and_unmatched_foreground(method):
    depth, camera, left, foreground, invalid, xyz = _scene()
    typed = DepthArray(depth * 1000, "mm", dtype=np.float64)
    result = _tool(normal_method=method).execute({"depth": typed, "camera": camera})
    assert result.passed
    outputs = result.outputs
    labels = outputs["labels"]
    assert labels.dtype == np.uint16 and set(np.unique(labels)) == {0, 1, 2, 65535}
    assert outputs["report"]["native_planes"] == outputs["report"]["plane_count"] == 2
    assert outputs["planes"]["frame"] == "optic" and outputs["planes"]["units"] == "m"
    assert (labels[foreground] == 0).all() and (labels[invalid] == 65535).all()
    for plane in outputs["planes"]["items"]:
        a, b, c, d = plane["coefficients"]
        region = labels == plane["label"]
        assert region.sum() == plane["support"] > 7_500
        assert c < 0 and np.linalg.norm([a, b, c]) == pytest.approx(1.0)
        expected = np.array([0, 0, -1, 0.9] if np.all(left[region]) else [0.29, -0.12, -1, 1.12])
        expected /= np.linalg.norm(expected[:3])
        np.testing.assert_allclose([a, b, c, d], expected, atol=1e-6)
        assert np.max(np.abs(xyz[region] @ [a, b, c] + d)) < 1e-6
    np.testing.assert_array_equal(outputs["valid_mask"] != 0, ~invalid)
    np.testing.assert_array_equal(outputs["unmatched_mask"] != 0, labels == 0)
    np.testing.assert_allclose(outputs["unmatched_depth"][foreground], depth[foreground], atol=2e-7)
    assert np.isnan(outputs["unmatched_depth"][labels != 0]).all()


def test_mask_and_uint16_sentinel_preserve_grid_and_ticket_context():
    depth, camera, _, foreground, invalid, _ = _scene()
    raw = np.rint(np.nan_to_num(depth) * 1000).astype(np.uint16)
    raw[invalid] = 65535
    before = raw.copy()
    mask = np.full(raw.shape, 255, np.uint8)
    mask[:3] = 0
    spatial = frame(list(raw.shape[::-1]), "depth-capture")
    payload = {"spatial": spatial, "optical_frame": "optic"}
    result = _tool().execute({"depth": ValuePacket(raw, payload), "camera": camera, "mask": mask})
    assert result.passed
    assert result.payloads["labels"]["spatial"] == spatial
    assert result.payloads["unmatched_depth"]["spatial"] == spatial
    np.testing.assert_array_equal(raw, before)
    report = result.outputs["report"]
    assert report["invalid_depth_pixels"] == 126 and report["excluded_valid_pixels"] == 504
    labels = result.outputs["labels"]
    assert np.all(labels[invalid] == 65535) and np.all(labels[:3] == 65535)
    assert np.all(labels[foreground] == 0)


def test_no_planes_keeps_usable_depth_and_uses_normal_validation():
    depth = DepthArray([[1, np.nan], [0, 1]])
    camera = camera_record(np.eye(3), [], (2, 2))
    result = _tool().execute({"depth": depth, "camera": camera})
    assert not result.passed
    assert result.fail_reasons[0]["code"] == "organized_planes_insufficient_support"
    np.testing.assert_array_equal(result.outputs["labels"], [[0, 65535], [65535, 0]])
    np.testing.assert_array_equal(result.outputs["unmatched_depth"], [[1, np.nan], [np.nan, 1]])
    tight = _tool(minimum_plane_pixels=3, normal_window="7")
    assert tight.execute({"depth": DepthArray(np.ones((2, 2))), "camera": camera}) is None
    assert "Normal window" in tight.error


def test_native_labels_are_verified_against_the_final_equation(monkeypatch):
    depth = DepthArray(np.ones((6, 7)))
    depth[2, 3] = 1.06
    camera = camera_record(np.eye(3), [], (7, 6))
    raw = np.zeros((6, 7), np.uint8)
    raw[-1] = 254

    monkeypatch.setattr(
        "opencv_dataflow.organized_planes.cv2.findPlanes",
        lambda *args, **kwargs: (raw, np.array([[[0, 0, -1, 1]]])),
    )
    result = _tool(normal_method="none", minimum_plane_pixels=6, minimum_planes=1).execute(
        {"depth": depth, "camera": camera}
    )
    assert result.passed
    assert result.outputs["report"]["native_labels_without_coefficients"] == 7
    labels = result.outputs["labels"]
    assert (labels[-1] == 0).all() and labels[2, 3] == 0
    assert np.count_nonzero(labels == 1) == 34
    assert result.outputs["planes"]["items"][0]["max_residual_m"] < 1e-6


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_scene_delivers_two_planes_and_raw_labels(profile):
    graph = Graph.load(EXAMPLE / "organized_plane_detection.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        result = graph.blocks["planes"].last_result
        assert result.passed
        labels = result.outputs["labels"]
        assert labels[42, 83] == 0 and labels[40, 26] == 65535
        assert 0 < labels[20, 30] != labels[20, 138] > 0
        assert result.outputs["report"]["plane_count"] == 2
        assert result.outputs["report"]["invalid_depth_pixels"] == 126
    finally:
        graph.close()
