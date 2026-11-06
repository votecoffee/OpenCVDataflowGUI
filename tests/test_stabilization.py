"""Verify causal motion correction through sequence, frame, and ticket contracts."""

from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import NO_VALUE, metadata

TYPE = "sequence.stabilize_camera"


def _texture():
    raw = np.random.default_rng(572).integers(0, 256, (128, 160), np.uint8)
    return cv2.GaussianBlur(raw, (5, 5), 1)


def _shift(image, x, y):
    return cv2.warpAffine(
        image,
        np.array([[1, 0, x], [0, 1, y]], np.float64),
        image.shape[1::-1],
        borderMode=cv2.BORDER_REFLECT_101,
    )


def test_synthetic_camera_replay_recovers_rotation_and_translation():
    project = (
        Path(__file__).parents[1]
        / "examples/flow_control/camera_stabilization/camera_stabilization.json"
    )
    poses = [(0, 0, 0), (5, -3, 1.6), (-4, 4, -1.2), (2, 6, 0.8), (-6, -4, -1.6), (3, -1, 1)]
    points = np.array([[28, 36, 1], [110, 90, 1], [160, 110, 1]])
    graph = Graph.load(project, builtin_registry())
    try:
        graph.set_run_profile("lean")
        root = None
        for index, (dx, dy, angle) in enumerate(poses):
            graph.run()
            node = graph.blocks["stabilize"]
            result = node.last_result
            assert result is not None, node.error
            if index == 0:
                assert result.outputs["image"] is NO_VALUE
                assert result.outputs["status"]["reason"] == "initial_frame"
                continue
            assert node.state == "pass"
            stabilized = result.outputs["image"]
            covered = result.outputs["coverage"] != 0
            assert covered.mean() > 0.9
            assert np.all(stabilized[~covered] == 0)
            raw = graph.blocks["frames"].last_result.outputs["image"]
            anchor = graph.blocks["reference"].last_result.outputs["image"]
            before = np.abs(raw[covered].astype(float) - anchor[covered]).mean()
            after = graph.blocks["stabilized_error"].last_result.outputs["value"]
            assert after < before * 0.2
            spatial = metadata(stabilized)["spatial"]
            root = spatial["root_id"] if root is None else root
            assert spatial["root_id"] == root
            assert metadata(stabilized)["ticket"] == metadata(raw)["ticket"]
            assert metadata(result.outputs["coverage"])["spatial"] == spatial
            transform = result.outputs["transform"]
            assert transform["source_spatial"] == metadata(raw)["spatial"]
            assert transform["target_spatial"] == spatial
            expected = cv2.getRotationMatrix2D((103.5, 77.5), angle, 1)
            expected[:, 2] += [dx, dy]
            correction = np.asarray(transform["source_to_target"])
            np.testing.assert_allclose(
                points @ np.vstack((expected, [0, 0, 1])).T @ correction.T, points, atol=0.25
            )
    finally:
        graph.close()


def test_translation_smoothing_preserves_color_depth_and_coordinate_mapping():
    registry = builtin_registry()
    block = registry.create(TYPE, parameters={"motion_model": "translation", "follow": 0.25})
    gray = _texture().astype(np.uint16) * 257
    color = np.repeat(gray[..., None], 3, axis=2)
    first = ImageArray(color, "rgb")
    second = ImageArray(_shift(color, 8, -4), "rgb")
    assert block.execute({"image": first}).outputs["image"] is NO_VALUE
    result = block.execute({"image": second})
    assert result is not None and result.outputs["ready"], block.error
    output = result.outputs["image"]
    assert output.dtype == np.uint16 and image_format(output) == "rgb"
    correction = np.asarray(result.outputs["transform"]["source_to_target"])
    np.testing.assert_allclose(correction[:2, 2], [-6, 3], atol=0.15)
    np.testing.assert_allclose(
        correction @ np.asarray(result.outputs["transform"]["target_to_source"]),
        np.eye(3),
        atol=1e-12,
    )


def test_scene_cut_withholds_output_and_establishes_a_new_coordinate_frame():
    block = builtin_registry().create(TYPE)
    image = _texture()
    assert block.execute({"image": image}).outputs["image"] is NO_VALUE
    established = block.execute({"image": _shift(image, 2, 0)})
    old_frame = metadata(established.outputs["image"])["spatial"]
    interrupted = block.execute({"image": np.zeros_like(image)})
    assert interrupted is not None and not interrupted.outputs["ready"], block.error
    assert interrupted.outputs["image"] is NO_VALUE
    assert interrupted.outputs["status"]["reason"] == "insufficient_tracks"
    assert block.execute({"image": image}).outputs["image"] is NO_VALUE
    recovered = block.execute({"image": _shift(image, 2, 0)})
    assert recovered is not None and recovered.outputs["ready"], block.error
    assert metadata(recovered.outputs["image"])["spatial"]["root_id"] != old_frame["root_id"]


def test_empty_feature_mask_prevents_tracking_and_coverage_gate_reanchors():
    registry = builtin_registry()
    image = _texture()
    block = registry.create(TYPE)
    block.execute({"image": image, "mask": np.zeros_like(image)})
    masked = block.execute({"image": _shift(image, 4, 2)})
    assert not masked.outputs["ready"]
    assert masked.outputs["status"]["reason"] == "insufficient_features"

    block = registry.create(TYPE, parameters={"min_coverage_fraction": 1.0})
    block.execute({"image": image})
    limited = block.execute({"image": _shift(image, 4, 2)})
    assert not limited.outputs["ready"] and limited.outputs["image"] is NO_VALUE
    assert limited.outputs["status"]["reason"] == "insufficient_coverage"
