"""Verify causal motion correction through sequence, frame, and ticket contracts."""

from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import NO_VALUE, frame, metadata

TYPE = "sequence.stabilize_camera"
VALIDATION_EXAMPLE = Path(__file__).parents[1] / "examples/custom tools/stabilization_validation"


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
        / "examples/custom tools/camera_stabilization/camera_stabilization.json"
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


def test_image_quality_mask_alignment_requires_an_explicit_pixel_grid_choice():
    raw_frame = {"spatial": frame([3, 2], "raw")}
    stabilized_frame = {"spatial": frame([3, 2], "stabilized")}
    raw = ImageArray(np.full((2, 3), 30, np.uint8), "gray", raw_frame)
    anchor = np.array([[20, 0, 0], [20, 0, 0]], np.uint8)
    mask = ImageArray(np.array([[255, 0, 0], [255, 0, 0]], np.uint8), "gray", stabilized_frame)
    block = builtin_registry().create("inspect.image_quality", parameters={"metric": "mae"})
    inputs = {"image_a": raw, "image_b": anchor, "mask": mask}
    assert block.execute(inputs) is None
    assert "Selection coordinate frame differs" in block.error
    block.parameters["mask_alignment"] = "pixel_grid"
    result = block.execute(inputs)
    assert result is not None and result.outputs["value"] == 10
    assert block.execute({**inputs, "mask": np.asarray(mask)[:, :2]}) is None
    assert "matching image dimensions" in block.error


def test_recorded_camera_replay_reduces_matched_error_and_reanchors_on_coverage_loss():
    project = (
        Path(__file__).parents[1]
        / "examples/custom tools/recorded_camera_stabilization/recorded_camera_stabilization.json"
    )
    graph = Graph.load(project, builtin_registry())
    try:
        graph.set_run_profile("lean")
        frames = graph.blocks["frames"]
        assert frames.parameters["end_frame"] == 48
        frames.parameters["end_frame"] = 60
        errors = []
        old_root = None
        for index in range(60):
            graph.run()
            source = frames.last_result.outputs
            raw = source["image"]
            assert source["info"]["frame_index"] == index
            result = graph.blocks["stabilize"].last_result
            assert result is not None
            out = result.outputs
            status = out["status"]
            if index in (0, 48):
                assert not out["ready"]
                assert out["image"] is out["coverage"] is out["transform"] is NO_VALUE
                expected = "initial_frame" if index == 0 else "insufficient_coverage"
                assert status["reason"] == expected
                for name in ("raw_error", "stabilized_error"):
                    assert graph.blocks[name].state == "inactive"
                    assert not any(graph.blocks[name].current_emissions.values())
                continue
            assert out["ready"] and status["coverage_fraction"] >= 0.64
            spatial = metadata(out["image"])["spatial"]
            assert metadata(out["coverage"])["spatial"] == spatial
            assert metadata(out["image"])["ticket"] == metadata(raw)["ticket"]
            old_root = spatial["root_id"] if old_root is None else old_root
            if index > 48:
                assert status["anchor"] == 1 and spatial["root_id"] != old_root
                continue
            assert status["anchor"] == 0 and spatial["root_id"] == old_root
            covered = out["coverage"] != 0
            anchor = graph.blocks["reference"].last_result.outputs["image"]
            before = graph.blocks["raw_error"].last_result.outputs["value"]
            after = graph.blocks["stabilized_error"].last_result.outputs["value"]
            np.testing.assert_allclose(
                before, np.abs(raw[covered].astype(float) - anchor[covered]).mean()
            )
            np.testing.assert_allclose(
                after, np.abs(out["image"][covered].astype(float) - anchor[covered]).mean()
            )
            errors.append((before, after))
        assert graph.sources_exhausted()
        assert len(errors) == 47
        before, after = np.asarray(errors).T
        assert np.all(after < before)
        assert after.mean() < 0.25 * before.mean()
    finally:
        graph.close()


def test_validation_rejects_accumulated_drift_even_when_the_background_fit_passes():
    graph = Graph.load(VALIDATION_EXAMPLE / "stabilization_validation.json", builtin_registry())
    try:
        graph.set_run_profile("lean")
        checked = graph.blocks["checked"]
        baseline = graph.blocks["baseline"]
        first_root = None
        for index in range(6):
            graph.run()
            out = checked.last_result.outputs
            status = out["status"]
            if index == 0:
                assert status["reason"] == "initial_frame"
                assert status["validation"]["new_anchor_features"] >= 12
                continue
            assert baseline.last_result.outputs["ready"]
            assert baseline.last_result.outputs["status"]["inlier_ratio"] > 0.95
            if index == 3:
                assert status["reason"] == "validation_motion_rejected"
                assert status["validation"]["tracked"] >= 12
                assert not status["validation"]["accepted"]
                assert abs(status["validation"]["median_error_pixels"] - 4) < 0.05
                assert out["image"] is out["coverage"] is out["transform"] is NO_VALUE
                assert not checked.current_emissions.get("image")
                continue
            assert out["ready"] and status["validation"]["accepted"]
            root = metadata(out["image"])["spatial"]["root_id"]
            if index == 1:
                first_root = root
            if index == 2:
                assert root == first_root
                assert abs(status["validation"]["median_error_pixels"] - 2) < 0.05
            if index > 3:
                assert status["anchor"] == 1 and root != first_root
                assert status["validation"]["p90_error_pixels"] < 0.05
    finally:
        graph.close()


def test_validation_mask_excludes_its_features_from_the_fit_and_requires_tracks():
    images = [
        cv2.imread(str(VALIDATION_EXAMPLE / "assets" / f"frame-{i:02d}.png"), cv2.IMREAD_GRAYSCALE)
        for i in range(4)
    ]
    block = builtin_registry().create(
        TYPE, parameters={"motion_model": "translation", "max_validation_error_pixels": 2.5}
    )
    mask = np.zeros_like(images[0])
    mask[92:] = 255
    for image in images:
        result = block.execute({"image": image, "validation_mask": mask})
        assert result is not None, block.error
    assert result.outputs["status"]["reason"] == "validation_motion_rejected"
    assert abs(result.outputs["status"]["validation"]["median_error_pixels"] - 4) < 0.2

    empty = np.zeros_like(mask)
    result = block.execute({"image": images[0], "validation_mask": empty, "reset": True})
    assert result.outputs["status"]["validation"]["new_anchor_features"] == 0
    result = block.execute({"image": images[1], "validation_mask": empty})
    assert result.outputs["image"] is NO_VALUE
    assert result.outputs["status"]["reason"] == "insufficient_validation_tracks"
    assert result.outputs["status"]["validation"]["tracked"] == 0

    # A stream can resume without validation, starting from a fresh anchor.
    result = block.execute({"image": images[0]})
    assert result.outputs["status"]["reason"] == "initial_frame"
    result = block.execute({"image": images[1]})
    assert result.outputs["ready"]


def test_recorded_depth_separated_regions_reject_motion_and_resume(tmp_path, monkeypatch):
    folder = VALIDATION_EXAMPLE.parent / "recorded_stabilization_validation"
    monkeypatch.chdir(tmp_path)
    graph = Graph.load(folder / "recorded_stabilization_validation.json", builtin_registry())
    try:
        graph.set_run_profile("lean")
        old_root = None
        for index in range(8):
            graph.run()
            baseline = graph.blocks["baseline"].last_result.outputs
            checked = graph.blocks["checked"]
            outputs = checked.last_result.outputs
            status = outputs["status"]
            if index == 0:
                assert status["reason"] == "initial_frame"
                continue
            assert baseline["ready"]
            if index == 5:
                assert baseline["status"]["inlier_ratio"] > 0.95
                assert baseline["status"]["coverage_fraction"] > 0.97
                assert status["reason"] == "validation_motion_rejected"
                validation = status["validation"]
                assert validation["tracked"] >= 40
                assert validation["median_error_pixels"] > 3.5
                assert validation["inlier_ratio"] < 0.25
                assert outputs["image"] is outputs["coverage"] is outputs["transform"] is NO_VALUE
                assert not checked.current_emissions.get("image")
                continue
            assert outputs["ready"] and status["validation"]["accepted"]
            assert metadata(outputs["image"])["ticket"] == metadata(baseline["image"])["ticket"]
            root = metadata(outputs["image"])["spatial"]["root_id"]
            old_root = root if old_root is None else old_root
            assert (root == old_root) == (index < 5)
            assert status["anchor"] == int(index > 5)

        assert graph.sources_exhausted()
        depth = graph.blocks["depth"].last_result.outputs["depth"]
        assert depth.units == "m" and depth.shape == (240, 320)
        samples = graph.blocks["sample"].last_result.outputs["samples"]
        np.testing.assert_allclose(
            [sample["depth"] for sample in samples["samples"]], [1.1872, 0.8252]
        )
        rejecting_depth = cv2.imread(
            str(folder / "assets/depth-1305031107.130308.png"), cv2.IMREAD_UNCHANGED
        ).astype(float)
        rejecting_depth[rejecting_depth == 0] = np.nan
        for image, expected in (
            (depth, [1.0966, 0.8332]),
            (rejecting_depth / 5000, [1.2938, 0.9298]),
        ):
            areas = []
            for name in ("background", "check"):
                x, y, width, height = graph.blocks[name].parameters["rectangle"]
                areas.append(image[y : y + height, x : x + width])
            np.testing.assert_allclose([np.nanmedian(area) for area in areas], expected)
            assert np.nanpercentile(areas[0], 10) > np.nanpercentile(areas[1], 90)
    finally:
        graph.close()
