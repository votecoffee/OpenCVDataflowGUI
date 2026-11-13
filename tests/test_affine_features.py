"""Affine feature contracts and repeatability against independent image-generation truth."""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.affine_features import AffineFeatures, affine_views, source_keypoint
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/affine_features"


def image():
    return cv2.imread(str(EXAMPLE / "assets/reference.png"), cv2.IMREAD_GRAYSCALE)


def execute(inputs, **parameters):
    block = builtin_registry().create("inspect.affine_features", parameters=parameters)
    result = block.execute(inputs)
    assert result is not None, block.error
    return block, result


@pytest.mark.parametrize(
    "backend,wta,depth",
    [
        ("sift", "2", "float32"),
        ("sift", "2", "uint8"),
        ("orb", "2", "float32"),
        ("orb", "3", "float32"),
        ("orb", "4", "float32"),
    ],
)
def test_descriptors_reuse_existing_matchers(backend, wta, depth):
    _, result = execute(
        {"image": image()},
        backend=backend,
        max_tilt=1,
        max_features=150,
        orb_wta_k=wta,
        sift_descriptor_depth=depth,
    )
    features = result.outputs["features"]
    count = result.outputs["count"]
    assert 50 <= count <= 150
    assert features["algorithm"] == backend.upper()
    assert features["extractor"] == "AffineFeature"
    assert np.asarray(features["descriptors"]).shape == (count, 128 if backend == "sift" else 32)
    matcher = builtin_registry().create(
        f"inspect.match_{backend}", parameters={"mode": "cross_check"}
    )
    matched = matcher.execute({"query": features, "train": features})
    assert matched is not None, matcher.error
    assert matched.outputs["count"] > 40
    assert all(m["distance"] == 0 for m in matched.outputs["matches"]["matches"])


def test_source_scale_orientation_ellipse_and_view_provenance():
    key = cv2.KeyPoint(23, 31, 10, 45, 0.2, 3, 1)
    record = source_keypoint(key, [(1, 0), (2, 30)])
    angle = math.radians(45)
    roll = math.radians(30)
    rotation = np.array([[math.cos(roll), -math.sin(roll)], [math.sin(roll), math.cos(roll)]])
    source_direction = rotation.T @ np.diag([2, 1]) @ [math.cos(angle), math.sin(angle)]
    assert record["point"] == [23, 31]
    assert record["size"] == pytest.approx(10 * math.sqrt(2))
    assert record["angle"] == pytest.approx(
        math.degrees(math.atan2(source_direction[1], source_direction[0])) % 360
    )
    assert record["orientation_radius"] == pytest.approx(np.linalg.norm(source_direction) * 5)
    assert record["support_ellipse"] == {
        "major_diameter": 20,
        "minor_diameter": 10,
        "angle_degrees": 150,
    }
    assert record["view_size"] == 10 and record["view_angle"] == 45
    assert record["view_index"] == 1 and record["octave"] == 3
    unknown = source_keypoint(cv2.KeyPoint(0, 0, 4, -1, 0, 0, 0), [(1, 0)])
    assert unknown["angle"] == -1 and unknown["orientation_radius"] is None


def test_selection_sorting_and_truncation_keep_descriptor_rows_aligned():
    source = image()
    mask = np.zeros_like(source)
    mask[20:200, 30:190] = 255
    region = {
        "model": "image_region_v1",
        "image_size": [320, 256],
        "shape": "rectangle",
        "rectangle": [20, 40, 230, 170],
    }
    _, result = execute(
        {"image": source, "mask": mask, "region": region},
        view_mode="custom",
        custom_views=[[1, 0]],
        features_per_view=180,
        max_features=37,
    )
    expected_mask = mask.copy()
    expected_mask[:40] = 0
    expected_mask[210:] = 0
    keys, descriptors = cv2.SIFT_create(180).detectAndCompute(source, expected_mask)
    keep = sorted(range(len(keys)), key=lambda i: -keys[i].response)[:37]
    assert result.outputs["count"] == 37
    np.testing.assert_allclose(result.outputs["points"], [keys[i].pt for i in keep])
    np.testing.assert_array_equal(result.outputs["features"]["descriptors"], descriptors[keep])
    for x, y in np.rint(result.outputs["points"]).astype(int):
        assert expected_mask[y, x]


def test_ticket_frame_pyramid_and_original_color_depth_survive():
    gray = image().astype(np.uint16) * 257
    rgb = ImageArray(np.dstack([gray, gray, gray]), "rgb")
    saved = rgb.copy()
    ticket = TicketClock().issue("part-128")
    spatial = frame([320, 256], "sensor")
    spatial.update(root_size=[640, 512], to_root=[[2, 0, 0], [0, 2, 0], [0, 0, 1]])
    pyramid = {
        "level": 1,
        "largest_size": [640, 512],
        "to_largest": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
    }
    context = {"ticket": ticket, "spatial": spatial, "pyramid": pyramid, "encoder": 912}
    block, result = execute({"image": ValuePacket(rgb, context)}, max_tilt=0, max_features=70)
    np.testing.assert_array_equal(result.outputs["image"], saved)
    np.testing.assert_array_equal(rgb, saved)
    assert result.outputs["image"].color_format == "rgb"
    for port in ("image", "features", "points", "sampling", "overlay"):
        assert result.payloads[port]["ticket"] == ticket
        assert result.payloads[port]["spatial"] == spatial
        assert result.payloads[port]["pyramid"] == pyramid
        assert result.payloads[port]["encoder"] == 912
    assert result.drawings["Affine features"].objects
    assert result.outputs["overlay"].dtype == np.uint8
    block.set_graphics_mode("off")
    lean = block.execute({"image": ValuePacket(rgb, context)})
    assert lean is not None, block.error
    assert lean.outputs["overlay"] is NO_VALUE
    assert not lean.drawings and not lean.graphics
    assert lean.outputs["features"] == result.outputs["features"]


@pytest.mark.parametrize("backend", ["sift", "orb"])
def test_blank_and_empty_selection_are_valid(backend):
    for source, mask in (
        (np.zeros((64, 80), np.uint8), None),
        (image(), np.zeros((256, 320), np.uint8)),
    ):
        inputs = {"image": source, **({"mask": mask} if mask is not None else {})}
        _, result = execute(inputs, backend=backend, max_tilt=1)
        assert result.outputs["count"] == 0
        assert result.outputs["features"]["descriptors"] == []


@pytest.mark.parametrize(
    "parameters,message",
    [
        ({"max_tilt": 1, "min_tilt": 2}, "Minimum tilt"),
        ({"max_tilt": 6, "tilt_step": 2}, "exceeds eight"),
        ({"rotate_step_base": 1}, "64 views"),
        ({"view_mode": "custom", "custom_views": []}, "1..64"),
        ({"view_mode": "custom", "custom_views": [[0.5, 0]]}, "1..8"),
        ({"view_mode": "custom", "custom_views": [[1, 181]]}, "-180..180"),
        ({"view_mode": "custom", "custom_views": [[1, float("nan")]]}, "finite"),
        ({"view_mode": "custom", "custom_views": [[True, 0]]}, "numbers"),
        ({"view_mode": "custom", "custom_views": [[1, 0], [1, 0]]}, "duplicate"),
        ({"backend": "orb", "orb_first_level": 8}, "first level"),
        ({"orb_patch_size": 30}, "odd"),
        ({"features_per_view": 0}, "between"),
        ({"sift_sigma": -1}, "between"),
        ({"backend": "invalid"}, "one of"),
        (
            {
                "view_mode": "custom",
                "custom_views": [[1, i] for i in range(40)],
                "features_per_view": 2000,
            },
            "64,000",
        ),
    ],
)
def test_invalid_configuration_fails_safely(parameters, message):
    block = builtin_registry().create("inspect.affine_features", parameters=parameters)
    assert block.execute({"image": np.zeros((32, 32), np.uint8)}) is None
    assert block.state == "error" and message in block.error


def test_budget_is_checked_before_native_detection(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Native detector called before checking sampling budget")

    monkeypatch.setattr(cv2, "AffineFeature_create", forbidden)
    block = AffineFeatures(parameters={"max_tilt": 3})
    assert block.execute({"image": np.zeros((1600, 1600), np.uint8)}) is None
    assert "canvas-work limit" in block.error


def test_native_exception_is_contained(monkeypatch):
    def broken(*args, **kwargs):
        raise cv2.error("native detector failed")

    monkeypatch.setattr(cv2, "AffineFeature_create", broken)
    block = AffineFeatures(parameters={"max_tilt": 0})
    assert block.execute({"image": image()}) is None
    assert block.state == "error" and "native detector failed" in block.error
    assert not block.current_emissions


def test_generated_minimum_is_an_absolute_tilt_exponent():
    p = {**AffineFeatures.defaults, "min_tilt": 2, "max_tilt": 2, "tilt_step": 2}
    views = affine_views(p)
    assert all(tilt == 4 for tilt, _ in views)
    assert views[0] == (4, 0)
    assert all(roll < 180 for _, roll in views)


def test_selection_frame_mismatch_is_rejected():
    source = ValuePacket(image(), {"spatial": frame([320, 256], "camera-a")})
    mask = ValuePacket(np.ones((256, 320), np.uint8), {"spatial": frame([320, 256], "camera-b")})
    block = AffineFeatures(parameters={"max_tilt": 0})
    assert block.execute({"image": source, "mask": mask}) is None
    assert "frame differs" in block.error


@pytest.mark.parametrize("graphics", ["on", "off"])
def test_saved_example_repeatability_against_known_tilt(graphics, tmp_path):
    graph = Graph.load(EXAMPLE / "project.json", builtin_registry())
    try:
        for block in graph.blocks.values():
            block.set_graphics_mode(graphics)
        graph.run()
        assert all(b.state == "pass" for b in graph.processing_blocks())
        truth = np.asarray(
            json.loads((EXAMPLE / "assets/truth.json").read_text())["source_to_target"]
        )
        pairs = graph.blocks["match"].last_result.outputs["matches"]
        query = np.asarray(pairs["query_points"], np.float64)
        train = np.asarray(pairs["train_points"], np.float64)
        expected = cv2.perspectiveTransform(query[:, None], truth).reshape(-1, 2)
        errors = np.linalg.norm(train - expected, axis=1)
        assert len(errors) >= 100
        assert np.mean(errors <= 3) > 0.9
        assert np.median(errors) < 1.5
        # Independent spatial support, not a count of repeated affine-view keypoints.
        assert len(np.unique(np.rint(query[errors <= 3] / 5), axis=0)) >= 50
        fit = graph.blocks["fit"].last_result.outputs["transform"]
        grid = np.array([[[30, 30], [290, 30], [290, 220], [30, 220]]], np.float64)
        reference = cv2.perspectiveTransform(grid, truth)
        predicted = cv2.perspectiveTransform(grid, np.asarray(fit["source_to_target"]))
        assert np.max(np.linalg.norm(predicted - reference, axis=2)) < 2
        # Save/reload field values without rebasing the project-relative asset paths.
        saved = tmp_path / "roundtrip.json"
        graph.save(saved)
        loaded = Graph.load(saved, builtin_registry())
        try:
            for name in ("reference_features", "tilted_features"):
                assert loaded.blocks[name].parameters == graph.blocks[name].parameters
                assert loaded.blocks[name].position == graph.blocks[name].position
        finally:
            loaded.close()
    finally:
        graph.close()
