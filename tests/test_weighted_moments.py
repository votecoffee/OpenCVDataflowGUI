"""Independent mass/covariance truth, finite scaling and reusable pose contracts."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.opencv_reference import opencv_reference
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.poses import validate_pose
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.weighted_moments import IntensityMoments

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/weighted_moments"


def run(image, extra=None, **parameters):
    block = builtin_registry().create("inspect.intensity_moments", parameters=parameters)
    result = block.execute({"image": image, **(extra or {})})
    assert result is not None, block.error
    return block, result


def reference(weights):
    y, x = np.mgrid[: weights.shape[0], : weights.shape[1]]
    mass = weights.sum()
    center = np.array([(weights * x).sum(), (weights * y).sum()]) / mass
    dx, dy = x - center[0], y - center[1]
    covariance = (
        np.array(
            [
                [(weights * dx * dx).sum(), (weights * dx * dy).sum()],
                [(weights * dx * dy).sum(), (weights * dy * dy).sum()],
            ]
        )
        / mass
    )
    return mass, center, covariance


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
@pytest.mark.parametrize("weighting", ["intensity", "uniform_foreground", "uniform_selection"])
def test_center_covariance_and_global_moments_match_independent_pixel_sums(dtype, weighting):
    image = np.zeros((20, 28), dtype)
    image[7:12, 11:20] = np.arange(1, 46).reshape(5, 9)
    original = image.copy()
    block, result = run(image, weighting=weighting, emit_weights=True)
    expected = (
        image.astype(float)
        if weighting == "intensity"
        else (image > 0).astype(float)
        if weighting == "uniform_foreground"
        else np.ones(image.shape)
    )
    mass, center, covariance = reference(expected)
    np.testing.assert_allclose(result.outputs["center"], center, atol=1e-12)
    np.testing.assert_allclose(
        result.outputs["measurements"]["weighted"]["covariance"], covariance, atol=1e-11
    )
    assert result.outputs["mass"] == pytest.approx(mass)
    np.testing.assert_array_equal(result.outputs["image"], original)
    np.testing.assert_array_equal(image, original)
    scale = result.payloads["moments"]["moments_basis"]["weight_scale"]
    yy, xx = np.mgrid[:20, :28]
    for key in ("m00", "m10", "m01", "m20", "m11", "m02", "m30", "m21", "m12", "m03"):
        i, j = int(key[1]), int(key[2])
        assert result.outputs["moments"][key] == pytest.approx(
            float(np.sum(expected / scale * xx**i * yy**j)), abs=1e-9
        )
    np.testing.assert_allclose(result.outputs["weights"] * scale, expected)
    assert "moments" in opencv_reference(block)


@pytest.mark.parametrize("factor", [1e-300, 1.0, 1e300])
def test_peak_normalization_retains_tiny_and_hdr_centers(factor):
    image = np.zeros((8, 10), np.float64)
    image[3:5, 5:7] = np.array([[0.25, 1], [0.5, 0.75]]) * factor
    block = IntensityMoments(parameters={"emit_weights": True})
    block.set_graphics_mode("off")
    result = block.execute({"image": image})
    assert result is not None, block.error
    _, center, covariance = reference(image / factor)
    np.testing.assert_allclose(result.outputs["center"], center, atol=1e-12)
    np.testing.assert_allclose(
        result.outputs["measurements"]["weighted"]["covariance"], covariance, atol=1e-12
    )
    assert result.outputs["mass"] / factor == pytest.approx(2.5)


def test_small_feature_near_far_image_edge_uses_stable_central_moments():
    source = np.zeros((3, 16384), np.float64)
    source[1:3, 16000:16002] = [[1, 3], [1, 3]]
    _, result = run(source)
    np.testing.assert_allclose(result.outputs["center"], [16000.75, 1.5], atol=1e-12)
    np.testing.assert_allclose(
        result.outputs["measurements"]["weighted"]["covariance"],
        [[0.1875, 0], [0, 0.25]],
        atol=1e-14,
    )
    assert result.outputs["angle_degrees"] == 90


@pytest.mark.parametrize("polarity", ["bright", "dark", "absolute"])
def test_background_threshold_and_polarity_have_explicit_weight_meaning(polarity):
    image = np.array([[2, 7, 8, 12], [3, 7, 9, 10]], np.float64)
    difference = image - 7
    weights = (
        np.maximum(difference, 0)
        if polarity == "bright"
        else np.maximum(-difference, 0)
        if polarity == "dark"
        else np.abs(difference)
    )
    weights[weights <= 1] = 0
    _, result = run(image, polarity=polarity, background=7, minimum_signal=1)
    mass, center, _ = reference(weights)
    np.testing.assert_allclose(result.outputs["center"], center)
    assert result.outputs["mass"] == pytest.approx(mass)
    assert np.array_equal(result.outputs["support"] != 0, weights > 0)


def test_color_channels_normalized_units_selection_and_payloads():
    source = np.zeros((24, 32, 4), np.uint16)
    source[5:18, 8:24, 0] = 12000
    source[10:18, 15:24, 2] = 50000
    source[..., 3] = 65535
    rgb = ImageArray(source, "rgba")
    ticket = TicketClock().issue("part-131")
    spatial = frame([32, 24], "camera")
    spatial.update(root_size=[64, 48], to_root=[[2, 0, 0], [0, 2, 0], [0, 0, 1]])
    pyramid = {
        "level": 1,
        "largest_size": [64, 48],
        "to_largest": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
    }
    context = {"ticket": ticket, "spatial": spatial, "pyramid": pyramid, "encoder": 800}
    mask = np.zeros((24, 32), np.uint8)
    mask[4:20, 10:26] = 255
    region = {
        "model": "image_region_v1",
        "image_size": [32, 24],
        "shape": "polygon",
        "vertices": [[8, 4], [26, 4], [22, 20], [8, 20]],
    }
    polygon = np.zeros_like(mask)
    cv2.fillPoly(polygon, [np.asarray(region["vertices"], np.int32)], 255)
    selected = (mask != 0) & (polygon != 0)
    _, result = run(
        ValuePacket(rgb, context),
        {"mask": mask, "region": region},
        channel="R",
        intensity_units="normalized",
        emit_weights=True,
    )
    expected = source[..., 0].astype(float) / 65535 * selected
    mass, center, _ = reference(expected)
    np.testing.assert_allclose(result.outputs["center"], center)
    assert result.outputs["mass"] == pytest.approx(mass)
    assert result.outputs["image"].color_format == "rgba"
    np.testing.assert_array_equal(result.outputs["image"], source)
    for name in ("image", "center", "pose", "moments", "weights", "support"):
        assert result.payloads[name]["ticket"] == ticket
        assert result.payloads[name]["spatial"] == spatial
        assert result.payloads[name]["pyramid"] == pyramid
        assert result.payloads[name]["encoder"] == 800
    _, luminance = run(rgb, {"mask": mask})
    expected_luma = (0.299 * source[..., 0] + 0.587 * source[..., 1] + 0.114 * source[..., 2]) * (
        mask != 0
    )
    np.testing.assert_allclose(luminance.outputs["center"], reference(expected_luma)[1])


def test_ambiguous_and_point_support_have_a_center_without_an_invented_axis():
    for source in (np.ones((8, 8), np.uint8), np.array([[0, 0, 3]], np.uint8)):
        block, result = run(source)
        assert result.passed and result.outputs["center"] is not NO_VALUE
        assert result.outputs["angle_degrees"] is NO_VALUE and result.outputs["pose"] is NO_VALUE
        block.parameters["require_orientation"] = True
        failed = block.execute({"image": source})
        assert failed is not None and not failed.passed
        assert failed.fail_reasons[0]["code"] == "ambiguous_moment_orientation"
        block.parameters.update(require_orientation=False, pose_orientation="translation_only")
        translated = block.execute({"image": source})
        pose = validate_pose(translated.outputs["pose"])
        assert pose["angle_degrees"] == 0
        np.testing.assert_allclose(pose["origin"], result.outputs["center"])


@pytest.mark.parametrize("policy,passed", [("fail", False), ("report", True)])
def test_missing_center_policies_and_threshold_limits(policy, passed):
    block, result = run(np.zeros((10, 12), np.uint8), missing_center=policy)
    assert result.passed == passed
    assert result.outputs["center"] is NO_VALUE and result.outputs["pose"] is NO_VALUE
    assert result.outputs["measurements"]["invalid_reasons"] == ["empty_moment_support"]
    block.parameters.update(minimum_mass=4)
    limited = block.execute({"image": np.ones((2, 2), np.uint8)})
    assert limited.outputs["center"] is NO_VALUE  # Strictly above mass, not inclusive.
    block.parameters.update(minimum_mass=0, minimum_support=5)
    assert block.execute({"image": np.ones((2, 2), np.uint8)}).outputs["center"] is NO_VALUE


def test_pose_consumption_and_graphics_optimization():
    image = np.zeros((80, 100), np.uint16)
    image[20:50, 25:75] = 4000
    image[35:50, 60:75] = 24000
    block, result = run(image, emit_weights=True)
    pose = validate_pose(result.outputs["pose"])
    direction = np.asarray(pose["source_to_target"])[:2, 0]
    covariance = np.asarray(result.outputs["measurements"]["weighted"]["covariance"])
    variance = float(result.outputs["measurements"]["weighted"]["major_variance"])
    np.testing.assert_allclose(covariance @ direction, variance * direction)
    region = builtin_registry().create("region.pose", parameters={"shape": "circle", "radius": 4})
    placed = region.execute({"image": image, "pose": result.outputs["pose"]})
    assert placed is not None, region.error
    assert np.count_nonzero(placed.outputs["mask"]) > 30
    assert result.drawings["Intensity moments"].objects
    block.set_graphics_mode("off")
    lean = block.execute({"image": image})
    assert lean is not None and not lean.graphics and not lean.drawings
    assert lean.outputs["overlay"] is NO_VALUE
    assert lean.outputs["measurements"] == result.outputs["measurements"]
    np.testing.assert_array_equal(lean.outputs["weights"], result.outputs["weights"])


@pytest.mark.parametrize(
    "parameters,message",
    [
        ({"channel": "A"}, "unavailable"),
        ({"minimum_support": 0}, "between"),
        ({"minimum_anisotropy": 2}, "between"),
        ({"ellipse_sigma": 0}, "between"),
        ({"background": float("nan")}, "finite"),
        ({"polarity": "negative"}, "one of"),
        ({"emit_weights": "yes"}, "boolean"),
    ],
)
def test_invalid_parameters_are_contained(parameters, message):
    block = IntensityMoments(parameters=parameters)
    assert block.execute({"image": np.ones((20, 20), np.uint8)}) is None
    assert message in block.error and block.state == "error"


def test_frame_mismatch_mass_overflow_pixel_budget_and_native_exception(monkeypatch):
    block = IntensityMoments()
    assert block.execute({"image": np.full((2, 2), 1e308, np.float64)}) is None
    assert "mass exceeds" in block.error
    assert block.execute({"image": np.zeros((2100, 2100), np.uint8)}) is None
    assert "4.2 MP" in block.error
    image = np.ones((20, 20), np.uint8)
    assert (
        block.execute(
            {
                "image": ValuePacket(image, {"spatial": frame([20, 20], "a")}),
                "mask": ValuePacket(image, {"spatial": frame([20, 20], "b")}),
            }
        )
        is None
    )
    assert "frame differs" in block.error

    def broken(*args, **kwargs):
        raise cv2.error("moment backend unavailable")

    monkeypatch.setattr(cv2, "moments", broken)
    assert block.execute({"image": image}) is None
    assert "moment backend unavailable" in block.error and not block.current_emissions


@pytest.mark.parametrize("graphics", ["on", "off"])
def test_original_example_matches_independent_rectangle_mixture(graphics, tmp_path):
    graph = Graph.load(EXAMPLE / "project.json", builtin_registry())
    truth = json.loads((EXAMPLE / "assets/truth.json").read_text(encoding="utf-8"))
    try:
        for block in graph.blocks.values():
            block.set_graphics_mode(graphics)
        graph.run()
        assert all(b.state == "pass" for b in graph.processing_blocks())
        weighted = graph.blocks["weighted"].last_result
        geometric = graph.blocks["geometric"].last_result
        np.testing.assert_allclose(weighted.outputs["center"], truth["weighted_center"], atol=1e-9)
        np.testing.assert_allclose(
            geometric.outputs["center"], truth["geometric_center"], atol=1e-9
        )
        assert weighted.outputs["mass"] == pytest.approx(truth["mass"])
        assert abs(weighted.outputs["angle_degrees"] - truth["weighted_angle_degrees"]) < 1e-9
        np.testing.assert_allclose(
            weighted.outputs["measurements"]["weighted"]["covariance"],
            truth["weighted_covariance"],
            atol=1e-9,
        )
        saved = tmp_path / "roundtrip.json"
        graph.save(saved)
        restored = Graph.load(saved, builtin_registry())
        try:
            assert restored.blocks["weighted"].parameters == graph.blocks["weighted"].parameters
            assert restored.blocks["weighted"].position == graph.blocks["weighted"].position
        finally:
            restored.close()
    finally:
        graph.close()
