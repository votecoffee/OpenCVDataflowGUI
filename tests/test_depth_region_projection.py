"""Independent wide-angle projection, buffer ownership and grid/ticket checks."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow import depth_region_projection
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.depth_region_projection import ReprojectDepthRegion
from opencv_dataflow.geometry3d_common import camera_record, pose_record
from opencv_dataflow.images import DepthArray
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.selection import rasterize
from opencv_dataflow.tickets import TicketClock

EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples/opencv tools/depth_region_projection/depth_region_projection.json"
)


def analytic_candidates(source_z):
    """Project a zero-distortion equidistant ray through a unified camera by hand."""
    best = {}
    cos, sin = np.cos(0.14), np.sin(0.14)
    for y in range(26, 77):
        for x in range(42, 115):
            z = float(source_z[y, x]) / 1000
            if not np.isfinite(z):
                continue
            dx, dy = (x - 79.5) / 96, (y - 54.5) / 93
            theta = np.hypot(dx, dy)
            scale = np.tan(theta) / theta if theta else 1
            sx, sy = (dx * scale * z, dy * scale * z)
            tx, ty, tz = (cos * sx + sin * z - 0.07, sy + 0.013, -sin * sx + cos * z + 0.04)
            divisor = tz + 0.55 * np.sqrt(tx**2 + ty**2 + tz**2)
            u = int(np.floor(122 * tx / divisor + 72))
            v = int(np.floor(120 * ty / divisor + 50))
            assert 0 <= u < 144 and 0 <= v < 100
            previous = best.get((v, u))
            if previous is None or tz < previous[0]:
                best[v, u] = (tz, y, x)
    return best


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_wide_angle_transfer_matches_analytic_footprint_and_z(profile):
    graph = Graph.load(EXAMPLE, builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        block = graph.blocks["transfer"]
        result = block.last_result
        assert result and result.passed, block.error
        source = graph.blocks["source"].last_result
        target = graph.blocks["target"].last_result
        best = analytic_candidates(source.outputs["depth"])
        measured = target.outputs["depth"]
        accepted = np.zeros(measured.shape, bool)
        occluded = accepted.copy()
        unverified = accepted.copy()
        expected = np.full(measured.shape, np.nan)
        support = np.zeros(source.outputs["depth"].shape, bool)
        for (v, u), (z, y, x) in best.items():
            delta = z - float(measured[v, u]) / 1000
            if delta > 0.012:
                occluded[v, u] = True
            elif not np.isfinite(delta) or delta < -0.012:
                unverified[v, u] = True
            else:
                accepted[v, u] = True
                expected[v, u] = z
                support[y, x] = True
        outputs = result.outputs
        np.testing.assert_array_equal(outputs["mask"], accepted * 255)
        np.testing.assert_array_equal(outputs["occluded_mask"], occluded * 255)
        np.testing.assert_array_equal(outputs["unverified_mask"], unverified * 255)
        np.testing.assert_array_equal(outputs["source_support"], support * 255)
        np.testing.assert_allclose(outputs["depth"], expected, atol=2e-7, rtol=0)
        np.testing.assert_array_equal(rasterize(outputs["region"], measured.shape), outputs["mask"])
        assert outputs["depth"].units == "m"
        assert result.payloads["depth"]["spatial"] == target.payloads["depth"]["spatial"]
        assert result.payloads["source_support"]["spatial"] == source.payloads["depth"]["spatial"]
        assert result.payloads["mask"]["ticket"] == source.payloads["depth"]["ticket"]
        assert outputs["region"]["coordinate_context"] == result.payloads["mask"]["spatial"]
        assert outputs["report"]["retained_target_pixels"] == 2400
        assert outputs["report"]["colliding_samples"] == 1138
        assert outputs["report"]["occluded_target_pixels"] == 132
        assert outputs["report"]["unverified_target_pixels"] == 11
        assert outputs["report"]["invalid_source_depth_pixels"] == 42
        measure = graph.blocks["measure"]
        assert measure.last_result and measure.last_result.passed, measure.error
        assert measure.last_result.outputs["area"] == 2400
        if profile == "debug":
            np.testing.assert_array_equal(result.graphics["Coverage"][43, 65], [0, 0, 255])
        else:
            assert not result.graphics
    finally:
        graph.close()


def _inputs(source_values, source_size=None, target_size=None, lens="pinhole"):
    shape = source_values.shape[::-1]
    source_size = source_size or shape
    target_size = target_size or shape
    k = [[45, 0, shape[0] / 2], [0, 47, shape[1] / 2], [0, 0, 1]]
    d = [0.01, -0.001, 0, 0]
    models = {"lens": lens, **({"xi": 0.6} if lens == "omnidir" else {})}
    return {
        "source_depth": source_values,
        "source_camera": camera_record(k, d, source_size, "m", "source", **models),
        "target_camera": camera_record(k, d, target_size, "mm", "target", **models),
        "transform": pose_record(np.eye(4), "source", "target", "mm"),
    }


@pytest.mark.parametrize("lens", ["pinhole", "fisheye", "omnidir"])
def test_identity_with_distortion_preserves_circular_selection_and_holes(lens):
    source = DepthArray(np.ones((9, 13)), "m")
    source[3, 6] = np.nan
    mask = np.full(source.shape, 255, np.uint8)
    mask[:, :6] = 0
    region = {
        "model": "image_region_v1",
        "shape": "circle",
        "center": [6, 4],
        "radius": 3,
        "image_size": [13, 9],
    }
    y, x = np.indices(source.shape)
    expected = (x >= 6) & ((x - 6) ** 2 + (y - 4) ** 2 <= 9) & np.isfinite(source)
    block = ReprojectDepthRegion(parameters={"emit_region": False})
    result = block.execute({**_inputs(source, lens=lens), "mask": mask, "region": region})
    assert result and result.passed, block.error
    outputs = result.outputs
    np.testing.assert_array_equal(outputs["mask"], expected * 255)
    np.testing.assert_array_equal(outputs["source_support"], expected * 255)
    np.testing.assert_array_equal(outputs["depth"], np.where(expected, 1, np.nan))
    np.testing.assert_array_equal(outputs["unverified_mask"], outputs["mask"])
    assert outputs["region"] is NO_VALUE


def test_closest_positive_z_wins_with_equal_depth_ties_across_projection_batches(monkeypatch):
    monkeypatch.setattr(depth_region_projection, "_BATCH", 1)
    raw = np.array([[1400, 1000, 1000, 0]], np.uint16)
    inputs = _inputs(raw, target_size=(1, 1))
    inputs["target_camera"] = camera_record(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [], (1, 1), "m", "target"
    )
    block = ReprojectDepthRegion()
    result = block.execute(inputs)
    assert result and result.passed, block.error
    np.testing.assert_array_equal(result.outputs["source_support"], [[0, 255, 0, 0]])
    np.testing.assert_array_equal(result.outputs["depth"], [[1]])
    assert result.outputs["report"]["colliding_samples"] == 2


def test_visibility_modes_distinguish_occluded_ahead_and_unobserved_depth():
    source = DepthArray(np.ones((3, 3)), "m")
    measured = DepthArray([[1.04, 0.96, np.nan], [1, 1, 1], [1, 1, 1]], "m")
    target_context = {"spatial": frame((3, 3), "measured")}
    inputs = {
        **_inputs(source),
        "target_depth": ValuePacket(measured, target_context),
        "target_image": ValuePacket(np.zeros((3, 3, 3), np.uint8), target_context),
    }
    for policy, row in (("reject_occluded", [255, 0, 255]), ("require_agreement", [0, 0, 0])):
        block = ReprojectDepthRegion(parameters={"visibility_policy": policy})
        result = block.execute(inputs)
        assert result and result.passed, block.error
        o = result.outputs
        np.testing.assert_array_equal(o["mask"][0], row)
        np.testing.assert_array_equal(o["occluded_mask"][0], [0, 255, 0])
        np.testing.assert_array_equal(o["unverified_mask"][0], [255, 0, 255])
        assert result.payloads["mask"]["spatial"] == target_context["spatial"]
        assert result.drawings


def test_empty_projection_keeps_diagnostics_and_retention_limit_is_a_normal_failure():
    source = DepthArray(np.ones((4, 4)), "m")
    inputs = _inputs(source)
    translation = np.eye(4)
    translation[2, 3] = -2
    block = ReprojectDepthRegion()
    result = block.execute(
        {**inputs, "transform": pose_record(translation, "source", "target", "m")}
    )
    assert result and not result.passed
    assert result.outputs["report"]["outside_target_depth_samples"] == 16
    assert result.fail_reasons[0]["code"] == "depth_region_no_coverage"
    assert np.isnan(result.outputs["depth"]).all()
    source[2, 2] = np.nan
    block = ReprojectDepthRegion(parameters={"minimum_retained_fraction": 1})
    result = block.execute(inputs)
    assert result and not result.passed
    assert result.fail_reasons[0]["code"] == "depth_region_retention_below_minimum"
    assert result.outputs["report"]["retained_source_fraction"] == 15 / 16


def test_strict_mode_frame_and_ticket_validation_fail_before_projecting():
    data = DepthArray(np.ones((4, 4)), "m")
    inputs = _inputs(data)
    block = ReprojectDepthRegion(parameters={"visibility_policy": "require_agreement"})
    assert block.execute(inputs) is None and "target depth input" in block.error

    block = ReprojectDepthRegion()
    wrong = pose_record(np.eye(4), "target", "source", "m")
    assert block.execute({**inputs, "transform": wrong}) is None
    assert "source to target optical frames" in block.error
    clock = TicketClock("test")
    a, b = (clock.issue(key, "fixed", key) for key in ("first", "second"))
    disconnected = {
        **inputs,
        "source_depth": ValuePacket(data, {"ticket": a}),
        "target_depth": ValuePacket(data, {"ticket": b}),
    }
    assert block.execute(disconnected) is None and "same ticket" in block.error
    disconnected = {
        **inputs,
        "target_depth": ValuePacket(data, {"spatial": frame((4, 4), "target-a")}),
        "target_image": ValuePacket(
            np.zeros((4, 4), np.uint8), {"spatial": frame((4, 4), "target-b")}
        ),
    }
    assert block.execute(disconnected) is None and "different pixel frames" in block.error


def test_fisheye_inverse_rejects_samples_outside_the_positive_z_lens_domain():
    source = DepthArray([[1, 1, 1, 1]], "m")
    inputs = _inputs(source, lens="fisheye")
    k = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for key in ("source", "target"):
        inputs[f"{key}_camera"] = camera_record(k, [0] * 4, (4, 1), "m", key, "fisheye")
    block = ReprojectDepthRegion()
    result = block.execute(inputs)
    assert result and result.passed, block.error
    assert result.outputs["report"]["invalid_source_rays"] == 2
    np.testing.assert_array_equal(result.outputs["mask"], [[255, 255, 0, 0]])
