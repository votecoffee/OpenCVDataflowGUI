"""Check registered fusion against known metric surfaces and support controls."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow import depth_fusion
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.depth_fusion import FuseRegisteredDepth
from opencv_dataflow.images import DepthArray
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock

EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples/opencv tools/registered_depth_fusion/registered_depth_fusion.json"
)


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_fusion_preserves_thin_foreground_and_fuses_complementary_holes(profile):
    graph = Graph.load(EXAMPLE, builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        block = graph.blocks["fuse"]
        result = block.last_result
        assert result and result.passed, block.error
        outputs = result.outputs
        shape = (96, 144)
        expected = np.full(shape, 2.0)
        confidence = np.full(shape, 2 / 3)
        support = np.full(shape, 2)
        conflict = np.zeros(shape, np.uint8)
        thin = np.s_[14:78, 67:69]
        expected[thin] = 1.25
        confidence[thin] = support[thin] = 1
        conflict[thin] = 255
        for ys, xs, z, weight, count in (
            (slice(24, 34), slice(19, 29), 1.97, 1 / 3, 1),
            (slice(42, 54), slice(90, 101), 2.01, 1.0, 1),
            (slice(15, 19), slice(117, 125), 2.01, 1.0, 1),
            (slice(81, 85), slice(102, 107), np.nan, 0, 0),
        ):
            expected[ys, xs] = z
            confidence[ys, xs] = weight
            support[ys, xs] = count
        assert outputs["depth"].units == "m"
        np.testing.assert_allclose(outputs["depth"], expected, atol=3e-7, rtol=0)
        np.testing.assert_allclose(outputs["confidence"], confidence, atol=1e-7, rtol=0)
        np.testing.assert_array_equal(outputs["support_count"], support)
        np.testing.assert_array_equal(outputs["conflict_mask"], conflict)
        np.testing.assert_array_equal(outputs["valid_mask"], np.isfinite(expected) * 255)
        report = outputs["report"]
        assert report["fused_pixels"] == 13804
        assert report["conflicting_pixels"] == report["views"][1]["farther_samples"] == 128
        assert report["coverage"] == 13804 / 13824
        assert report["no_eligible_depth_pixels"] == 20
        assert report["insufficient_support_pixels"] == 0
        source = graph.blocks["depth_a"].last_result
        for name in ("depth", "confidence", "support_count", "valid_mask", "conflict_mask"):
            for field in ("spatial", "ticket"):
                assert result.payloads[name][field] == source.payloads["depth"][field]
        verify = graph.blocks["verify"].last_result
        assert verify and verify.passed, graph.blocks["verify"].error
        comparison = verify.outputs["report"]
        assert comparison["bad_pixels"] == 0
        assert comparison["mae_mm"] == pytest.approx(4640 / 13804, abs=0.001)
        if profile == "debug":
            np.testing.assert_array_equal(result.graphics["Fusion support"][14, 67], [0, 255, 255])
        else:
            assert not result.graphics
    finally:
        graph.close()


def test_four_view_weighted_result_against_independent_pixel_reference(monkeypatch):
    monkeypatch.setattr(depth_fusion, "_BAND_ROWS", 1)
    a = DepthArray([[1000, 1000, 1250], [0, 1000, 1000]], "mm")
    b = np.array([[1625, 1250, 1250], [1000, 1000, 1200]], np.uint16)
    c = DepthArray([[1.125, 1.5, 1.875], [1.25, 1.25, np.nan]], "m")
    d = DepthArray([[1.25, 0.75, 0], [1.0, 1.125, 1.125]], "m")
    wb = np.array([[65535, 32768, 0], [65535, 32768, 65535]], np.uint16)
    wc = np.array([[0.5, 0.75, 1], [0.5, 0.25, 1]], np.float64)
    wd = np.array([[255, 0, 255], [85, 255, 255]], np.uint8)
    block = FuseRegisteredDepth(
        parameters={"surface_tolerance_m": 0.25, "minimum_confidence": 0.25}
    )
    result = block.execute(
        {
            "depth_a": a,
            "depth_b": b,
            "depth_c": c,
            "depth_d": d,
            "confidence_b": wb,
            "confidence_c": wc,
            "confidence_d": wd,
        }
    )
    assert result and result.passed, block.error
    values = [np.asarray(a) / 1000, b / 1000.0, np.asarray(c), np.asarray(d)]
    weights = [np.ones(a.shape), wb / 65535.0, wc, wd / 255.0]
    for index in np.ndindex(a.shape):
        candidates = [
            (z[index], w[index])
            for z, w in zip(values, weights, strict=True)
            if w[index] >= 0.25 and z[index] > 0
        ]
        closest = min(z for z, _ in candidates)
        same_surface = [(z, w) for z, w in candidates if z <= closest + 0.25]
        outputs = result.outputs
        assert outputs["depth"][index] == pytest.approx(
            sum(z * w for z, w in same_surface) / sum(w for _, w in same_surface), abs=2e-7
        )
        assert outputs["confidence"][index] == pytest.approx(
            sum(w for _, w in same_surface) / len(same_surface)
        )
        assert outputs["support_count"][index] == len(same_surface)
        assert bool(outputs["conflict_mask"][index]) == (len(candidates) != len(same_surface))
    np.testing.assert_array_equal(b, [[1625, 1250, 1250], [1000, 1000, 1200]])


def test_selection_and_minimum_support_keep_conflict_diagnostics_after_failure():
    a = DepthArray([[1, 1, 1, 0], [1, 1, 1, 1]], "m")
    b = DepthArray([[1, 2, 1, 0], [1, 1, 1, 1]], "m")
    mask = np.array([[255, 255, 0, 255], [255, 255, 255, 255]], np.uint8)
    region = {
        "model": "image_region_v1",
        "shape": "rectangle",
        "rectangle": [0, 0, 4, 1],
        "image_size": [4, 2],
    }
    block = FuseRegisteredDepth(parameters={"minimum_support": 2, "minimum_coverage": 0.5})
    result = block.execute({"depth_a": a, "depth_b": b, "mask": mask, "region": region})
    assert result and not result.passed and not block.error
    outputs = result.outputs
    assert result.fail_reasons[0]["code"] == "depth_fusion_coverage_below_minimum"
    assert outputs["report"]["insufficient_support_pixels"] == 1
    assert outputs["report"]["no_eligible_depth_pixels"] == 1
    assert outputs["report"]["coverage"] == 1 / 3
    np.testing.assert_array_equal(outputs["support_count"][0], [2, 1, 0, 0])
    np.testing.assert_array_equal(outputs["conflict_mask"][0], [0, 255, 0, 0])
    np.testing.assert_array_equal(outputs["confidence"][0], [1, 0, 0, 0])
    assert np.isnan(outputs["depth"][1]).all()
    empty = block.execute({"depth_a": a, "depth_b": b, "mask": np.zeros(a.shape, np.uint8)})
    assert empty and not empty.passed
    assert empty.fail_reasons[0]["code"] == "depth_fusion_no_supported_pixels"


def test_alignment_and_ticket_checks_prevent_fusion_of_unrelated_inputs():
    a = DepthArray(np.ones((3, 3)), "m")
    base = {
        "depth_a": ValuePacket(a, {"spatial": frame((3, 3), "camera-a")}),
        "depth_b": ValuePacket(a, {"spatial": frame((3, 3), "camera-b")}),
    }
    block = FuseRegisteredDepth()
    assert block.execute(base) is None and "one common registered pixel frame" in block.error
    block = FuseRegisteredDepth(parameters={"alignment": "pixel_grid"})
    assert block.execute(base).passed
    clock = TicketClock("test")
    for name in ("depth_a", "depth_b"):
        base[name].payload["ticket"] = clock.issue(name, "fixed", name)
    assert block.execute(base) is None and "same ticket" in block.error


@pytest.mark.parametrize(
    ("overrides", "settings", "error"),
    [
        ({"depth_b": DepthArray([[1, 2]], "m")}, {}, "matching dimensions"),
        ({"confidence_b": np.full((2, 2), np.nan, np.float32)}, {}, "finite float"),
        ({"confidence_a": np.ones((2, 2, 3), np.uint8)}, {}, "monochrome"),
        ({"confidence_d": np.ones((2, 2), np.float32)}, {}, "matching depth"),
        ({}, {"minimum_support": 3}, "exceeds connected depth views"),
    ],
)
def test_invalid_depth_and_weight_contracts(overrides, settings, error):
    values = DepthArray(np.ones((2, 2)), "m")
    block = FuseRegisteredDepth(parameters=settings)
    assert block.execute({"depth_a": values, "depth_b": values, **overrides}) is None
    assert error in block.error
