"""Qualify sparse stereo curves against scalar costs and the dense matcher."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.binary_stereo import BinaryStereoDisparity
from opencv_dataflow.binary_stereo_costs import InspectBinaryStereoCosts
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.calibration_revisions import digest
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock

EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples/opencv tools/binary_stereo_costs/binary_stereo_costs.json"
)


def _scalar_signature(patch, variant):
    size = len(patch)
    half = size // 2
    mean = sum(int(value) for row in patch for value in row) / size**2
    bits = []
    for y, row in enumerate(patch):
        for x, value in enumerate(row):
            if (y, x) == (half, half):
                continue
            if variant == "star" and y != half and x != half and abs(y - half) != abs(x - half):
                continue
            if variant == "symmetric":
                if y * size + x > size**2 // 2:
                    continue
                reference = patch[size - 1 - y, size - 1 - x]
            else:
                reference = mean if variant == "mean" else patch[half, half]
            bits.append(value < reference)
    return np.array(bits)


def _scalar_cost(left, right, x, y, d, variant, census=5, aggregation=3):
    pad = aggregation // 2
    radius = census // 2
    cost = 0
    for v in range(y - pad, y + pad + 1):
        ys = slice(v - radius, v + radius + 1)
        for u in range(x - pad, x + pad + 1):
            a = left[ys, u - radius : u + radius + 1]
            b = right[ys, u - d - radius : u - d + radius + 1]
            cost += np.count_nonzero(_scalar_signature(a, variant) != _scalar_signature(b, variant))
    return cost


@pytest.mark.parametrize(
    "variant,bits", [("center", 24), ("mean", 24), ("symmetric", 12), ("star", 16)]
)
def test_curves_match_independent_scalar_bits_and_signed_costs(variant, bits):
    rng = np.random.default_rng(1122)
    left, right = rng.integers(0, 256, (2, 35, 44), dtype=np.uint8)
    params = {"census_size": "5", "aggregation_size": 3, "min_disparity": -5, "num_disparities": 11}
    block = InspectBinaryStereoCosts(
        parameters={**params, "probe_points": [[14, 14], [26, 22]], "census_variant": variant}
    )
    inputs = {"left": left, "right": right}
    result = block.execute(inputs)
    assert result and result.passed, block.error
    assert result.outputs["report"]["bits_per_descriptor"] == bits
    assert result.outputs["report"]["maximum_cost"] == bits * 9
    for curve in result.outputs["curves"]:
        x, y = curve["left_xy"]
        expected = [_scalar_cost(left, right, x, y, d, variant) for d in range(-5, 6)]
        assert [row["cost"] for row in curve["candidates"]] == expected
        assert [row["right_x"] for row in curve["candidates"]] == list(range(x + 5, x - 6, -1))
        assert all(row["unavailable"] is None for row in curve["candidates"])
        assert [curve["minimum_cost"], curve["next_cost"]] == sorted(expected)[:2]
        if variant == "center":
            dense = BinaryStereoDisparity(parameters={**params, "left_right_check": False})
            output = dense.execute(inputs).outputs
            assert output["best_cost"][y, x] == curve["minimum_cost"]
            assert output["second_best_cost"][y, x] == curve["next_cost"]
            assert output["confidence"][y, x] == pytest.approx(curve["relative_gap"])


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_published_project_demonstrates_texture_ambiguity_and_occlusion(profile):
    graph = Graph.load(EXAMPLE, builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.set_branch_workers(2)
        graph.run()
        for name, other_cost, plot in (("center", 272, 1), ("symmetric", 138, 3)):
            block = graph.blocks[name]
            result = block.last_result
            assert result and result.passed, block.error
            textured, flat, occluded = result.outputs["curves"]
            assert textured["status"] == "matched"
            assert textured["selected_disparity_px"] == 8 and textured["minimum_cost"] == 0
            assert textured["next_cost"] == other_cost
            assert flat["status"] == "ambiguous"
            assert flat["minimum_disparities_px"] == list(range(16))
            assert flat["relative_gap"] == 0 and flat["selected_disparity_px"] is None
            assert occluded["status"] == "partial_support"
            assert occluded["supported_candidates"] == 4
            assert occluded["candidates"][8] == {
                "disparity_px": 8,
                "right_x": 166,
                "cost": None,
                "unavailable": "right_masked",
            }
            assert occluded["selected_disparity_px"] is None
            assert result.outputs["report"]["measured_candidates"] == 36
            assert (
                result.payloads["curves"]["spatial"]
                == graph.blocks["left"].last_result.payloads["image"]["spatial"]
            )
            if profile == "debug":
                assert result.graphics[f"Probe {plot} cost curve"].shape == (300, 640, 3)
                assert len(result.drawings["Stereo probe pixels"].objects) == 6
            else:
                assert not result.graphics and not result.drawings
    finally:
        graph.close()


def test_signed_depth_search_respects_each_calibrated_roi_and_connected_probe_frame():
    rng = np.random.default_rng(19597)
    left = rng.integers(0, 200, (32, 50), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, 6:] = left[:, :-6]
    rec = {
        "model": "stereo_rectification_v1",
        "units": "m",
        "image_size": [50, 32],
        "roi_left": [3, 2, 44, 28],
        "roi_right": [8, 2, 39, 28],
        "P2": [[200, 0, 0, 24], [0, 200, 0, 0], [0, 0, 1, 0]],
        "Q": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 200], [0, 0, -1 / 0.12, 0]],
    }
    ticket = TicketClock("census-probe").issue("pair", "fixed", "stereo")
    captures = {"rectification": rec}
    for side, data in (("left", left), ("right", right)):
        spatial = frame((50, 32), side)
        stamp = {"sha256": digest(rec), "side": side, "spatial": spatial}
        captures[side] = ValuePacket(
            data, {"spatial": spatial, "stereo_calibration_grid": stamp, "ticket": ticket}
        )
    points = [[18, 15], [6, 15], [40, 15]]
    block = InspectBinaryStereoCosts(
        parameters={
            "search_mode": "depth",
            "minimum_depth": 3.5,
            "maximum_depth": 5.5,
            "census_size": "3",
            "aggregation_size": 1,
            "plot_probe": 2,
        }
    )
    result = block.execute(
        {
            **captures,
            "points": ValuePacket(points, {"spatial": captures["left"].payload["spatial"]}),
        }
    )
    assert result and result.passed, block.error
    core, partial_left, partial_right = result.outputs["curves"]
    assert [c["disparity_px"] for c in core["candidates"]] == [-7, -6, -5, -4]
    assert core["selected_disparity_px"] == -6
    assert partial_left["left_xy"] == [6, 15] and partial_left["supported_candidates"] == 4
    assert partial_right["candidates"][0]["unavailable"] == "right_outside_roi"
    assert partial_right["selected_disparity_px"] is None
    assert result.payloads["image"]["ticket"] == ticket
    wrong = ValuePacket(points, {"spatial": captures["right"].payload["spatial"]})
    assert block.execute({**captures, "points": wrong}) is None
    assert "left pixel frame" in block.error
    assert (
        block.execute({**captures, "rectification": {**rec, "roi_right": [9, 2, 38, 28]}}) is None
    )
    assert "sealed" in block.error


def test_unsupported_footprint_has_no_cost_and_reports_valid_failure():
    image = np.zeros((20, 48), np.uint8)
    probe = InspectBinaryStereoCosts(parameters={"probe_points": [[1, 1]], "num_disparities": 6})
    result = probe.execute({"left": image, "right": image})
    assert result and not result.passed
    assert result.fail_reasons[0]["code"] == "binary_stereo_no_probe_support"
    curve = result.outputs["curves"][0]
    assert curve["supported_candidates"] == 0
    assert all(
        c["unavailable"] == "left_outside_roi" and c["cost"] is None for c in curve["candidates"]
    )
    assert probe.execute({"left": image, "right": image, "points": [[4.5, 3]]}) is None
    assert "whole" in probe.error
