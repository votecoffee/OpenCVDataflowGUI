"""Verify native histogram unwrap, shadow support and region reference semantics."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.phase_unwrap import UnwrapFringePhase
from opencv_dataflow.tickets import TicketClock

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/phase_unwrap"
TAU = 2 * np.pi


def _asset(name):
    return cv2.imread(str(EXAMPLE / "assets" / name), cv2.IMREAD_UNCHANGED)


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_measured_fixture_recovers_independent_absolute_regions(profile):
    graph = Graph.load(EXAMPLE / "phase_unwrap.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        for block in graph.processing_blocks():
            assert block.last_result and block.last_result.passed, (block.id, block.error)
        relative = graph.blocks["relative"].last_result
        based = graph.blocks["referenced"].last_result
        source = graph.blocks["wrapped"].last_result
        data = based.outputs
        chosen = _asset("shadow_mask.png") != 0
        truth = _asset("reference_phase.tiff")
        assert data["report"]["valid_pixels"] == chosen.sum() == 8127
        np.testing.assert_array_equal(data["valid_mask"], chosen * 255)
        assert np.isnan(data["unwrapped_phase"][~chosen]).all()
        assert np.isnan(data["inverse_reliability"][~chosen]).all()
        assert data["inverse_reliability"][30, 30] < 1e-4
        assert data["inverse_reliability"][30, 63] == pytest.approx(16 * np.pi**2)
        rows = data["report"]["components"]
        assert [row["status"] for row in rows] == ["anchored", "anchored"]
        assert [row["offset_cycles"] for row in rows] == [-3, 4]
        for row in rows:
            region = data["component_labels"] == row["id"]
            delta = data["unwrapped_phase"][region] - relative.outputs["unwrapped_phase"][region]
            np.testing.assert_allclose(delta, row["offset_cycles"] * TAU, atol=1e-12)
        err = data["unwrapped_phase"][chosen] - truth[chosen]
        assert np.max(np.abs(err)) < 3e-5
        assert np.sqrt(np.mean(err**2)) < 1.1e-5
        for field in ("spatial", "ticket"):
            assert based.payloads["unwrapped_phase"][field] == source.payloads["image"][field]
        assert based.payloads["unwrapped_phase"]["phase_unwrap"]["anchored_labels"] == [1, 2]
        assert relative.payloads["unwrapped_phase"]["phase_unwrap"]["relative_labels"] == [1, 2]
        assert bool(based.graphics) == (profile == "debug")
    finally:
        graph.close()


def test_wrapped_fixture_is_demodulated_from_recorded_integer_exposures():
    a, b, c = (_asset(f"fringe_{i}.png").astype(np.float64) for i in range(3))
    assert a.shape == (64, 144)
    phase = np.arctan2(2 * a - b - c, np.sqrt(3) * (b - c))
    np.testing.assert_allclose(_asset("wrapped_phase.tiff"), phase, atol=1.25e-7, rtol=0)


def _case():
    y, x = np.mgrid[:12, :21]
    truth = 11.5 + x * 0.7 + y * 0.12
    phase = ((truth + np.pi) % TAU - np.pi).astype(np.float32)
    mask = np.full(phase.shape, 7, np.uint8)
    mask[:, 9:11] = 0
    return phase, mask, truth


def test_nondefault_histogram_and_shadow_support_match_direct_native_reference():
    wrapped, mask, _ = _case()
    wrapped[4, 5] = np.nan
    wrapped[3, 4] = np.inf
    include = (mask != 0) & np.isfinite(wrapped)
    native_in = np.where(include, wrapped, 0).astype(np.float32)
    p = cv2.phase_unwrapping.HistogramPhaseUnwrapping.Params()
    p.height, p.width = wrapped.shape
    p.histThresh, p.nbrOfSmallBins, p.nbrOfLargeBins = 20, 12, 8
    reference = cv2.phase_unwrapping.HistogramPhaseUnwrapping.create(p)
    absolute_native = reference.unwrapPhaseMap(native_in, shadowMask=include.astype(np.uint8) * 255)
    inverse = reference.getInverseReliabilityMap()
    block = UnwrapFringePhase(
        parameters={"histogram_threshold": 20, "small_bins": 12, "large_bins": 8}
    )
    outcome = block.execute({"wrapped_phase": wrapped, "shadow_mask": mask})
    assert outcome and outcome.passed, block.error
    out = outcome.outputs
    for record in out["report"]["components"]:
        matched = out["component_labels"] == record["id"]
        np.testing.assert_allclose(
            out["unwrapped_phase"][matched],
            absolute_native[matched].astype(float) + record["offset_cycles"] * TAU,
            atol=1e-12,
        )
    np.testing.assert_array_equal(out["inverse_reliability"][include], inverse[include])
    np.testing.assert_array_equal(out["valid_mask"], include * 255)
    assert np.isnan(out["inverse_reliability"][~include]).all()
    repeated = block.execute({"wrapped_phase": wrapped, "shadow_mask": mask})
    np.testing.assert_array_equal(repeated.outputs["unwrapped_phase"], out["unwrapped_phase"])


@pytest.mark.parametrize(
    ("fault", "reason"),
    [
        ("conflict", "conflicting_reference_cycles"),
        ("off_phase", "reference_out_of_tolerance"),
        ("missing", "missing_reference"),
    ],
)
def test_rejected_reference_withholds_its_component_and_retains_other_component(fault, reason):
    wrapped, mask, truth = _case()
    anchors = [{"pixel": [14, 3], "phase_rad": float(truth[3, 14])}]
    if fault == "conflict":
        anchors += [
            {"pixel": [3, 3], "phase_rad": float(truth[3, 3])},
            {"pixel": [4, 3], "phase_rad": float(truth[3, 4] + TAU)},
        ]
    elif fault == "off_phase":
        anchors.append({"pixel": [4, 3], "phase_rad": float(truth[3, 4] + 0.1)})
    block = UnwrapFringePhase(parameters={"reference_mode": "required", "references": anchors})
    outcome = block.execute({"wrapped_phase": wrapped, "shadow_mask": mask})
    assert outcome and not outcome.passed and not block.error
    assert [r["code"] for r in outcome.fail_reasons] == [f"phase_unwrap_{reason}"]
    out = outcome.outputs
    assert np.isnan(out["unwrapped_phase"][:, :9]).all()
    assert np.isfinite(out["inverse_reliability"][:, :9]).all()
    np.testing.assert_allclose(out["unwrapped_phase"][:, 11:], truth[:, 11:], atol=3e-6, rtol=0)
    assert out["report"]["components"][0]["status"] == reason
    assert out["report"]["components"][1]["status"] == "anchored"


def test_diagonal_only_pixels_are_separate_components():
    phase = np.zeros((3, 3), np.float32)
    mask = np.zeros((3, 3), np.uint8)
    phase[1, 1] = np.pi
    mask[0, 0] = mask[1, 1] = 255
    block = UnwrapFringePhase(parameters={"reference_mode": "required"})
    result = block.execute(
        {
            "wrapped_phase": phase,
            "shadow_mask": mask,
            "references": [
                {"pixel": [0, 0], "phase_rad": 0},
                {"pixel": [1, 1], "phase_rad": 5 * np.pi},
            ],
        }
    )
    assert result and result.passed, block.error
    rows = result.outputs["report"]["components"]
    assert [r["status"] for r in rows] == ["anchored", "anchored"]
    assert [r["offset_cycles"] for r in rows] == [0, 2]
    assert result.outputs["unwrapped_phase"][1, 1] == pytest.approx(5 * np.pi, abs=1e-6)


def test_empty_selection_emits_nan_and_fails_cleanly_without_native_call():
    block = UnwrapFringePhase()
    result = block.execute(
        {"wrapped_phase": np.full((4, 4), 0, np.float32), "shadow_mask": np.zeros((4, 4), np.uint8)}
    )
    assert result and not result.passed and not block.error
    assert result.outputs["report"]["components"] == []
    assert np.isnan(result.outputs["unwrapped_phase"]).all()
    assert np.isnan(result.outputs["inverse_reliability"]).all()


@pytest.mark.parametrize(
    ("fault", "error"),
    [
        ("frame", "coordinate frame differs"),
        ("ticket", "same ticket"),
        ("anchor_mask", "outside valid phase"),
        ("wrapped_range", "radians within"),
        ("half_turn", "ambiguous half-turn"),
        ("bin_limit", "stable native rounding"),
    ],
)
def test_configuration_and_phase_contract(fault, error):
    wrapped, mask, _ = _case()
    arguments = {"wrapped_phase": wrapped, "shadow_mask": mask}
    settings = {}
    if fault == "frame":
        for key, value in arguments.items():
            arguments[key] = ValuePacket(value, {"spatial": frame((21, 12), key)})
    elif fault == "ticket":
        clock = TicketClock()
        for key, value in arguments.items():
            arguments[key] = ValuePacket(value, {"ticket": clock.issue(key)})
    elif fault == "anchor_mask":
        arguments["references"] = [{"pixel": [9, 2], "phase_rad": 2}]
    elif fault == "wrapped_range":
        wrapped[3, 3] = 4
    elif fault == "half_turn":
        wrapped[3, 3:5] = [0, np.pi]
    else:
        settings = {"histogram_threshold": 45, "large_bins": 7}
    block = UnwrapFringePhase(parameters=settings)
    assert block.execute(arguments) is None and block.state == "error"
    assert error in block.error
