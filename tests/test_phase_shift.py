"""Independent projector truth, native binding behavior and temporal group contracts."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import FlowGroup, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock

TYPE = "3d.phase_shift_decode"
PATTERNS = "3d.phase_shift_patterns"
EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/phase_shift"


def generate(axis="x", periods=8):
    generator = builtin_registry().create(
        PATTERNS, parameters={"width": 160, "height": 120, "axis": axis, "fine_periods": periods}
    )
    result = generator.execute({})
    assert result and result.passed, generator.error
    return {k: result.outputs[k] for k in ("patterns", "specification")}


@pytest.mark.parametrize(("axis", "periods"), [("x", 8), ("y", 8), ("x", 2)])
def test_generate_and_roundtrip_projector_axes(axis, periods):
    inputs = generate(axis, periods)
    block = builtin_registry().create(TYPE)
    result = block.execute(inputs)
    assert result and result.passed, block.error
    expected = np.indices((120, 160))[int(axis == "x")]
    np.testing.assert_allclose(result.outputs["projector_coordinate"], expected, atol=0.05)
    assert np.all(result.outputs["valid_mask"] == 255)
    assert result.outputs["report"]["fine_period_px"] == (160 if axis == "x" else 120) // periods


def test_native_psp_and_histogram_unwrapping_on_ideal_fine_triplet():
    module = cv2.structured_light
    params = module.SinusoidalPattern.Params()
    params.width, params.height, params.nbrOfPeriods, params.methodId = 160, 120, 8, module.PSP
    native = module.SinusoidalPattern.create(params)
    ok, frames = native.generate()
    assert ok
    generated = generate()["patterns"].items
    for observed, expected in zip(frames, generated[3:], strict=True):
        np.testing.assert_array_equal(observed, expected.value)
    wrapped, mask = native.computePhaseMap(frames)
    assert wrapped.shape == (120, 160) and wrapped.dtype == np.float32
    assert np.all(mask == 255)
    unwrapped = native.unwrapPhaseMap(wrapped, (160, 120), shadowMask=mask)
    relative = unwrapped - unwrapped[:, :1]
    expected = np.tile(np.arange(160) * (2 * np.pi / 20), (120, 1))
    np.testing.assert_allclose(relative, expected, atol=0.003)


@pytest.mark.parametrize("axis", ("x", "y"))
def test_saved_capture_replay_matches_independent_absolute_subpixel_truth(axis):
    graph = Graph.load(EXAMPLE / f"projector_{axis}.json", builtin_registry())
    try:
        graph.set_run_profile("debug" if axis == "x" else "lean")
        block = graph.blocks["decode"]
        for _ in range(5):
            graph.run()
            assert block.last_result is None, block.error
        graph.run()
        result = block.last_result
        assert result and result.passed, block.error
        labels = cv2.imread(str(EXAMPLE / "assets" / f"labels_{axis}.png"), cv2.IMREAD_GRAYSCALE)
        y, x = np.indices(labels.shape)
        expected = 12.35 + 1.15 * x + 0.035 * y if axis == "x" else 8.25 + 0.035 * x + 1.36 * y
        expected[:, 57:] += 3.6
        valid = labels == 0
        outputs = result.outputs
        np.testing.assert_array_equal(outputs["valid_mask"] != 0, valid)
        coordinate = outputs["projector_coordinate"]
        np.testing.assert_allclose(coordinate[valid], expected[valid], atol=0.001)
        assert np.isnan(coordinate[~valid]).all()
        report = outputs["report"]
        assert report["valid_pixels"] == 7167
        assert report["low_signal_pixels"] == 510 and report["unwrapping_rejected_pixels"] == 99
        cycles = outputs["absolute_phase"][valid]
        np.testing.assert_allclose(
            cycles, expected[valid] * (2 * np.pi / report["fine_period_px"]), atol=0.001
        )
        assert result.payloads["valid_mask"]["spatial"]["image_size"] == [108, 72]
        assert result.payloads["valid_mask"]["ticket"]["source"] == "batch"
        phase_context = result.payloads["projector_coordinate"]["phase_shift"]
        assert phase_context["axis"] == axis and phase_context["projector_size"] == [161, 121]
    finally:
        graph.close()


def test_selections_frame_context_and_ticket_propagation():
    inputs = generate()
    group = inputs["patterns"]
    anchor = frame([160, 120], "camera_grid")
    items = tuple(
        ValuePacket(packet.value, {**packet.payload, "spatial": anchor}) for packet in group.items
    )
    ticket = TicketClock().issue("group")
    inputs["patterns"] = ValuePacket(FlowGroup(items, group.labels), {"ticket": ticket})
    mask = np.zeros((120, 160), np.uint8)
    mask[:, :14] = 255
    region = {
        "model": "image_region_v1",
        "image_size": [160, 120],
        "shape": "rectangle",
        "rectangle": [6, 4, 20, 10],
        "coordinate_context": anchor,
    }
    block = builtin_registry().create(TYPE)
    result = block.execute(
        {**inputs, "mask": ValuePacket(mask, {"spatial": anchor}), "region": region}
    )
    assert result and result.passed, block.error
    assert result.outputs["report"]["valid_pixels"] == 80
    assert result.payloads["projector_coordinate"]["spatial"] == anchor
    assert result.payloads["projector_coordinate"]["ticket"] == ticket
    choices = result.outputs["valid_mask"] > 0
    assert choices[4:14, 6:14].all() and choices.sum() == 80


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("order", "Pattern index"),
        ("spec", "digest"),
        ("shape", "grids must match"),
        ("value", "normalized"),
        ("count", "six captures"),
        ("spatial", "shared known camera grid"),
    ],
)
def test_incompatible_phase_sets_are_rejected(mode, message):
    inputs = generate()
    group = inputs["patterns"]
    items = list(group.items)
    if mode == "order":
        items.reverse()
    elif mode == "spec":
        inputs["specification"] = {**inputs["specification"], "fine_period_px": 99}
    elif mode == "shape":
        items[3] = ValuePacket(np.ones((12, 16), np.uint16), {})
    elif mode == "value":
        items[3] = ValuePacket(np.full((120, 160), 1.1, np.float32), {})
    elif mode == "count":
        items.pop()
    elif mode == "spatial":
        geometry = frame([160, 120], "other")
        geometry["to_root"][0][2] = 1
        items[0] = ValuePacket(items[0].value, {"spatial": geometry})
    inputs["patterns"] = FlowGroup(tuple(items), group.labels[: len(items)])
    block = builtin_registry().create(TYPE)
    assert block.execute(inputs) is None
    assert message in block.error


def test_flat_and_missing_samples_fail_and_saturation_is_optional():
    inputs = generate()
    group = inputs["patterns"]
    block = builtin_registry().create(TYPE, parameters={"reject_saturation": True})
    result = block.execute(inputs)
    assert result and result.passed, block.error
    assert result.outputs["report"]["saturated_pixels"] > 0
    frame_values = np.full((120, 160), 0.5, np.float32)
    frame_values[0, 0] = np.nan
    inputs["patterns"] = FlowGroup(
        tuple(ValuePacket(frame_values, {}) for _ in group.items), group.labels
    )
    result = block.execute(inputs)
    assert result is not None and not result.passed
    assert result.fail_reasons[0]["code"] == "no_valid_phase_correspondences"
    assert not result.outputs["valid_mask"].any()
    assert np.isnan(result.outputs["projector_coordinate"]).all()
