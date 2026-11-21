"""Qualify native gradient similarity, spatial support and the saved project."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame, moved_payload
from opencv_dataflow.tickets import TicketClock

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/gradient_quality"
TYPE = "inspect.gmsd_quality"


def block(**parameters):
    return builtin_registry().create(TYPE, parameters=parameters)


def reference_pair():
    rng = np.random.default_rng(5542)
    ref = rng.integers(0, 256, (35, 51), dtype=np.uint8)
    cmp = ref.copy()
    cmp[6:25, 8:35] = rng.integers(0, 256, (19, 27), dtype=np.uint8)
    return ref, cmp


def gradients(image, shape):
    padded = np.pad(image.astype(np.float64), ((0, 1), (0, 1)))
    ow, oh = shape[::-1]
    tiles = np.lib.stride_tricks.sliding_window_view(padded, (2, 2))[::2, ::2][:oh, :ow]
    reduced = np.pad(tiles.mean(axis=(2, 3)), 1)
    grid = np.lib.stride_tricks.sliding_window_view(reduced, (3, 3))
    dx = (grid[..., 2] - grid[..., 0]).mean(axis=-1)
    dy = (grid[..., 2, :] - grid[..., 0, :]).mean(axis=-1)
    return np.hypot(dx, dy)


def test_native_map_matches_independent_prewitt_reference_on_odd_grid():
    ref, cmp = reference_pair()
    tool = block()
    result = tool.execute({"reference": ref, "image": cmp})
    assert result and result.passed, tool.error
    image = result.outputs["map"]
    assert image.dtype == np.float32 and image.shape == (18, 26)
    left, right = (gradients(src, image.shape) for src in (ref, cmp))
    expected = (2 * left * right + 170) / (left**2 + right**2 + 170)
    np.testing.assert_allclose(image, expected, rtol=0, atol=4e-7)
    assert result.outputs["value"] == pytest.approx(expected.std(), abs=4e-7)


def test_per_channel_uses_native_bgra_order_and_reports_threshold_failure():
    a, b = reference_pair()
    raw = np.dstack((a, a, a, a))
    changed = np.dstack((b, a, a, b // 2))
    scores, expected = cv2.quality.QualityGMSD_compute(raw, changed)
    ref = ImageArray(raw[..., [2, 1, 0, 3]], "rgba")
    cmp = ImageArray(changed[..., [2, 1, 0, 3]], "rgba")
    tool = block(
        channel_mode="per_channel",
        channel_reduction="mean",
        check_threshold=True,
        maximum_deviation=0,
    )
    result = tool.execute({"reference": ref, "image": cmp})
    assert result and not result.passed, tool.error
    assert result.fail_reasons[0]["code"] == "gmsd_above_limit"
    assert "exceeds" in result.fail_reasons[0]["message"]
    assert result.outputs["measurements"]["channel_values"] == dict(
        zip(("blue", "green", "red", "alpha"), scores, strict=True)
    )
    assert result.outputs["value"] == pytest.approx(np.mean(scores))
    np.testing.assert_array_equal(result.outputs["map"], expected)


def test_mask_selects_complete_tiles_and_preserves_native_map():
    ref, cmp = reference_pair()
    mask = np.zeros_like(ref)
    mask[6:22, 8:28] = 255
    mask[6, 8] = 0
    mask[-1, -1] = 255
    tool = block()
    result = tool.execute({"reference": ref, "image": cmp, "mask": mask})
    assert result and result.passed, tool.error
    quality = result.outputs["map"]
    _, expected = cv2.quality.QualityGMSD_compute(ref, cmp)
    np.testing.assert_array_equal(quality, expected)
    selected = quality[3:11, 4:14].flatten()[1:]
    assert result.outputs["measurements"]["selected_map_cells"] == 79
    assert result.outputs["value"] == pytest.approx(np.std(selected, dtype=np.float64))
    assert not result.graphics["Gradient similarity loss"][3, 4].any()
    assert not result.graphics["Gradient similarity loss"][-1, -1].any()

    # A selected partial tile at the padded odd edge is not a measurement cell.
    mask[:] = 0
    mask[-1, -1] = 255
    assert tool.execute({"reference": ref, "image": cmp, "mask": mask}) is None
    assert "complete 2" in tool.error


def test_map_and_scalar_follow_the_inspection_ticket_and_grid():
    ref, cmp = reference_pair()
    original = frame([200, 145], "inspection")
    origin = np.array([[1, 0, 14], [0, 1, 8], [0, 0, 1]])
    mapped = moved_payload({"spatial": original}, origin, [51, 35])
    check = TicketClock("test").issue("camera", "fixed", "item-under-inspection")
    inputs = {
        "reference": ValuePacket(ref, {"spatial": frame([51, 35], "reference")}),
        "image": ValuePacket(cmp, {**mapped, "ticket": check}),
    }
    tool = block()
    result = tool.execute(inputs)
    assert result and result.passed, tool.error
    assert result.payloads["value"]["ticket"] == result.payloads["map"]["ticket"] == check
    context = result.payloads["map"]["spatial"]
    assert context["root_id"] == "inspection"
    assert context["image_size"] == [26, 18]
    np.testing.assert_allclose(context["to_root"], [[2, 0, 14.5], [0, 2, 8.5], [0, 0, 1]])
    strict = block(image_alignment="same_frame")
    assert strict.execute(inputs) is None and "different spatial frames" in strict.error


def test_map_can_be_hidden_and_invalid_inputs_stay_in_the_block():
    reference, current = reference_pair()
    tool = block(channel_mode="per_channel")
    tool.exposed_outputs = ("value", "measurements")
    result = tool.execute({"reference": reference, "image": current})
    assert result and result.passed and result.outputs["map"] is NO_VALUE
    assert tool.execute({"reference": reference, "image": current.astype(np.uint16)}) is None
    assert tool.error_kind == "invalid" and "8-bit" in tool.error
    assert tool.execute({"reference": reference, "image": current[:, :-1]}) is None
    assert "matching image grids" in tool.error
    color = np.repeat(current[..., None], 3, axis=2)
    assert tool.execute({"reference": reference, "image": color}) is None
    assert "Per-channel" in tool.error


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_project_orders_identity_blur_and_noise(profile):
    graph = Graph.load(EXAMPLE / "gradient_quality.json", builtin_registry())
    graph.set_run_profile(profile)
    graph.set_branch_workers(2)
    try:
        graph.run()
        values = []
        for condition in ("identity", "blur", "noise"):
            tool = graph.blocks[f"{condition}_quality"]
            result = tool.last_result
            assert result and result.passed, tool.error
            values.append(result.outputs["value"])
            assert tuple(result.outputs["map"].shape) == (96, 128, 3)
            assert bool(result.graphics) == (profile == "debug")
        assert values[0] == pytest.approx(0, abs=1e-6)
        assert values[0] < values[1] < values[2]
    finally:
        graph.close()
