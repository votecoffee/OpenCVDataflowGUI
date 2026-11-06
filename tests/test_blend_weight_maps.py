"""Check pixel weighting, selection, and the feathered image example."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import frame, metadata
from opencv_dataflow.tickets import TicketClock


@pytest.mark.parametrize("dtype", (np.uint8, np.uint16, np.int16, np.float32))
def test_single_weight_map_uses_complement_and_preserves_pixel_depth(dtype):
    a = np.array([[32, 32, 32]], dtype=dtype)
    b = np.array([[192, 192, 192]], dtype=dtype)
    block = builtin_registry().create("cv.blend_weight_maps")
    result = block.execute({"image_a": a, "image_b": b, "weight_a": np.array([[0, 0.25, 1]])})
    assert result is not None and result.passed, block.error
    output = result.outputs["image"]
    np.testing.assert_array_equal(output, [[192, 152, 32]])
    assert output.dtype == dtype


@pytest.mark.parametrize(("mode", "fallback"), (("keep_a", 10), ("zero", 0)))
def test_relative_weights_normalize_and_handle_zero_coverage(mode, fallback):
    block = builtin_registry().create("cv.blend_weight_maps", parameters={"zero_weight": mode})
    result = block.execute(
        {
            "image_a": np.full((1, 3), 10, np.uint8),
            "image_b": np.full((1, 3), 90, np.uint8),
            "weight_a": np.array([[1, 0, 1e-300]], np.float64),
            "weight_b": np.array([[3 / 4, 0, 1e-300]], np.float64),
        }
    )
    assert result is not None and result.passed, block.error
    np.testing.assert_array_equal(result.outputs["image"], [[44, fallback, 50]])


def test_uint16_weights_apply_in_stored_rgb_order_inside_the_selection():
    a = np.full((2, 2, 3), (65535, 8, 6), np.uint16)
    b = np.full((2, 2, 3), (1, 14, 50), np.uint16)
    spatial = frame([2, 2], "camera")
    region = {
        "model": "image_region_v1",
        "image_size": [2, 2],
        "shape": "rectangle",
        "rectangle": [0, 0, 2, 1],
    }
    block = builtin_registry().create("cv.blend_weight_maps")
    result = block.execute(
        {
            "image_a": ImageArray(a, "rgb", {"spatial": spatial}),
            "image_b": ImageArray(b, "rgb", {"spatial": spatial}),
            "weight_a": np.zeros((2, 2), np.uint16),
            "mask": np.array([[255, 0], [255, 255]], np.uint8),
            "region": region,
        }
    )
    assert result is not None and result.passed, block.error
    expected = a.copy()
    expected[0, 0] = b[0, 0]
    np.testing.assert_array_equal(result.outputs["image"], expected)
    assert result.outputs["image"].color_format == "rgb"
    assert metadata(result.outputs["image"])["spatial"] == spatial


def test_large_float32_samples_do_not_overflow_and_keep_small_secondary_weights():
    limit = np.finfo(np.float32).max
    a = np.array([[limit, -limit, 0]], np.float32)
    b = np.array([[limit, limit, limit]], np.float32)
    block = builtin_registry().create("cv.blend_weight_maps")
    result = block.execute(
        {
            "image_a": a,
            "image_b": b,
            "weight_a": np.full((1, 3), 1.0),
            "weight_b": np.array([[1, 1, 1e-10]]),
        }
    )
    assert result is not None and result.passed, block.error
    np.testing.assert_allclose(result.outputs["image"], [[limit, 0, float(limit) * 1e-10]])


@pytest.mark.parametrize(
    "weights", (np.array([[-0.01]]), np.array([[1.01]]), np.zeros((1, 1, 3), np.uint8))
)
def test_invalid_weight_maps_are_rejected(weights):
    block = builtin_registry().create("cv.blend_weight_maps")
    pixels = np.zeros((1, 1), np.uint8)
    assert block.execute({"image_a": pixels, "image_b": pixels, "weight_a": weights}) is None
    assert "weight_a" in block.error


def test_frame_policy_and_ticket_identity_cover_weight_maps():
    pixels = np.ones((1, 1), np.uint8)
    a = ImageArray(pixels, "gray", {"spatial": frame([1, 1], "a")})
    different = ImageArray(pixels, "gray", {"spatial": frame([1, 1], "b")})
    registry = builtin_registry()
    strict = registry.create("cv.blend_weight_maps")
    assert strict.execute({"image_a": a, "image_b": a, "weight_a": different}) is None
    assert "same coordinate frame" in strict.error
    paired = registry.create("cv.blend_weight_maps", parameters={"alignment": "pixel_grid"})
    clock = TicketClock(session="test")
    first = ImageArray(a, "gray", {**metadata(a), "ticket": clock.issue("a", "fixed", 1)})
    other = ImageArray(
        different, "gray", {**metadata(different), "ticket": clock.issue("b", "fixed", 2)}
    )
    assert paired.execute({"image_a": first, "image_b": a, "weight_a": other}) is None
    assert "same ticket" in paired.error
    result = paired.execute({"image_a": a, "image_b": a, "weight_a": other})
    assert result is not None and result.passed, paired.error
    assert metadata(result.outputs["image"])["ticket"]["value"] == 2


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_example_keeps_both_endpoints_and_feathers_the_transition(profile):
    project = (
        Path(__file__).parents[1] / "examples" / "blend_weight_maps" / "blend_weight_maps.json"
    )
    graph = Graph.load(project, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    a = graph.blocks["source"].last_result.outputs["image"]
    b = graph.blocks["blur"].last_result.outputs["image"]
    weights = graph.blocks["feather"].last_result.outputs["image"]
    output = graph.blocks["blend"].last_result.outputs["image"]
    np.testing.assert_array_equal(output[weights == 255], a[weights == 255])
    np.testing.assert_array_equal(output[weights == 0], b[weights == 0])
    alpha = weights.astype(np.float64)[..., None] / 255
    reference = np.rint(a * alpha + b * (1 - alpha))
    np.testing.assert_allclose(output, reference, atol=1)
    between = (weights > 0) & (weights < 255)
    assert (output[between] != a[between]).any() and (output[between] != b[between]).any()
