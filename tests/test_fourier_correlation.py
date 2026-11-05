"""Verify signed lags, masks and the native correlation against direct sums."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock


def test_linear_fourier_correlation_matches_direct_sums_for_different_shapes_and_masks():
    reference = np.array([[0, 2, 1], [3, -1, 4]], np.int16)
    image = np.array([[2, 3, 2, 0], [4, 2, 5, 1], [1, 3, 4, 2]], np.uint16)
    mask = np.ones(image.shape, np.uint8)
    mask[1, 2] = 0
    reference_mask = np.ones(reference.shape, np.uint8)
    reference_mask[0, 0] = reference_mask[1, 1] = 0
    ticket = TicketClock().issue("sample")
    sample = ValuePacket(image, {"spatial": frame([4, 3], "sample"), "ticket": ticket})
    block = builtin_registry().create(
        "inspect.fourier_correlation", parameters={"normalization": "none"}
    )
    result = block.execute(
        {"image": sample, "reference": reference, "mask": mask, "reference_mask": reference_mask}
    )
    assert result is not None, block.error
    output = result.outputs["correlation"]
    expected = np.zeros((4, 6))
    for y, x in np.ndindex(expected.shape):
        dy, dx = y - 1, x - 2
        for yy, xx in np.ndindex(reference.shape):
            u, v = yy + dy, xx + dx
            if 0 <= u < 3 and 0 <= v < 4 and mask[u, v] and reference_mask[yy, xx]:
                expected[y, x] += int(image[u, v]) * int(reference[yy, xx])
    np.testing.assert_allclose(output, expected, atol=1e-12)
    y, x = np.unravel_index(expected.argmax(), expected.shape)
    assert result.outputs["shift"]["x"] == x - 2
    assert result.outputs["shift"]["y"] == y - 1
    assert result.outputs["peak"] == pytest.approx(expected.max())
    assert result.graphics["lag map"].shape == (*expected.shape, 3)
    payload = result.payloads["correlation"]
    assert payload["lag_origin"] == [2, 1]
    assert payload["ticket"] == ticket
    assert "spatial" not in payload


@pytest.mark.parametrize("shape", [(9, 14), (10, 15)])
def test_circular_fourier_correlation_recovers_signed_roll_with_different_dc_levels(shape):
    reference = np.random.default_rng(15).normal(size=shape)
    image = np.roll(reference, shift=(3, -4), axis=(0, 1)) * 12 + 30
    block = builtin_registry().create(
        "inspect.fourier_correlation", parameters={"boundary": "circular", "subtract_mean": True}
    )
    result = block.execute({"image": image, "reference": reference})
    assert result is not None, block.error
    assert result.outputs["shift"]["x"] == -4
    assert result.outputs["shift"]["y"] == 3
    assert result.outputs["peak"] == pytest.approx(1.0)
    assert result.payloads["correlation"]["lag_origin"] == [shape[1] // 2, shape[0] // 2]


def test_fourier_correlation_rejects_empty_masks_and_flat_signals():
    registry = builtin_registry()
    block = registry.create("inspect.fourier_correlation", parameters={"boundary": "circular"})
    image = np.ones((4, 7), np.uint8)
    assert block.execute({"image": image, "reference": image[:, :4]}) is None
    assert "equal image dimensions" in block.error
    assert block.execute({"image": image, "reference": image, "mask": np.zeros_like(image)}) is None
    assert "contains no pixels" in block.error
    assert block.execute({"image": image, "reference": image}) is None
    assert "flat across all displacements" in block.error
    block.parameters["subtract_mean"] = True
    assert block.execute({"image": image, "reference": image}) is None
    assert block.error_kind == "invalid" and "zero correlation energy" in block.error
