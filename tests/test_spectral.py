"""Check native spectral transforms through the graph block contracts."""

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import ValuePacket, frame, packet_for
from opencv_dataflow.tickets import TicketClock


@pytest.mark.parametrize(
    "mode,shape,padded,dtype",
    [
        ("2d", (11, 17), (12, 18), np.uint8),
        ("rows", (11, 17), (11, 18), np.uint16),
        ("2d", (9, 1), (10, 1), np.float64),
        ("rows", (9, 1), (9, 1), np.float32),
    ],
)
def test_dct_round_trip_preserves_source_size_depth_and_frame(mode, shape, padded, dtype):
    image = np.random.default_rng(26).integers(0, 256, size=shape).astype(dtype)
    context = frame(list(shape[::-1]), "fixture")
    registry = builtin_registry()
    forward = registry.create("cv.dct", parameters={"mode": mode})
    ticket = TicketClock().issue("fixture", "fixed", 38)
    result = forward.execute({"image": ValuePacket(image, {"spatial": context, "ticket": ticket})})
    assert result is not None, forward.error
    packet = result.outputs["coefficients"]
    assert packet.values.shape == padded
    assert result.graphics["magnitude"].shape == padded
    inverse = registry.create("cv.idct")
    restored = inverse.execute({"coefficients": packet_for(result, "coefficients")})
    assert restored is not None, inverse.error
    output = restored.outputs["image"]
    assert output.dtype == dtype and output.shape == shape
    assert restored.payloads["image"]["spatial"] == context
    assert restored.payloads["image"]["ticket"] == ticket
    if image.dtype.kind == "u":
        np.testing.assert_array_equal(output, image)
    else:
        np.testing.assert_allclose(output, image, atol=1e-4, rtol=1e-6)


def test_dct_matches_native_values_and_validates_odd_policy_and_packets():
    image = np.arange(24, dtype=np.float32).reshape(3, 8)
    registry = builtin_registry()
    forward = registry.create("cv.dct", parameters={"mode": "rows", "odd_size": "reject"})
    result = forward.execute({"image": image})
    assert result is not None, forward.error
    packet = result.outputs["coefficients"]
    np.testing.assert_allclose(packet.values, cv2.dct(image, flags=cv2.DCT_ROWS))
    assert forward.execute({"image": image[:, :-1]}) is None
    assert forward.error_kind == "invalid" and "even transformed dimensions" in forward.error
    inverse = registry.create("cv.idct")
    assert inverse.execute({"coefficients": replace(packet, image_shape=(4, 8))}) is None
    assert inverse.error_kind == "invalid"


@pytest.mark.parametrize("mode,padded", [("2d", (12, 18, 2)), ("rows", (11, 18, 2))])
def test_dft_round_trip_preserves_shape_and_depth_with_optimal_padding(mode, padded):
    image = np.random.default_rng(40).integers(0, 65536, (11, 17), dtype=np.uint16)
    registry = builtin_registry()
    forward = registry.create("cv.dft", parameters={"mode": mode, "size": "optimal"})
    result = forward.execute({"image": image})
    assert result is not None, forward.error
    assert result.outputs["spectrum"].values.shape == padded
    assert result.graphics["magnitude"].shape == padded[:2]
    inverse = registry.create("cv.idft")
    restored = inverse.execute({"spectrum": packet_for(result, "spectrum")})
    assert restored is not None, inverse.error
    output = restored.outputs["image"]
    assert output.dtype == np.uint16
    np.testing.assert_array_equal(output, image)


@pytest.mark.parametrize(
    "mode,cutoff,upper_cutoff,expected",
    [
        ("low_pass", 0.1, 0.3, "slow"),
        ("high_pass", 0.1, 0.3, "stripes"),
        ("band_pass", 0.2, 0.3, "stripes"),
        ("band_stop", 0.2, 0.3, "slow"),
    ],
)
def test_fourier_filter_selects_known_frequencies_and_preserves_frame(
    mode, cutoff, upper_cutoff, expected
):
    height, width = 48, 64
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    slow = 0.5 + 0.12 * np.cos(2 * np.pi * y / 24)
    stripes = 0.18 * np.cos(2 * np.pi * x / 4)
    image = (slow + stripes).astype(np.float32)
    context = frame([width, height], "stripe-fixture")
    registry = builtin_registry()
    forward = registry.create("cv.dft")
    transformed = forward.execute({"image": ValuePacket(image, {"spatial": context})})
    assert transformed is not None, forward.error
    band = registry.create(
        "cv.filter_fourier_spectrum",
        parameters={"mode": mode, "cutoff": cutoff, "upper_cutoff": upper_cutoff},
    )
    filtered = band.execute({"spectrum": packet_for(transformed, "spectrum")})
    assert filtered is not None, band.error
    inverse = registry.create("cv.idft")
    result = inverse.execute({"spectrum": packet_for(filtered, "spectrum")})
    assert result is not None, inverse.error
    assert result.payloads["image"]["spatial"] == context
    assert result.outputs["image"].dtype == np.float32
    np.testing.assert_allclose(
        result.outputs["image"], slow if expected == "slow" else stripes, atol=2e-7
    )


@pytest.mark.parametrize("shape,mode", [((11, 17), "2d"), ((12, 18), "rows")])
def test_fourier_gain_notch_preserves_source_frame_and_discards_gain_ticket(shape, mode):
    height, width = shape
    x = np.arange(width)
    pattern = np.broadcast_to(np.cos(2 * np.pi * 3 * x / width), shape)
    image = (0.5 + 0.2 * pattern).astype(np.float32)
    registry = builtin_registry()
    source_context = frame([width, height], "source")
    source = ValuePacket(image, {"spatial": source_context})
    forward = registry.create("cv.dft", parameters={"mode": mode})
    transformed = forward.execute({"image": source})
    assert transformed is not None, forward.error
    gain = np.full(shape, 65535, np.uint16)
    if mode == "rows":
        gain[:, width // 2 + 3] = 0
    else:
        gain[height // 2, width // 2 + 3] = 0
    mask = ValuePacket(
        gain, {"spatial": frame([width, height], "gain"), "ticket": TicketClock().issue("gain")}
    )
    inputs = {"spectrum": packet_for(transformed, "spectrum"), "gain": mask}
    tool = registry.create("cv.mask_fourier_spectrum")
    assert tool.execute(inputs) is None
    assert tool.error_kind == "invalid" and "paired frequencies" in tool.error
    tool.parameters["symmetry"] = "minimum"
    filtered = tool.execute(inputs)
    assert filtered is not None, tool.error
    assert filtered.payloads["spectrum"]["spatial"] == source_context
    assert "ticket" not in filtered.payloads["spectrum"]
    assert filtered.graphics["magnitude"].shape == shape
    inverse = registry.create("cv.idft")
    restored = inverse.execute({"spectrum": packet_for(filtered, "spectrum")})
    assert restored is not None, inverse.error
    np.testing.assert_allclose(restored.outputs["image"], 0.5, atol=2e-7)


@pytest.mark.parametrize("policy,gain_value", [("average", 0.5), ("maximum", 0.75)])
def test_fourier_gain_native_pairs_support_fractional_gains(policy, gain_value):
    image = np.cos(2 * np.pi * np.arange(10) / 10)[None, :]
    registry = builtin_registry()
    forward = registry.create("cv.dft")
    transformed = forward.execute({"image": image})
    assert transformed is not None, forward.error
    gain = np.ones(image.shape, np.float64)
    gain[0, 1], gain[0, -1] = 0.25, 0.75
    tool = registry.create(
        "cv.mask_fourier_spectrum", parameters={"layout": "native", "symmetry": policy}
    )
    filtered = tool.execute({"spectrum": packet_for(transformed, "spectrum"), "gain": gain})
    assert filtered is not None, tool.error
    inverse = registry.create("cv.idft")
    result = inverse.execute({"spectrum": packet_for(filtered, "spectrum")})
    assert result is not None, inverse.error
    np.testing.assert_allclose(result.outputs["image"], image * gain_value, atol=1e-12)


def test_fourier_gain_requires_padded_dimensions_and_nonnegative_values():
    image = np.ones((11, 17), np.uint8)
    registry = builtin_registry()
    forward = registry.create("cv.dft", parameters={"size": "optimal"})
    transformed = forward.execute({"image": image})
    assert transformed is not None, forward.error
    tool = registry.create("cv.mask_fourier_spectrum")
    spectrum = packet_for(transformed, "spectrum")
    assert tool.execute({"spectrum": spectrum, "gain": image}) is None
    assert "full spectrum dimensions" in tool.error
    gain = np.ones((12, 18), np.float32)
    gain[0, 0] = -1
    assert tool.execute({"spectrum": spectrum, "gain": gain}) is None
    assert "nonnegative" in tool.error
    gain.fill(2)
    filtered = tool.execute({"spectrum": spectrum, "gain": gain})
    assert filtered is not None, tool.error
    assert filtered.outputs["spectrum"].image_shape == image.shape


@pytest.mark.parametrize("mode", ("2d", "rows"))
@pytest.mark.parametrize("operation", ("multiply", "multiply_conjugate", "conjugate"))
def test_combine_fourier_spectra_round_trips_in_the_expected_coordinates(mode, operation):
    registry = builtin_registry()
    a = np.zeros((4, 9), np.float32)
    b = np.zeros_like(a, dtype=np.float64)
    a[1, 2] = 3
    b[1, 4] = 2
    source_frame = frame([9, 4], "camera")
    first = registry.create("cv.dft", parameters={"mode": mode}).execute(
        {"image": ValuePacket(a, {"spatial": source_frame})}
    )
    second = registry.create("cv.dft", parameters={"mode": mode}).execute({"image": b})
    combine = registry.create(
        "cv.combine_fourier_spectra", block_id="combine", parameters={"operation": operation}
    )
    inputs = {"spectrum": packet_for(first, "spectrum")}
    axes = (-1,) if mode == "rows" else (-2, -1)
    expected = np.fft.fftn(a, axes=axes)
    if operation == "conjugate":
        expected = expected.conj()
    else:
        inputs["spectrum_b"] = second.outputs["spectrum"]
        other = np.fft.fftn(b, axes=axes)
        expected *= other.conj() if operation == "multiply_conjugate" else other
    result = combine.execute(inputs)
    assert result is not None, combine.error
    values = result.outputs["spectrum"].values
    assert values.dtype == (np.float32 if operation == "conjugate" else np.float64)
    assert result.graphics["magnitude"].shape == a.shape

    inverse = registry.create("cv.idft", parameters={"output_depth": "float64"})
    restored = inverse.execute({"spectrum": packet_for(result, "spectrum")})
    assert restored is not None, inverse.error
    np.testing.assert_allclose(
        restored.outputs["image"], np.fft.ifftn(expected, axes=axes).real, atol=1e-6
    )
    payload = restored.payloads["image"]
    if operation == "multiply":
        assert payload["spatial"] == source_frame
    else:
        assert payload["spatial"] == frame([9, 4], "combine:spectrum")
        assert payload.get("lag_origin") == ([0, 0] if operation == "multiply_conjugate" else None)


def test_combine_fourier_spectra_rejects_incompatible_grids_and_overflow():
    registry = builtin_registry()
    image = np.ones((7, 11), np.float32)
    spectrum = registry.create("cv.dft").execute({"image": image}).outputs["spectrum"]
    rows = registry.create("cv.dft", parameters={"mode": "rows"})
    padded = registry.create("cv.dft", parameters={"size": "optimal"})
    block = registry.create("cv.combine_fourier_spectra")
    assert block.execute({"spectrum": spectrum}) is None
    assert block.error_kind == "invalid" and "Connect Spectrum B" in block.error
    for forward in (rows, padded):
        other = forward.execute({"image": image}).outputs["spectrum"]
        assert block.execute({"spectrum": spectrum, "spectrum_b": other}) is None
        assert block.error_kind == "invalid" and "bin grid" in block.error

    huge = replace(spectrum, values=np.full_like(spectrum.values, 1e30))
    assert block.execute({"spectrum": huge, "spectrum_b": huge}) is None
    assert block.error_kind == "invalid" and "overflowed" in block.error


def test_spectrum_product_example_finds_a_wrapped_displacement():
    path = (
        Path(__file__).parents[1]
        / "examples/opencv tools/spectrum_product/spectrum_product.json"
    )
    graph = Graph.load(path, builtin_registry())
    try:
        graph.set_run_profile("lean")
        graph.run()
        assert all(block.state == "pass" for block in graph.blocks.values())
        result = graph.blocks["peak"].last_result
        assert result.outputs["maximum_location"] == [17, 351]
    finally:
        graph.close()
