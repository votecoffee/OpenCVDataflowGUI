"""Analytic Fourier components, real reconstruction and derived-grid contracts."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame, metadata, packet_for
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.spectral import DFTSpectrum
from opencv_dataflow.tickets import TicketClock

IMAGE = "cv.reconstruct_fourier_image"
COMPONENTS = "cv.fourier_components"
PROJECT = Path(__file__).parents[1] / "examples/opencv tools/fourier_components/project.json"


def reconstruct(image, parameters=None, extra=None):
    block = builtin_registry().create(IMAGE, parameters=parameters or {})
    result = block.execute({"image": image, **(extra or {})})
    assert result is not None, block.error
    return block, result


@pytest.mark.parametrize("components", ("full", "phase_only", "magnitude_only"))
@pytest.mark.parametrize("mode", ("2d", "rows"))
def test_impulse_has_an_independent_analytic_reconstruction(components, mode):
    image = np.zeros((17, 23), np.uint16)
    image[5, 7] = 32768
    _, result = reconstruct(image, {"components": components, "mode": mode})
    expected = np.zeros_like(image, dtype=float)
    position = (5, 7) if components != "magnitude_only" else (5 if mode == "rows" else 0, 0)
    expected[position] = 1 if components == "phase_only" else 32768 / 65535
    np.testing.assert_allclose(result.outputs["image"], expected, atol=1e-14)
    assert result.outputs["image"].dtype == np.float64


@pytest.mark.parametrize("mode", ("2d", "rows"))
def test_sinusoid_retains_phase_or_removes_it_and_uses_native_inverse_scaling(mode):
    h, w, phase = 8, 16, 0.4
    x = np.arange(w)
    image = np.broadcast_to(0.5 + 0.2 * np.cos(2 * np.pi * 2 * x / w + phase), (h, w)).copy()
    n = w if mode == "rows" else h * w
    _, phase_result = reconstruct(image, {"components": "phase_only", "mode": mode, "dc": "remove"})
    _, magnitude_result = reconstruct(
        image, {"components": "magnitude_only", "mode": mode, "dc": "remove"}
    )
    np.testing.assert_allclose(
        phase_result.outputs["image"],
        np.broadcast_to(2 / n * np.cos(2 * np.pi * 2 * x / w + phase), (h, w)),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        magnitude_result.outputs["image"],
        np.broadcast_to(0.2 * np.cos(2 * np.pi * 2 * x / w), (h, w)),
        atol=1e-14,
    )
    assert phase_result.outputs["image"].min() < 0


@pytest.mark.parametrize("value", (0.0, 2.0, -2.0))
@pytest.mark.parametrize("dc", ("keep", "remove"))
def test_zero_and_constant_signals_do_not_invent_undefined_phase(value, dc):
    _, result = reconstruct(np.full((5, 8), value), {"dc": dc})
    expected = np.sign(value) / 40 if dc == "keep" else 0
    np.testing.assert_allclose(result.outputs["image"], expected, atol=1e-15)
    assert result.outputs["info"]["planes"][0]["phase_valid_bins"] == int(
        value != 0 and dc == "keep"
    )


def test_phase_floors_are_strict_and_do_not_discard_full_or_magnitude_components():
    spectrum = np.zeros((1, 8, 2), np.float64)
    spectrum[0, 1, 0] = spectrum[0, -1, 0] = 0.25
    packet = DFTSpectrum(spectrum, "rows", (1, 8), "float64")
    registry = builtin_registry()
    for limits in ({"relative_floor": 1.0}, {"absolute_floor": 0.25}):
        block = registry.create(COMPONENTS, parameters=limits)
        result = block.execute({"spectrum": packet})
        assert result is not None, block.error
        assert (
            not result.outputs["spectrum"].values.any() and not result.outputs["valid_phase"].any()
        )
    for components in ("full", "magnitude_only"):
        block = registry.create(
            COMPONENTS, parameters={"components": components, "relative_floor": 1}
        )
        result = block.execute({"spectrum": packet})
        assert result is not None
        np.testing.assert_array_equal(result.outputs["spectrum"].values, spectrum)


def test_tiny_nonzero_phase_survives_without_squaring_underflow():
    values = np.zeros((1, 8, 2), np.float64)
    values[0, 1, 0] = values[0, -1, 0] = 1e-300
    block = builtin_registry().create(COMPONENTS, parameters={"relative_floor": 0})
    result = block.execute({"spectrum": DFTSpectrum(values, "rows", (1, 8), "float64")})
    assert result is not None, block.error
    assert np.count_nonzero(result.outputs["spectrum"].values) == 2
    assert result.outputs["spectrum"].values[0, 1, 0] == 1


def test_roundoff_pairs_are_cleaned_but_inconsistent_spectrum_frames_are_rejected():
    values = np.zeros((4, 6, 2), np.float32)
    values[0, 0, 0] = 1
    values[1, 1, 0] = values[-1, -1, 0] = 0.25
    values[1, 1, 1] = np.finfo(np.float32).eps
    packet = DFTSpectrum(values, "2d", (4, 6), "float32")
    block = builtin_registry().create(
        COMPONENTS, parameters={"components": "full", "layout": "native"}
    )
    result = block.execute({"spectrum": packet})
    assert result is not None, block.error
    modified = result.outputs["spectrum"].values
    assert modified[1, 1, 1] == -modified[-1, -1, 1]
    assert result.payloads["phase"]["spectral_value"]["units"] == "radians"
    assert result.outputs["info"]["diagnostic_stage"] == "DC-adjusted input spectrum"
    assert (
        block.execute({"spectrum": ValuePacket(packet, {"spatial": frame([7, 4], "wrong")})})
        is None
    )
    assert block.error_kind == "invalid" and "original image size" in block.error


@pytest.mark.parametrize("mode", ("2d", "rows"))
def test_symmetry_is_checked_or_explicitly_repaired_before_real_inverse(mode):
    values = np.random.default_rng(135).normal(size=(4, 6, 2))
    original = values.copy()
    packet = DFTSpectrum(values, mode, (4, 6), "float64")
    registry = builtin_registry()
    block = registry.create(COMPONENTS, parameters={"components": "full"})
    assert block.execute({"spectrum": packet}) is None
    assert block.error_kind == "invalid" and "conjugate symmetric" in block.error
    block.parameters["symmetry"] = "average"
    result = block.execute({"spectrum": packet})
    assert result is not None, block.error
    z = values[..., 0] + 1j * values[..., 1]
    x = -np.arange(6) % 6
    paired = z[:, x] if mode == "rows" else z[np.ix_(-np.arange(4) % 4, x)]
    expected = (z + paired.conj()) / 2
    modified = result.outputs["spectrum"].values
    np.testing.assert_allclose(modified[..., 0] + 1j * modified[..., 1], expected)
    inverse = registry.create("cv.idft")
    restored = inverse.execute({"spectrum": packet_for(result, "spectrum")})
    assert restored is not None, inverse.error
    reference = (
        np.fft.ifft(expected, axis=1).real if mode == "rows" else np.fft.ifft2(expected).real
    )
    np.testing.assert_allclose(restored.outputs["image"], reference, atol=1e-14)
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("mode", ("2d", "rows"))
def test_typed_components_keep_padding_and_prevent_implicit_integer_clipping(mode):
    image = np.random.default_rng(135).integers(0, 65536, (11, 17), dtype=np.uint16)
    registry = builtin_registry()
    forward = registry.create("cv.dft", parameters={"mode": mode, "size": "optimal"})
    transformed = forward.execute({"image": image})
    block = registry.create(COMPONENTS)
    result = block.execute({"spectrum": packet_for(transformed, "spectrum")})
    assert result is not None, block.error
    inverse = registry.create("cv.idft")
    restored = inverse.execute({"spectrum": packet_for(result, "spectrum")})
    assert restored is not None, inverse.error
    values = result.outputs["spectrum"].values
    z = values[..., 0] + 1j * values[..., 1]
    reference = np.fft.ifft(z, axis=1).real if mode == "rows" else np.fft.ifft2(z).real
    np.testing.assert_allclose(restored.outputs["image"], reference[:11, :17], atol=1e-7)
    assert restored.outputs["image"].dtype == np.float32 and restored.outputs["image"].min() < 0
    assert result.outputs["magnitude"].shape == values.shape[:2]
    assert result.payloads["magnitude"]["spatial"] != result.payloads["spectrum"]["spatial"]


def test_color_channels_alpha_source_and_phase_grid_preserve_input_provenance():
    image = np.zeros((9, 13, 4), np.uint16)
    for channel, position in enumerate(((2, 3), (4, 6), (7, 9))):
        image[*position, channel] = 20000 + 10000 * channel
    image[..., 3] = 40000
    payload = {
        "spatial": frame([13, 9], "camera"),
        "ticket": TicketClock("components").issue("camera"),
        "pyramid": {
            "level": 1,
            "largest_size": [26, 18],
            "to_largest": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        },
        "fixture": {"value": 135},
        "alpha_representation": "straight",
    }
    source = ImageArray(image, "rgba", payload)
    _, result = reconstruct(source)
    expected = np.zeros((9, 13, 3), float)
    for channel, position in enumerate(((2, 3), (4, 6), (7, 9))):
        expected[*position, channel] = 1
    np.testing.assert_allclose(result.outputs["image"], expected, atol=1e-14)
    assert result.outputs["image"].color_format == "rgb" and result.outputs["info"]["alpha_omitted"]
    np.testing.assert_array_equal(result.outputs["source"], source)
    assert result.outputs["source"].color_format == "rgba"
    for name in ("image", "source", "info"):
        for key in ("spatial", "ticket", "pyramid", "fixture"):
            assert result.payloads[name][key] == payload[key]
    assert "alpha_representation" not in result.payloads["image"]
    assert result.payloads["source"]["alpha_representation"] == "straight"
    assert metadata(result.outputs["image"]) == result.payloads["image"]


def test_zero_phase_reconstruction_has_a_new_grid_and_retains_source_identity():
    source = ImageArray(
        np.eye(8),
        "gray",
        {
            "spatial": frame([8, 8], "camera"),
            "pyramid": {
                "level": 1,
                "largest_size": [16, 16],
                "to_largest": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
            },
        },
    )
    _, result = reconstruct(source, {"components": "magnitude_only"})
    output = result.payloads["image"]
    assert output["spatial"]["root_id"] != "camera" and "pyramid" not in output
    assert output["spectral_source_grid"]["spatial"] == source.metadata["spatial"]
    assert output["spectral_source_grid"]["pyramid"] == source.metadata["pyramid"]
    assert result.payloads["source"]["spatial"] == source.metadata["spatial"]


@pytest.mark.parametrize("encoding", ("rgb", "bgr"))
def test_grayscale_conversion_uses_declared_color_order(encoding):
    image = ImageArray(np.tile(np.uint16([10000, 30000, 50000]), (4, 6, 1)), encoding)
    _, result = reconstruct(image, {"components": "full", "channels": "grayscale"})
    weights = [0.299, 0.587, 0.114] if encoding == "rgb" else [0.114, 0.587, 0.299]
    expected = np.dot([10000, 30000, 50000], weights) / 65535
    np.testing.assert_allclose(result.outputs["image"], expected, atol=1e-14)
    assert result.outputs["image"].color_format == "gray"


@pytest.mark.parametrize("normalization", ("unit_rms", "minmax"))
@pytest.mark.parametrize("scope", ("per_channel", "shared"))
def test_output_scaling_has_explicit_channel_scope(normalization, scope):
    plane = np.arange(24, dtype=float).reshape(4, 6) - 5
    image = ImageArray(np.stack((plane, plane * 2, plane * 4 + 1), axis=2), "rgb")
    _, result = reconstruct(
        image, {"components": "full", "normalization": normalization, "normalization_scope": scope}
    )
    expected = np.asarray(image).copy()
    axes = (0, 1) if scope == "per_channel" else (0, 1, 2)
    if normalization == "unit_rms":
        expected /= np.sqrt(np.mean(expected**2, axis=axes, keepdims=True))
    else:
        low, high = expected.min(axis=axes, keepdims=True), expected.max(axis=axes, keepdims=True)
        expected = (expected - low) / (high - low)
    np.testing.assert_allclose(result.outputs["image"], expected, atol=1e-14)


def test_flat_normalization_and_source_selection_have_defined_behavior():
    for normalization in ("minmax", "unit_rms"):
        _, result = reconstruct(np.zeros((4, 6)), {"normalization": normalization})
        assert not result.outputs["image"].any()
    image = np.random.default_rng(135).random((8, 12))
    mask = np.ones((8, 12), np.uint8)
    mask[:, :3] = 0
    region = {
        "model": "image_region_v1",
        "shape": "rectangle",
        "image_size": [12, 8],
        "rectangle": [0, 0, 9, 8],
    }
    _, result = reconstruct(image, {"components": "full"}, {"mask": mask, "region": region})
    expected = image.copy()
    expected[:, :3] = expected[:, 9:] = 0
    np.testing.assert_allclose(result.outputs["image"], expected, atol=1e-14)
    np.testing.assert_array_equal(result.outputs["source"], image)
    assert result.outputs["info"]["selected_source_pixels"] == 48
    _, phase = reconstruct(image, extra={"mask": mask, "region": region})
    assert np.any(np.abs(phase.outputs["image"][:, :3]) > 1e-5)


def test_phase_and_magnitude_packets_can_be_combined_without_losing_phase_coordinates():
    registry = builtin_registry()
    packets = []
    for position, amplitude, component in (
        ((2, 3), 1.0, "phase_only"),
        ((5, 7), 0.25, "magnitude_only"),
    ):
        image = np.zeros((8, 10))
        image[position] = amplitude
        forward = registry.create("cv.dft")
        source = forward.execute(
            {"image": ValuePacket(image, {"spatial": frame([10, 8], str(position))})}
        )
        select = registry.create(COMPONENTS, parameters={"components": component})
        selected = select.execute({"spectrum": packet_for(source, "spectrum")})
        packets.append(packet_for(selected, "spectrum"))
    combined = registry.create("cv.combine_fourier_spectra").execute(
        {"spectrum": packets[0], "spectrum_b": packets[1]}
    )
    inverse = registry.create("cv.idft").execute({"spectrum": packet_for(combined, "spectrum")})
    expected = np.zeros((8, 10))
    expected[2, 3] = 0.25
    np.testing.assert_allclose(inverse.outputs["image"], expected, atol=1e-14)
    assert inverse.payloads["image"]["spatial"] == packets[0].payload["spatial"]


def test_invalid_formats_limits_and_selection_frames_are_rejected():
    bad = [
        ImageArray(np.ones((4, 6, 3)), "hsv"),
        np.full((4, 6), np.nan),
        np.zeros((1500, 1500), np.uint8),
    ]
    for image in bad:
        block = builtin_registry().create(IMAGE)
        assert block.execute({"image": image}) is None
        assert block.error_kind == "invalid"
    source = ImageArray(np.zeros((4, 6)), "gray", {"spatial": frame([6, 4], "a")})
    mask = ImageArray(np.ones((4, 6), np.uint8), "gray", {"spatial": frame([6, 4], "b")})
    block = builtin_registry().create(IMAGE)
    assert block.execute({"image": source, "mask": mask}) is None
    assert "Selection coordinate frame" in block.error


def test_graphics_off_skips_diagnostic_rendering(monkeypatch):
    import opencv_dataflow.fourier_components as implementation

    def unexpected(*args, **kwargs):
        raise AssertionError("Unexpected graphics work")

    monkeypatch.setattr(implementation, "graphic_object", unexpected)
    monkeypatch.setattr(implementation, "magnitude_graphics", unexpected)
    block = builtin_registry().create(IMAGE)
    block.set_graphics_mode("off")
    result = block.execute({"image": np.eye(8)})
    assert result is not None, block.error
    assert not result.graphics and not result.drawings and result.outputs["preview"] is NO_VALUE
    forward = builtin_registry().create("cv.dft").execute({"image": np.eye(8)})
    selector = builtin_registry().create(COMPONENTS)
    selector.set_graphics_mode("off")
    assert selector.execute({"spectrum": packet_for(forward, "spectrum")}) is not None


def test_xml_preserves_all_component_fields(tmp_path):
    graph, registry = Graph(), builtin_registry()
    try:
        for kind in (IMAGE, COMPONENTS):
            block = graph.add(
                registry.create(kind, parameters={"components": "magnitude_only", "dc": "remove"})
            )
            block.documentation = "<p>Fourier component setup.</p>"
        expected = graph.to_dict()
        path = tmp_path / "components.xml"
        export_project_xml(expected, path)
        restored = Graph.from_dict(read_project_xml(path), registry)
        try:
            assert restored.to_dict() == expected
        finally:
            restored.close()
    finally:
        graph.close()


@pytest.mark.local_examples
def test_public_example_matches_analytic_impulse_truth():
    graph = Graph.load(PROJECT.with_name("verification.json"), builtin_registry())
    try:
        graph.run()
        assert all(block.state == "pass" for block in graph.processing_blocks())
        for name in ("phase", "magnitude", "full"):
            assert graph.blocks[f"verify_{name}"].last_result.outputs["value"] < 1e-12
    finally:
        graph.close()
    for project in (PROJECT, PROJECT.with_name("spectrum_pipeline.json")):
        graph = Graph.load(project, builtin_registry())
        try:
            graph.run()
            assert all(block.state == "pass" for block in graph.processing_blocks())
        finally:
            graph.close()
