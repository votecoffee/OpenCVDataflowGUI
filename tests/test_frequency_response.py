"""Independent transfer references, unsupported bins, selections and persistence."""

import copy
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest

from opencv_dataflow import frequency_response
from opencv_dataflow.block_xml import save_block_xml
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.project_xml import read_block_xml
from opencv_dataflow.run_profiles import profile
from opencv_dataflow.tickets import TicketClock


def run(sample, reference, **parameters):
    block = builtin_registry().create("inspect.frequency_response", parameters=parameters)
    result = block.execute({"image": sample, "reference": reference})
    assert result is not None, block.error
    return block, result


def rawmap(result, key):
    return np.fft.ifftshift(result.outputs[key])


@pytest.mark.parametrize("shape", [(31, 47), (32, 48)])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
def test_identity_preserves_type_and_exact_inclusive_gain_gate(shape, dtype):
    image = np.random.default_rng(166).integers(10, 200, shape).astype(dtype)
    original = image.copy()
    _, result = run(
        image,
        image,
        minimum_mean_gain=1,
        maximum_mean_gain=1,
        check_phase=True,
        maximum_phase_degrees=0,
    )
    assert result.passed
    valid = result.outputs["supported"] != 0
    np.testing.assert_allclose(result.outputs["gain"][valid], 1, atol=2e-14, rtol=0)
    assert result.outputs["mean_gain"] == pytest.approx(1, abs=1e-14)
    assert result.outputs["maximum_phase_degrees"] < 1e-12
    assert result.outputs["image"].dtype == dtype
    np.testing.assert_array_equal(result.outputs["image"], original)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("shape", [(35, 49), (48, 64)])
@pytest.mark.parametrize("operation", ["blur", "shift", "negative_gain"])
def test_transfer_matches_independent_numpy_and_analytic_model(shape, operation):
    reference = np.random.default_rng(17).normal(size=shape)
    fy, fx = np.fft.fftfreq(shape[0]), np.fft.fftfreq(shape[1])
    if operation == "blur":
        row = (np.roll(reference, -1, 1) + 2 * reference + np.roll(reference, 1, 1)) / 4
        sample = (np.roll(row, -1, 0) + 2 * row + np.roll(row, 1, 0)) / 4
        expected = ((1 + np.cos(2 * np.pi * fy)) / 2)[:, None] * ((1 + np.cos(2 * np.pi * fx)) / 2)[
            None, :
        ]
    elif operation == "shift":
        sample = np.roll(reference, (2, -3), axis=(0, 1))
        expected = np.exp(-2j * np.pi * (2 * fy[:, None] - 3 * fx[None, :]))
    else:
        sample, expected = -2.5 * reference, np.full(shape, -2.5, complex)
    _, result = run(sample, reference, maximum_frequency=float(np.sqrt(0.5)))
    valid, defined = rawmap(result, "supported") != 0, rawmap(result, "phase_valid") != 0
    r = np.fft.fft2(reference - reference.mean())
    independent = np.fft.fft2(sample - sample.mean()) / np.where(abs(r) > 0, r, 1)
    np.testing.assert_allclose(rawmap(result, "gain")[valid], abs(expected[valid]), atol=5e-12)
    np.testing.assert_allclose(rawmap(result, "gain")[valid], abs(independent[valid]), atol=5e-12)
    error = np.angle(np.exp(1j * (rawmap(result, "phase") - np.angle(expected))))
    # Near binomial zeros, tiny spatial/DFT roundoff becomes relative phase error.
    assert np.max(abs(error[defined])) < 1e-9
    rows = result.outputs["metrics"]["radial"]
    assert sum(row["supported_bins"] for row in rows) == valid.sum()
    mean = (
        sum(
            row["mean_gain"] * row["supported_bins"] for row in rows if row["mean_gain"] is not None
        )
        / valid.sum()
    )
    assert mean == pytest.approx(result.outputs["mean_gain"], abs=1e-12)


@pytest.mark.parametrize("window", ["none", "hann", "sqrt_hann"])
def test_constant_reference_has_no_response_even_with_window_roundoff(window):
    image = np.full((31, 47), 173, np.uint16)
    block, result = run(image, image, window=window)
    assert not result.passed and block.state == "fail"
    assert result.outputs["mean_gain"] is result.outputs["maximum_phase_degrees"] is NO_VALUE
    assert not result.outputs["supported"].any() and not result.outputs["phase_valid"].any()
    assert result.fail_reasons[0]["code"] == "frequency_reference_support_insufficient"
    assert not block.current_emissions["mean_gain"]
    assert all(row["mean_gain"] is None for row in result.outputs["metrics"]["radial"])


@pytest.mark.parametrize("direction,count", [("horizontal", 2), ("vertical", 0), ("angle", 2)])
def test_grating_marks_only_excited_antipodal_bins(direction, count):
    image = np.tile([10.0, 12.0, 10.0, 8.0], (32, 8))
    _, result = run(image, image, direction=direction, angular_tolerance_degrees=0)
    assert result.outputs["metrics"]["supported_bins"] == count
    assert result.passed == bool(count)
    if count:
        expected = np.zeros((32, 32), np.uint8)
        expected[0, [8, 24]] = 255
        np.testing.assert_array_equal(rawmap(result, "supported"), expected)


def test_zero_sample_has_valid_zero_gain_but_undefined_phase():
    reference = np.random.default_rng(3).normal(size=(32, 48))
    _, result = run(np.zeros_like(reference), reference)
    assert result.passed and result.outputs["mean_gain"] == 0
    assert result.outputs["maximum_phase_degrees"] is NO_VALUE
    assert not result.outputs["phase_valid"].any()
    _, checked = run(np.zeros_like(reference), reference, check_phase=True)
    assert (
        not checked.passed and checked.fail_reasons[0]["code"] == "frequency_phase_out_of_tolerance"
    )


@pytest.mark.parametrize("magnitude", [1e-100, 1e100])
def test_shared_scale_preserves_extreme_finite_ratio(magnitude):
    reference = np.random.default_rng(8).normal(size=(31, 47)) * magnitude
    _, result = run(1.75 * reference, reference)
    assert result.outputs["mean_gain"] == pytest.approx(1.75, abs=2e-13)


def test_absolute_floor_has_coherent_sample_units_and_dc_is_explicit():
    image = np.tile([0.0, 2.0, 0.0, -2.0], (32, 8))
    _, measured = run(image, image, absolute_floor=0.99)
    _, rejected = run(image, image, absolute_floor=1.01)
    assert measured.outputs["metrics"]["supported_bins"] == 2
    assert not rejected.passed and rejected.outputs["mean_gain"] is NO_VALUE
    _, dc = run(np.full((8, 8), 14.0), np.full((8, 8), 7.0), subtract_mean=False, include_dc=True)
    assert dc.passed and dc.outputs["metrics"]["supported_bins"] == 1
    assert dc.outputs["mean_gain"] == pytest.approx(2)


@pytest.mark.parametrize("window", ["none", "hann", "sqrt_hann"])
def test_shared_selection_suppresses_exterior_and_uses_exact_aperture(window):
    reference = np.random.default_rng(6).normal(size=(32, 48))
    sample = reference.copy()
    sample[:8] += 100
    mask = np.full((32, 48), 255, np.uint8)
    mask[:8] = 0
    block = builtin_registry().create("inspect.frequency_response", parameters={"window": window})
    result = block.execute({"image": sample, "reference": reference, "mask": mask})
    assert result is not None and result.passed, block.error
    assert result.outputs["mean_gain"] == pytest.approx(1, abs=2e-13)
    expected = (mask != 0).astype(float)
    if window != "none":
        native = cv2.createHanningWindow((48, 32), cv2.CV_64F)
        expected *= native**2 if window == "hann" else native
    np.testing.assert_array_equal(result.outputs["aperture"], expected)


@pytest.mark.parametrize("encoding", ["rgb", "bgr", "rgba", "bgra"])
def test_named_channels_use_semantic_color_order(encoding):
    rgb = np.random.default_rng(11).integers(10, 80, (32, 48, 3), np.uint16) * 16
    sample = rgb.copy()
    sample[..., 0] //= 2
    if encoding.startswith("bgr"):
        rgb, sample = rgb[..., ::-1], sample[..., ::-1]
    if encoding.endswith("a"):
        rgb, sample = (np.dstack((v, np.full((32, 48), 60000, np.uint16))) for v in (rgb, sample))
    a, b = ImageArray(sample, encoding), ImageArray(rgb, encoding)
    _, result = run(a, b, channel="R")
    assert result.outputs["mean_gain"] == pytest.approx(0.5)
    assert result.outputs["image"].color_format == encoding
    np.testing.assert_array_equal(result.outputs["image"], a)


def test_unsigned_unit_scale_compares_equivalent_u8_and_u16():
    reference = np.random.default_rng(5).integers(10, 200, (32, 48), np.uint8)
    sample = reference.astype(np.uint16) * 257
    _, raw = run(sample, reference)
    _, unit = run(sample, reference, unsigned_scale="unit_range")
    assert raw.outputs["mean_gain"] == pytest.approx(257)
    assert unit.outputs["mean_gain"] == pytest.approx(1)


def test_ignored_finite_exterior_cannot_overflow_the_selected_signal_scale():
    image = np.random.default_rng(7).normal(size=(32, 48)) * 1e-200
    image[:8] = 1e200
    mask = np.full((32, 48), 255, np.uint8)
    mask[:8] = 0
    block = builtin_registry().create("inspect.frequency_response")
    result = block.execute({"image": image, "reference": image, "mask": mask})
    assert result is not None and result.passed, block.error
    assert result.outputs["mean_gain"] == pytest.approx(1, abs=1e-13)


@pytest.mark.parametrize(
    "bad",
    ["grid", "ticket", "empty_selection", "mask_grid", "budget", "channel", "bounds", "format"],
)
def test_guards_reject_before_fft(bad, monkeypatch):
    image = ImageArray(np.random.default_rng(1).normal(size=(64, 96)), "gray")
    clock = TicketClock()
    contexts = [
        {"spatial": frame([96, 64], "camera"), "ticket": clock.issue("camera")} for _ in range(2)
    ]
    inputs = {
        "image": ValuePacket(image, contexts[0]),
        "reference": ValuePacket(image, contexts[1]),
    }
    p = {}
    if bad == "grid":
        p["reference_alignment"] = "same_frame"
        contexts[1]["spatial"]["to_root"][0][2] = 1
    elif bad == "ticket":
        p["ticket_policy"] = "same_ticket"
    elif bad == "empty_selection":
        inputs["mask"] = np.zeros((64, 96), np.uint8)
    elif bad == "mask_grid":
        inputs["mask"] = ValuePacket(
            np.ones((64, 96), np.uint8), {"spatial": frame([96, 64], "other")}
        )
    elif bad == "budget":
        p["maximum_workspace_mib"] = 1
    elif bad == "channel":
        p["channel"] = "R"
    elif bad == "bounds":
        p.update(minimum_frequency=0.4, maximum_frequency=0.1)
    else:
        inputs["image"] = inputs["reference"] = ImageArray(np.ones((64, 96, 3), np.uint8), "hsv")
    monkeypatch.setattr(frequency_response.cv2, "dft", lambda *a, **k: pytest.fail("Invalid FFT"))
    block = builtin_registry().create("inspect.frequency_response", parameters=p)
    assert block.execute(inputs) is None and block.error_kind == "invalid"
    assert not any(block.current_emissions.values())


def test_frequency_grid_is_separate_and_both_source_tickets_are_retained():
    image = np.random.default_rng(10).normal(size=(32, 48))
    context = {
        "ticket": TicketClock().issue("camera"),
        "spatial": frame([48, 32], "camera"),
        "pyramid": {
            "level": 1,
            "largest_size": [96, 64],
            "to_largest": [[2, 0, 0.5], [0, 2, 0.5], [0, 0, 1]],
        },
        "custom": {"part": "a"},
    }
    original = copy.deepcopy(context)
    block = builtin_registry().create(
        "inspect.frequency_response",
        parameters={"reference_alignment": "same_frame", "ticket_policy": "same_ticket"},
    )
    result = block.execute(
        {"image": ValuePacket(image, context), "reference": ValuePacket(image, context)}
    )
    assert result is not None and result.passed, block.error
    assert context == original
    for name in ("image", "aperture"):
        assert result.payloads[name]["spatial"] == context["spatial"]
        assert result.payloads[name]["pyramid"] == context["pyramid"]
    for name in ("gain", "phase", "supported", "phase_valid"):
        payload = result.payloads[name]
        assert payload["ticket"] == context["ticket"] and payload["custom"] == context["custom"]
        assert payload["spatial"]["root_id"] == block.id + ":frequency"
        assert (
            "pyramid" not in payload
            and payload["spectral_source_grid"]["pyramid"] == context["pyramid"]
        )
    assert all(
        c["ticket"] == context["ticket"] for c in result.outputs["metrics"]["source_contexts"]
    )


def test_native_exception_is_contained_and_recovers(monkeypatch):
    image = np.random.default_rng(1).normal(size=(32, 48))
    native = cv2.divSpectrums
    block = builtin_registry().create("inspect.frequency_response")
    assert block.execute({"image": image, "reference": image}).passed

    def broken(*a, **k):
        raise cv2.error("Controlled spectrum division failure")

    monkeypatch.setattr(frequency_response.cv2, "divSpectrums", broken)
    assert (
        block.execute({"image": image, "reference": image}) is None
        and block.error_kind == "exception"
    )
    assert not any(block.current_emissions.values())
    monkeypatch.setattr(frequency_response.cv2, "divSpectrums", native)
    assert block.execute({"image": image, "reference": image}).passed


def test_parallel_instances_and_lean_outputs():
    reference = np.random.default_rng(4).normal(size=(32, 48))

    def evaluate(gain):
        block = builtin_registry().create("inspect.frequency_response")
        block.run_profile = profile("lean")
        result = block.execute({"image": reference * gain, "reference": reference})
        assert result is not None and not result.graphics and not result.drawings
        return result.outputs["mean_gain"]

    with ThreadPoolExecutor(max_workers=3) as pool:
        np.testing.assert_allclose(list(pool.map(evaluate, [0.5, 1.0, 1.5])), [0.5, 1, 1.5])


def test_xml_reopen_and_offscreen_dropdowns(app, tmp_path):
    from PySide6.QtWidgets import QComboBox

    from opencv_dataflow.app import MainWindow
    from opencv_dataflow.opencv_reference import opencv_reference

    registry, graph = builtin_registry(), Graph()
    block = graph.add(
        registry.create(
            "inspect.frequency_response",
            parameters={"window": "hann", "direction": "angle", "angle_degrees": 30},
        )
    )
    xml = tmp_path / "response.inspect.frequency_response.xml"
    save_block_xml(block, xml)
    assert read_block_xml(xml)["parameters"] == block.parameters
    graph.save(tmp_path / "program.json")
    restored = Graph.load(tmp_path / "program.json", registry)
    assert restored.blocks[block.id].parameters == block.parameters
    restored.close()
    assert "divSpectrums" in opencv_reference(block)
    window = MainWindow()
    try:
        window.replace_graph(graph)
        window.inspector.select(block)
        for name in (
            "channel",
            "window",
            "direction",
            "unsigned_scale",
            "reference_alignment",
            "ticket_policy",
        ):
            assert isinstance(window.inspector.fields[name], QComboBox)
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
        graph.close()
