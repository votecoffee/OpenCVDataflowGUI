"""Independent contrast arithmetic, selection validity and inspection decisions."""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.region_contrast import MeasureRegionContrast
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.timing import advance_timing, mark
from opencv_dataflow.toolbox import block_entry


def target(signal=120, background=100, deviation=2, depth=np.float64):
    return np.array(
        [
            [signal - deviation, signal + deviation] * 2
            + [background - deviation, background + deviation] * 2
        ],
        depth,
    )


def masks():
    signal = np.array([[255] * 4 + [0] * 4], np.uint8)
    return {"signal_mask": signal, "background_mask": 255 - signal}


def run(parameters=None, image=None, **inputs):
    block = MeasureRegionContrast(parameters=parameters)
    result = block.execute({"image": target() if image is None else image, **masks(), **inputs})
    assert result is not None, block.error
    return block, result


@pytest.mark.parametrize("deviation", ["population", "sample"])
@pytest.mark.parametrize("noise", ["quadrature", "background"])
def test_independent_statistics_and_explicit_noise_definitions(deviation, noise):
    _, result = run({"deviation": deviation, "noise_model": noise})
    sigma = 2 * (math.sqrt(4 / 3) if deviation == "sample" else 1)
    denominator = sigma * (math.sqrt(2) if noise == "quadrature" else 1)
    row = result.outputs["measurements"]["channels"][0]
    assert row["signal_mean"] == 120 and row["background_mean"] == 100
    assert row["signal_stddev"] == pytest.approx(sigma)
    assert row["background_stddev"] == pytest.approx(sigma)
    assert result.outputs["value"] == pytest.approx(20 / denominator)
    assert row["weber"] == pytest.approx(0.2) and row["michelson"] == pytest.approx(20 / 220)
    assert result.outputs["signal_count"] == result.outputs["background_count"] == 4
    assert result.passed and result.score is None


@pytest.mark.parametrize("metric", ["cnr", "absolute_difference", "weber", "michelson"])
def test_dark_signal_reports_signed_contrast_and_positive_decision_magnitude(metric):
    _, result = run(
        {"metric": metric, "check_threshold": True, "minimum_value": 0, "polarity": "dark"},
        image=target(signal=80),
    )
    expected = {
        "cnr": 20 / math.sqrt(8),
        "absolute_difference": 20,
        "weber": 0.2,
        "michelson": 20 / 180,
    }
    assert result.outputs["value"] == pytest.approx(expected[metric])
    assert result.outputs["difference"] == -20
    assert result.outputs["weber"] == pytest.approx(-0.2)
    assert result.outputs["michelson"] == pytest.approx(-20 / 180)
    assert result.passed and result.score == 1


def test_limit_and_polarity_failures_keep_real_measurements_and_reasons():
    _, result = run({"check_threshold": True, "minimum_value": 8, "polarity": "dark"})
    assert result.outputs["valid"] and not result.passed
    assert result.outputs["value"] == pytest.approx(20 / math.sqrt(8))
    assert {reason["code"] for reason in result.fail_reasons} == {
        "contrast_below_minimum",
        "contrast_wrong_polarity",
    }
    assert result.score == 0
    _, report_only = run({"check_threshold": False, "minimum_value": 100, "polarity": "dark"})
    assert report_only.passed and report_only.score is None


def test_zero_contrast_and_numeric_zero_limit_emit_real_zero_without_division():
    block, result = run({"check_threshold": True, "minimum_value": "0"}, image=target(signal=100))
    assert result.passed and result.score == 1
    assert result.outputs["value"] == 0 and result.outputs["valid"]
    assert block.current_emissions["value"][0].value == 0


def test_zero_noise_is_undefined_until_an_explicit_raw_unit_floor_is_supplied():
    image = target(deviation=0)
    block, result = run(image=image)
    assert not result.passed and not result.outputs["valid"]
    assert result.outputs["value"] is result.outputs["cnr"] is NO_VALUE
    assert result.outputs["difference"] == 20 and result.outputs["signal_mean"] == 120
    row = result.outputs["measurements"]["channels"][0]
    assert row["undefined"]["cnr"] == "zero_noise"
    assert not block.current_emissions.get("value")
    _, fixed = run({"noise_floor": 2}, image=image)
    assert fixed.passed and fixed.outputs["cnr"] == 10
    assert fixed.outputs["noise"] == 2
    assert fixed.outputs["measurements"]["channels"][0]["noise_floor_applied"]


def test_zero_background_and_signed_samples_only_invalidate_affected_ratios():
    image = np.array([[0, 4, 0, 4, 0, 0, 0, 0]], np.float64)
    _, result = run({"metric": "weber"}, image=image)
    assert result.outputs["value"] is NO_VALUE and not result.outputs["valid"]
    assert result.outputs["cnr"] == 1 and result.outputs["michelson"] == 1
    assert (
        result.outputs["measurements"]["channels"][0]["undefined"]["weber"]
        == "nonpositive_background"
    )
    _, signed = run(image=target(signal=-80, background=-100))
    assert signed.passed and signed.outputs["cnr"] == pytest.approx(20 / math.sqrt(8))
    assert signed.outputs["weber"] is signed.outputs["michelson"] is NO_VALUE


@pytest.mark.parametrize(
    "case,status",
    [
        ("missing_signal", "missing_signal_selection"),
        ("missing_background", "missing_background_selection"),
        ("empty", "empty_signal"),
        ("small", "too_few_signal_pixels"),
        ("overlap", "overlapping_selections"),
    ],
)
def test_invalid_selections_report_diagnostics_without_false_numeric_outputs(case, status):
    inputs = {"image": target(), **masks()}
    if case == "missing_signal":
        del inputs["signal_mask"]
    elif case == "missing_background":
        del inputs["background_mask"]
    elif case == "empty":
        inputs["signal_mask"] = np.zeros((1, 8), np.uint8)
    elif case == "small":
        inputs["signal_mask"] = np.array([[255] + [0] * 7], np.uint8)
    else:
        inputs["background_mask"] = inputs["signal_mask"].copy()
    block = MeasureRegionContrast()
    result = block.execute(inputs)
    assert result is not None and not result.passed, block.error
    assert result.outputs["measurements"]["status"] == status
    assert not result.outputs["valid"]
    for name in ("value", "cnr", "difference", "signal_mean", "background_stddev"):
        assert result.outputs[name] is NO_VALUE and not block.current_emissions.get(name)
    assert result.outputs["measurements"]["channels"] == []
    assert result.fail_reasons[0]["code"] == "contrast_" + status


@pytest.mark.parametrize("policy", ["allow", "exclude_signal"])
def test_explicit_overlap_policies_report_original_overlap_and_final_counts(policy):
    _, result = run({"overlap": policy}, background_mask=np.full((1, 8), 255, np.uint8))
    assert result.passed and result.outputs["measurements"]["overlap_count"] == 4
    assert result.outputs["background_count"] == (8 if policy == "allow" else 4)
    if policy == "exclude_signal":
        assert result.outputs["cnr"] == pytest.approx(20 / math.sqrt(8))


def test_complement_background_respects_work_mask_and_region():
    image = np.c_[target(), np.full((1, 4), 250)]
    signal = np.array([[255] * 4 + [0] * 8], np.uint8)
    work = np.array([[255] * 8 + [0] * 4], np.uint8)
    region = {
        "model": "image_region_v1",
        "image_size": [12, 1],
        "shape": "rectangle",
        "rectangle": [0, 0, 8, 1],
    }
    block = MeasureRegionContrast(parameters={"background_mode": "outside_signal"})
    result = block.execute({"image": image, "signal_mask": signal, "mask": work, "region": region})
    assert result is not None and result.passed, block.error
    assert result.outputs["cnr"] == pytest.approx(20 / math.sqrt(8))
    assert result.outputs["signal_count"] == result.outputs["background_count"] == 4
    assert not result.outputs["background_mask"][0, 8:].any()


@pytest.mark.parametrize("shape", ["rectangle", "circle", "polygon"])
def test_both_region_selectors_accept_editable_shapes_and_mask_intersection(shape):
    y, x = np.indices((32, 64))
    image = (100 + (x + y) % 2 * 4).astype(np.uint16)
    signal = {"model": "image_region_v1", "image_size": [64, 32], "shape": shape}
    if shape == "rectangle":
        signal["rectangle"] = [4, 6, 12, 12]
    elif shape == "circle":
        signal.update(center=[10, 12], radius=6)
    else:
        signal["vertices"] = [[4, 6], [16, 6], [10, 18]]
    background = {
        "model": "image_region_v1",
        "image_size": [64, 32],
        "shape": "rectangle",
        "rectangle": [40, 6, 12, 12],
    }
    signal_mask = np.full(image.shape, 255, np.uint8)
    signal_mask[:8] = 0
    block = MeasureRegionContrast()
    result = block.execute(
        {
            "image": image,
            "signal_region": signal,
            "signal_mask": signal_mask,
            "background_region": background,
        }
    )
    assert result is not None and result.passed, block.error
    assert not result.outputs["signal_mask"][:8].any()
    assert result.outputs["signal_count"] == np.count_nonzero(result.outputs["signal_mask"])
    assert result.outputs["background_count"] == 144


def test_native_rgb_channels_alpha_choice_and_unquantized_gray_measurements():
    red = target()
    green = target(signal=110)
    blue = target(signal=100)
    alpha = np.full((1, 8), 255, np.float64)
    source = ImageArray(np.stack([red, green, blue, alpha], axis=2), "rgba")
    _, native = run({"channel_mode": "native", "channel": "R"}, image=source)
    rows = native.outputs["measurements"]["channels"]
    assert [r["channel"] for r in rows] == ["R", "G", "B"]
    np.testing.assert_allclose([r["cnr"] for r in rows], [20 / math.sqrt(8), 10 / math.sqrt(8), 0])
    _, gray = run(image=source)
    assert gray.outputs["difference"] == pytest.approx(0.299 * 20 + 0.587 * 10)
    assert gray.outputs["cnr"] == pytest.approx((0.299 * 20 + 0.587 * 10) / math.sqrt(8))
    _, a = run(
        {
            "channel_mode": "native",
            "include_alpha": True,
            "channel": "A",
            "metric": "absolute_difference",
        },
        image=source,
    )
    assert len(a.outputs["measurements"]["channels"]) == 4 and a.outputs["value"] == 0
    np.testing.assert_array_equal(native.outputs["image"], source)


@pytest.mark.parametrize("depth", [np.uint8, np.uint16, np.int16, np.float32, np.float64])
def test_supported_depths_retain_original_pixels_and_sample_units(depth):
    image = target(depth=depth)
    original = image.copy()
    _, result = run(image=image)
    assert result.outputs["image"].dtype == depth
    np.testing.assert_array_equal(result.outputs["image"], original)
    np.testing.assert_array_equal(image, original)
    assert result.outputs["difference"] == 20


@pytest.mark.parametrize("factor", [1e200, 1e-200, 1e-300])
def test_scaled_noise_remains_finite_and_cnr_is_scale_invariant(factor):
    _, result = run(image=target() * factor)
    assert result.passed
    assert result.outputs["cnr"] == pytest.approx(20 / math.sqrt(8), rel=1e-12)
    assert result.outputs["signal_stddev"] == pytest.approx(2 * factor, rel=1e-12, abs=0)
    json.dumps(result.outputs["measurements"], allow_nan=False)


def test_centering_preserves_low_contrast_on_a_large_dc_offset():
    signal = np.array([0, 0.25, 0.5, 0.75])
    background = signal - 1
    image = 1e12 + np.r_[signal, background][None]
    _, result = run(image=image)
    expected = 1 / math.sqrt(2 * np.var(signal))
    assert result.outputs["difference"] == 1
    assert result.outputs["cnr"] == pytest.approx(expected, rel=1e-13)


def test_valid_cnr_can_coexist_with_unrepresentable_raw_difference():
    image = np.array([[0.8e308, 1e308] * 2 + [-1e308, -0.8e308] * 2])
    _, result = run(image=image)
    assert result.passed and result.outputs["valid"]
    assert result.outputs["cnr"] == pytest.approx(18 / math.sqrt(2))
    assert result.outputs["difference"] is NO_VALUE
    assert (
        result.outputs["measurements"]["channels"][0]["undefined"]["difference"]
        == "unrepresentable_value"
    )
    json.dumps(result.outputs["measurements"], allow_nan=False)


def test_large_floor_with_tiny_images_does_not_overflow_normalized_denominator():
    _, result = run({"noise_floor": 1e12}, image=target() * 1e-300)
    assert result.passed
    assert result.outputs["cnr"] == pytest.approx(2e-311, rel=1e-12, abs=0)
    assert result.outputs["noise"] == 1e12


def test_sample_noise_can_be_unrepresentable_while_cnr_remains_finite():
    image = np.array([[1.6e308] * 4 + [-1.6e308, 1.6e308] * 2])
    _, result = run({"deviation": "sample"}, image=image)
    assert result.passed
    expected = math.sqrt(3) / 2
    assert result.outputs["cnr"] == pytest.approx(expected)
    assert result.outputs["background_stddev"] is result.outputs["noise"] is NO_VALUE


def test_opposite_hdr_color_channels_use_a_finite_gray_fallback():
    sample = np.array([1.2e308] * 4 + [1.5e308] * 4)
    noise = np.array([-1e306, 1e306] * 4)
    pixels = np.stack([sample + noise, -sample + noise, sample + noise], axis=1)[None]
    _, result = run(image=ImageArray(pixels, "rgb"))
    assert result.passed
    expected = (0.587 - 0.299 - 0.114) * 0.3e308 / math.hypot(1e306, 1e306)
    assert result.outputs["cnr"] == pytest.approx(expected, rel=1e-12)


def test_current_ticket_timing_frame_pyramid_and_companion_data_survive():
    timing = advance_timing(None, mark(), mark(), source=True)
    context = {
        "ticket": TicketClock().issue("camera"),
        "spatial": frame([8, 1], "camera-level"),
        "pyramid": {
            "level": 1,
            "largest_size": [16, 2],
            "to_largest": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        },
        "encoder": 123,
        "processing_time": timing,
        "alpha_representation": "straight",
    }
    _, result = run(image=ValuePacket(target(), context))
    for name in ("image", "value", "measurements", "signal_mask", "background_mask", "difference"):
        payload = result.payloads[name]
        for key in ("ticket", "spatial", "pyramid", "encoder"):
            assert payload[key] == context[key]
        assert payload["processing_time"]["started_at"] == timing["started_at"]
    assert "alpha_representation" not in result.payloads["signal_mask"]


def test_wrong_selection_frames_fail_before_measurements():
    source = ValuePacket(target(), {"spatial": frame([8, 1], "camera")})
    signal = ImageArray(masks()["signal_mask"], "gray", {"spatial": frame([8, 1], "other")})
    block = MeasureRegionContrast()
    assert block.execute({"image": source, **masks(), "signal_mask": signal}) is None
    assert "coordinate frame differs" in block.error and not block.current_emissions


def test_many_selection_outlines_are_bounded_and_graphics_off_skips_contours(monkeypatch):
    image = np.full((128, 128), 100, np.uint8)
    signal = np.zeros_like(image)
    signal[::4, ::4] = 255
    image[signal != 0] = 80
    block = MeasureRegionContrast(
        parameters={"background_mode": "outside_signal", "noise_floor": 1}
    )
    result = block.execute({"image": image, "signal_mask": signal})
    assert result and result.passed, block.error
    assert result.outputs["measurements"]["graphic_outlines_limited"]
    assert len(result.drawings["signal"].objects) <= 64

    def unexpected(*args, **kwargs):
        raise AssertionError("Graphics disabled")

    monkeypatch.setattr(cv2, "findContours", unexpected)
    block.set_graphics_mode("off")
    again = block.execute({"image": image, "signal_mask": signal})
    assert again and again.passed and not again.drawings, block.error
    assert again.outputs["value"] == result.outputs["value"]


@pytest.mark.parametrize(
    "parameters,inputs",
    [
        ({"noise_floor": -1}, {}),
        ({"minimum_pixels": 0}, {}),
        ({"minimum_value": float("inf")}, {}),
        ({"channel": "R"}, {}),
        ({"background_mode": "outside_signal"}, {}),
        ({}, {"signal_mask": np.zeros((1, 8), np.uint16)}),
    ],
)
def test_bad_controls_or_selection_types_fail_safely(parameters, inputs):
    block = MeasureRegionContrast(parameters=parameters)
    assert block.execute({"image": target(), **masks(), **inputs}) is None
    assert block.error and not block.current_emissions


def test_xml_preserves_parameters_documentation_layout_and_exposed_ports(tmp_path):
    graph = Graph()
    block = graph.add(MeasureRegionContrast(parameters={"noise_floor": 1, "deviation": "sample"}))
    block.position = [230, 70]
    block.node_size = "large"
    block.description = "Contrast check"
    block.documentation = "<p>Use known signal and background selections.</p>"
    entry = block_entry(block)
    assert entry.path == ("Custom", "Inspection", "Visibility")
    assert {"CNR", "Weber", "Michelson"} <= set(entry.keywords)
    block.exposed_outputs += ["difference", "cnr", "signal_count"]
    try:
        expected = graph.to_dict()
        export_project_xml(expected, tmp_path / "contrast.xml")
        restored = Graph.from_dict(read_project_xml(tmp_path / "contrast.xml"), builtin_registry())
        try:
            assert restored.to_dict() == expected
        finally:
            restored.close()
    finally:
        graph.close()


@pytest.mark.local_examples
def test_public_strong_weak_and_invalid_selection_projects_match_analytic_truth():
    folder = Path(__file__).parents[1] / "examples/custom tools/region_contrast"
    truth = json.loads((folder / "assets/truth.json").read_text(encoding="utf-8"))
    graph = Graph.load(folder / "project.json", builtin_registry())
    try:
        for name in ("strong", "weak"):
            graph.run()
            result = graph.blocks["contrast"].last_result
            assert result is not None, graph.blocks["contrast"].error
            assert result.passed == truth[name]["passed"]
            assert result.outputs["cnr"] == pytest.approx(truth[name]["cnr"], rel=1e-13)
            rows = result.outputs["measurements"]["channels"]
            np.testing.assert_allclose(
                [r["cnr"] for r in rows], truth[name]["channel_cnr"], rtol=1e-13, atol=0
            )
            assert result.payloads["value"]["ticket"]["value"] == name + ".png"
            assert result.outputs["signal_count"] == result.outputs["background_count"] == 1280
    finally:
        graph.close()
    invalid = Graph.load(folder / "invalid_selection.json", builtin_registry())
    try:
        invalid.run()
        result = invalid.blocks["contrast"].last_result
        assert result and not result.passed and not result.outputs["valid"]
        assert result.outputs["measurements"]["status"] == "overlapping_selections"
        assert result.outputs["value"] is NO_VALUE
    finally:
        invalid.close()
