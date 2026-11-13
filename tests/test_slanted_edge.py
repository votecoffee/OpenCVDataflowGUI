"""Independent Gaussian-edge references and selected measurement failure cases."""

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock


def edge(sigma=1.5, *, angle=5, invert=False, horizontal=False, dtype=np.float32):
    y, x = np.mgrid[:192, :256]
    slope = math.tan(math.radians(angle))
    distance = (x - 128 - (y - 96) * slope) / math.sqrt(1 + slope * slope)
    value = 0.1 + 0.4 * (1 + np.vectorize(math.erf)(distance / (math.sqrt(2) * sigma)))
    if invert:
        value = 1 - value
    if horizontal:
        value = value.T
    if np.dtype(dtype).kind == "u":
        value = np.rint(value * np.iinfo(dtype).max)
    return value.astype(dtype)


def measure(image, **parameters):
    block = builtin_registry().create("inspect.slanted_edge", parameters=parameters)
    result = block.execute({"image": image})
    assert result is not None, block.error
    return block, result, result.outputs["measurements"]


@pytest.mark.parametrize(
    "sigma,window,radius", [(0.8, "hann", 16), (1.5, "hamming", 24), (3, "none", 32)]
)
def test_gaussian_edges_match_independent_frequency_and_width_references(sigma, window, radius):
    _, result, report = measure(
        edge(sigma, angle=-5 if sigma == 3 else 5), window=window, half_width=radius
    )
    assert result.passed
    expected50 = math.sqrt(math.log(2)) / (math.sqrt(2) * math.pi * sigma)
    expected10 = math.sqrt(math.log(10)) / (math.sqrt(2) * math.pi * sigma)
    assert report["mtf50_cycles_per_pixel"] == pytest.approx(expected50, rel=0.035)
    assert report["edge_width_10_90_pixels"] == pytest.approx(2.5631031311 * sigma, rel=0.025)
    if expected10 < 0.5:
        assert report["mtf10_cycles_per_pixel"] == pytest.approx(expected10, rel=0.04)
    else:
        assert report["mtf10_status"] == "above_nyquist"
        assert result.outputs["mtf10"] is NO_VALUE
    frequency, response = report["sfr"]["cycles_per_pixel"], report["sfr"]["response"]
    assert frequency[0] == 0 and frequency[-1] == 0.5
    assert response[0] == pytest.approx(1)
    assert np.interp(0.1, frequency, response) == pytest.approx(
        math.exp(-2 * math.pi**2 * sigma**2 * 0.1**2), rel=0.035
    )


@pytest.mark.parametrize(
    "dtype,horizontal,invert",
    [
        (np.uint8, False, False),
        (np.uint8, True, True),
        (np.uint16, True, False),
        (np.uint16, False, True),
        (np.float32, False, True),
        (np.float32, True, False),
    ],
)
def test_orientation_polarity_and_depth(dtype, horizontal, invert):
    image = edge(horizontal=horizontal, invert=invert, dtype=dtype)
    _, result, report = measure(image, window="none")
    assert result.passed
    assert report["orientation"] == ("horizontal" if horizontal else "vertical")
    assert report["polarity"] == ("falling" if invert else "rising")
    assert report["slant_degrees"] == pytest.approx(5, abs=0.005)
    # Quantized 8-bit plateaus introduce additional low-contrast normalization error.
    assert report["mtf50"] == pytest.approx(0.187390625 / 1.5, rel=0.02)
    assert result.outputs["image"].dtype == dtype
    np.testing.assert_array_equal(image, result.outputs["image"])


def test_selection_preserves_color_image_ticket_and_frame_and_emits_graphics():
    gray = edge(dtype=np.uint16)
    color = ImageArray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), "rgb")
    color[:12] = 0  # Distracting/clipped content outside the selected inspection area.
    saved = color.copy()
    ticket = TicketClock().issue("fixture")
    spatial = frame([256, 192], "slanted-fixture")
    mask = np.zeros(gray.shape, np.uint8)
    mask[15:180, 35:230] = 255
    region = {
        "model": "image_region_v1",
        "image_size": [256, 192],
        "shape": "rectangle",
        "rectangle": [45, 25, 170, 140],
    }
    block = builtin_registry().create(
        "inspect.slanted_edge", parameters={"sample_pitch": 0.02, "length_unit": "mm"}
    )
    result = block.execute(
        {
            "image": ValuePacket(color, {"ticket": ticket, "spatial": spatial}),
            "mask": mask,
            "region": region,
        }
    )
    assert result is not None and result.passed, block.error
    report = result.outputs["measurements"]
    assert report["selected_rows"] == 140
    assert report["mtf50"] == pytest.approx(report["mtf50_cycles_per_pixel"] / 0.02)
    assert report["edge_width_10_90"] == pytest.approx(report["edge_width_10_90_pixels"] * 0.02)
    assert report["sfr"]["frequencies"][-1] == 25
    assert result.payloads["image"]["spatial"] == spatial
    assert result.payloads["mtf50"]["ticket"] == ticket
    assert result.outputs["image"].color_format == "rgb"
    np.testing.assert_array_equal(color, saved)
    assert result.drawings["slanted edge"].objects[-1].kind == "polyline"
    assert result.graphics["edge spread and SFR"].shape == (390, 760, 3)
    block.set_graphics_mode("off")
    lean = block.execute({"image": color, "mask": mask, "region": region})
    assert lean.passed and not lean.drawings and not lean.graphics
    mask[:, 127] = 0
    disconnected = block.execute({"image": color, "mask": mask, "region": region})
    assert not disconnected.passed
    assert disconnected.fail_reasons[0]["code"] == "edge_support"


def test_threshold_is_inclusive_and_failure_has_reason():
    block, result, report = measure(edge(), check_threshold=True, minimum_mtf50=0)
    block.parameters["minimum_mtf50"] = report["mtf50"]
    assert block.execute({"image": edge()}).passed
    block.parameters["minimum_mtf50"] = report["mtf50"] + 0.001
    failure = block.execute({"image": edge()})
    assert not failure.passed and failure.outputs["measurements"]["valid"]
    assert failure.fail_reasons[0]["code"] == "sharpness_below_minimum"


def test_above_nyquist_is_a_bound_and_cannot_prove_an_arbitrary_threshold():
    block, result, report = measure(edge(0.25), check_threshold=True, minimum_mtf50=0.45)
    assert result.passed
    assert result.outputs["mtf50"] is NO_VALUE
    assert report["mtf50_status"] == "above_nyquist"
    block.parameters["minimum_mtf50"] = 0.6
    failed = block.execute({"image": edge(0.25)})
    assert not failed.passed
    assert failed.fail_reasons[0]["code"] == "sharpness_beyond_measurement_band"


def test_derivative_correction_and_gamma_controls():
    image = edge(1)
    _, _, corrected = measure(image, window="none")
    _, _, raw = measure(image, window="none", derivative_correction=False)
    f = np.asarray(raw["sfr"]["cycles_per_pixel"])
    np.testing.assert_allclose(
        corrected["sfr"]["response"], np.asarray(raw["sfr"]["response"]) / np.sinc(2 * f / 4)
    )
    _, _, linearized = measure(image ** (1 / 2.2), window="none", gamma=2.2)
    assert linearized["mtf50"] == pytest.approx(corrected["mtf50"], rel=1e-4)


@pytest.mark.parametrize(
    "case,code",
    [
        ("flat", "edge_support"),
        ("aligned", "edge_angle"),
        ("clipped", "edge_clipping"),
        ("narrow", "edge_support"),
        ("curved", "edge_fit"),
        ("phases", "edge_sampling"),
        ("plateau", "edge_plateau"),
    ],
)
def test_unsuitable_edges_return_failed_reports_without_invented_metrics(case, code):
    image, parameters = edge(), {}
    if case == "flat":
        image.fill(0.3)
    elif case == "aligned":
        image = edge(angle=0)
    elif case == "clipped":
        image = np.clip((image - 0.1) / 0.8, 0, 1)
    elif case == "narrow":
        image = image[:, 115:140]
    elif case == "curved":
        columns = np.arange(image.shape[1])
        image = np.array(
            [
                np.interp(columns - 8 * ((y - 96) / 96) ** 2, columns, row)
                for y, row in enumerate(image)
            ]
        ).astype(np.float32)
    elif case == "phases":
        image = edge(angle=math.degrees(math.atan(0.5)))
        parameters = {"maximum_slant": 30.0, "oversampling": "8"}
    elif case == "plateau":
        image = edge(8)
    block, result, report = measure(image, **parameters)
    assert not result.passed and not report["valid"]
    assert result.fail_reasons[0]["code"] == code
    assert result.outputs["mtf50"] is NO_VALUE
    assert not block.error
    if case == "clipped":
        assert block.execute({"image": image}) is not None
        block.parameters["reject_clipping"] = False
        assert block.execute({"image": image}).passed


def test_invalid_values_and_frame_mismatch_use_validation_errors():
    block = builtin_registry().create("inspect.slanted_edge", parameters={"sample_pitch": 0})
    assert block.execute({"image": edge()}) is None
    assert "Sample pitch" in block.error
    block.parameters.update(sample_pitch=1, minimum_slant=10, maximum_slant=5)
    assert block.execute({"image": edge()}) is None
    assert "Minimum slant" in block.error
    block.parameters.update(minimum_slant=1, maximum_slant=15)
    assert block.execute({"image": edge() * 255}) is None
    assert "[0,1]" in block.error
    source = ValuePacket(edge(), {"spatial": frame([256, 192], "a")})
    mask = ValuePacket(np.ones((192, 256), np.uint8), {"spatial": frame([256, 192], "b")})
    assert block.execute({"image": source, "mask": mask}) is None
    assert "coordinate frame differs" in block.error


def test_example_sharp_and_blurred_images_have_expected_judgements():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples/custom tools/slanted_edge_sharpness/project.json"
    )
    graph = Graph.load(path, builtin_registry())
    try:
        measured = []
        for expected in (True, False):
            graph.run()
            result = graph.blocks["measure"].last_result
            assert result is not None and result.passed == expected
            measured.append(result.outputs["mtf50"])
        assert measured[0] > measured[1]
    finally:
        graph.close()
