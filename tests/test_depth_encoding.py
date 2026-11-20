"""Qualify native depth decoding and uint16 round-trip error."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.depth_encoding import ENCODING, NormalizeMetricDepth, QuantizeMetricDepth
from opencv_dataflow.images import DepthArray
from opencv_dataflow.payloads import ValuePacket, frame, packet_for
from opencv_dataflow.tickets import TicketClock

STEP = 1 / 1024
EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples/opencv tools/metric_depth_encoding/metric_depth_encoding.json"
)


@pytest.mark.parametrize(("invalid", "policy"), [(0, "clip"), (65535, "invalidate")])
def test_quantized_codes_and_tags_preserve_range_and_provenance(invalid, policy):
    cap = 65535 if invalid == 0 else 65534
    raw = np.array([[STEP / 2, 1.5 * STEP, 2.5 * STEP, cap * STEP, (cap + 2) * STEP, np.nan, -1]])
    config = {"invalid_code": invalid, "meters_per_code": STEP, "out_of_range": policy}
    quantizer = QuantizeMetricDepth(parameters=config)
    payload = {"spatial": frame([raw.shape[1], 1], "depth"), "ticket": TicketClock().issue("depth")}
    result = quantizer.execute({"depth": ValuePacket(raw, payload)})
    assert result and result.passed, quantizer.error
    clip = policy == "clip"
    expected = [[1 if clip else invalid, 2, 2, cap, cap if clip else invalid, invalid, invalid]]
    np.testing.assert_array_equal(result.outputs["codes"], expected)
    assert result.outputs["codes"].dtype == np.uint16
    assert result.outputs["report"]["out_of_range_samples"] == 2
    np.testing.assert_array_equal(result.outputs["out_of_range_mask"], [[255, 0, 0, 0, 255, 0, 0]])
    report = result.outputs["report"]
    assert report["in_range_error"]["max_absolute_m"] == STEP / 2
    assert report["in_range_error"]["rms_m"] == pytest.approx(STEP / np.sqrt(6))
    assert report["retained_error"]["max_absolute_m"] == (2 * STEP if clip else STEP / 2)
    assert report["clamped_samples"] == (2 if clip else 0)

    decoder = NormalizeMetricDepth(
        parameters={"output_units": "mm", "output_format": "float64", "meters_per_code": 0.2}
    )
    decoded = decoder.execute({"samples": packet_for(result, "codes")})
    assert decoded and decoded.passed, decoder.error
    depth = decoded.outputs["depth"]
    assert isinstance(depth, DepthArray) and depth.dtype == np.float64 and depth.units == "mm"
    valid = np.asarray(expected) != invalid
    np.testing.assert_allclose(depth[valid], np.asarray(expected)[valid] * STEP * 1000, atol=1e-11)
    assert np.isnan(depth[~valid]).all()
    for name in ("depth", "valid_mask", "report"):
        assert ENCODING not in decoded.payloads[name]
        assert decoded.payloads[name]["ticket"] == payload["ticket"]
        assert decoded.payloads[name]["spatial"] == payload["spatial"]
    assert ENCODING not in result.payloads["out_of_range_mask"]


def test_normalize_untagged_codes_and_nearby_float64_values():
    decoder = NormalizeMetricDepth(
        parameters={"invalid_code": 7, "meters_per_code": 0.0025, "output_format": "float64"}
    )
    raw = np.array([[0, 4, 7, 255, 65535]], np.uint16)
    result = decoder.execute({"samples": raw})
    assert result and result.passed, decoder.error
    np.testing.assert_allclose(result.outputs["depth"], [[np.nan, 0.01, np.nan, 0.6375, 163.8375]])
    assert result.outputs["report"]["invalid_input_samples"] == 2
    mm = DepthArray([[1000.0, 1000.000001]], units="mm", dtype=np.float64)
    decoded = decoder.execute({"samples": mm})
    assert decoded and decoded.passed
    values = decoded.outputs["depth"]
    np.testing.assert_allclose(values, [[1, 1.000000001]], atol=1e-14, rtol=0)
    legacy = DepthArray([[1000.0, 1000.000001]], units="mm")
    assert legacy.dtype == np.float32


def test_float_representability_and_nonfinite_are_separate():
    depth = np.array([[1, 1e100, 1e-100, np.inf, np.nan, -0.01]], np.float64)
    block = NormalizeMetricDepth()
    result = block.execute({"samples": depth})
    assert result and result.passed
    np.testing.assert_array_equal(result.outputs["valid_mask"], [[255, 0, 0, 0, 0, 0]])
    report = result.outputs["report"]
    assert report["valid_samples"] == 1
    assert report["invalid_input_samples"] == 3
    assert report["unrepresentable_samples"] == 2


@pytest.mark.parametrize(
    "tag",
    [
        None,
        {"model": "fake", "meters_per_code": 0.001, "invalid_code": 0},
        {"model": "linear_uint16_v1", "meters_per_code": -1.0, "invalid_code": 0},
        {"model": "linear_uint16_v1", "meters_per_code": 0.001, "invalid_code": 65536},
    ],
)
def test_bad_encoding_is_rejected(tag):
    block = NormalizeMetricDepth()
    result = block.execute({"samples": ValuePacket(np.ones((2, 3), np.uint16), {ENCODING: tag})})
    assert result is None and "encoding" in block.error


def test_no_positive_values_and_raw_codes_are_distinct_outcomes():
    depth = np.array([[0, np.nan]], np.float32)
    block = QuantizeMetricDepth()
    result = block.execute({"depth": depth})
    assert result is not None and not result.passed
    assert result.fail_reasons[0]["code"] == "no_encoded_depth"
    assert result.outputs["report"]["in_range_error"]["max_absolute_m"] is None
    assert block.execute({"depth": np.ones((1, 2), np.uint16)}) is None
    assert "Decode sensor codes first" in block.error
    decoded = NormalizeMetricDepth().execute({"samples": np.array([[0, 65535]], np.uint16)})
    assert decoded is not None and not decoded.passed
    assert decoded.fail_reasons[0]["code"] == "no_decoded_depth"


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_quantization_graph_round_trips_declared_bands(profile):
    graph = Graph.load(EXAMPLE, builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        encode, decode = graph.blocks["encode"], graph.blocks["decode"]
        assert encode.last_result and encode.last_result.passed, encode.error
        assert decode.last_result and decode.last_result.passed, decode.error
        assert encode.last_result.outputs["report"]["in_range_error"]["max_absolute_m"] == STEP / 2
        assert encode.last_result.outputs["report"]["clamped_samples"] == 1536
        assert decode.last_result.outputs["report"]["valid_samples"] == 3840
        codes = [65535, 1, 1024, 1536, 1538, 65534]
        np.testing.assert_array_equal(encode.last_result.outputs["codes"][0, ::16], codes)
        metric = decode.last_result.outputs["depth"][0, ::16]
        np.testing.assert_allclose(metric[1:], np.array(codes[1:]) * 1000 * STEP, atol=1e-10)
        assert np.isnan(metric[0])
        assert metric.dtype == np.float64
    finally:
        graph.close()
