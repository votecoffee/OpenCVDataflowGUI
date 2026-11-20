"""Check native speckle connectivity, sample units and saved-project wiring."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.disparity_speckles import FilterDisparitySpeckles
from opencv_dataflow.images import DepthArray
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock

EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples/opencv tools/disparity_speckles/disparity_speckles.json"
)


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_example_preserves_feature_boundaries_and_removes_only_three_islands(profile):
    graph = Graph.load(EXAMPLE, builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        expected_removed = np.zeros((80, 160), dtype=bool)
        expected_removed[15:17, 20:23] = True
        expected_removed[50:54, 52:56] = True
        expected_removed[68:71, 140:144] = True
        expected_kept = ~expected_removed
        expected_kept[:, :4] = False
        expected_kept[26:33, 120:129] = False
        for name, dtype in (("fixed_point", np.uint16), ("float", np.float32)):
            src = graph.blocks[f"source_{name}"].last_result.outputs["image"]
            assert src.dtype == dtype
            block = graph.blocks[name]
            result = block.last_result
            assert result and result.passed, block.error
            outputs = result.outputs
            np.testing.assert_array_equal(outputs["removed_mask"], expected_removed * 255)
            np.testing.assert_array_equal(outputs["valid_mask"], expected_kept * 255)
            pixels = src / 16 if name == "fixed_point" else src
            disparity = outputs["disparity"]
            assert disparity.dtype == np.float32
            assert np.isnan(disparity[~expected_kept]).all()
            np.testing.assert_array_equal(disparity[expected_kept], pixels[expected_kept])
            assert np.isfinite(disparity[20:26, 65:75]).all()
            report = outputs["report"]
            assert (report["input_valid"], report["output_valid"], report["removed"]) == (
                12417,
                12383,
                34,
            )
    finally:
        graph.close()


def test_signed_fixed_point_preserves_zero_and_packet_metadata():
    source = np.array(
        [[-32768, -32765, -16, -32, -32], [0, -16, 0, -32, -32], [0, 0, 0, -32, -32]],
        dtype=np.int16,
    )
    support = np.full(source.shape, 255, np.uint8)
    support[1, 2] = 0
    spatial = frame(source.shape[::-1], "rectified-left")
    ticket = TicketClock("speckle-test").issue("stereo", "fixed", "piece-1")
    block = FilterDisparitySpeckles(
        parameters={
            "fixed_point_scale": 32,
            "max_speckle_size": 2,
            "max_difference_px": 0.1,
            "invalid_value_enabled": True,
            "invalid_value": -16,
        }
    )
    disparity = ValuePacket(source, {"spatial": spatial, "ticket": ticket})
    result = block.execute(
        {"disparity": disparity, "valid_mask": ValuePacket(support, {"spatial": spatial})}
    )
    assert result and result.passed, block.error
    outputs = result.outputs
    assert outputs["report"]["effective_difference_px"] == 3 / 32
    assert outputs["report"]["input_valid"] == 12
    assert outputs["report"]["removed"] == 2
    np.testing.assert_array_equal(outputs["removed_mask"][0], [255, 255, 0, 0, 0])
    assert outputs["disparity"][1, 0] == 0
    assert outputs["disparity"][1, -1] == -1
    assert np.isnan(outputs["disparity"][1, 2])
    for name in ("disparity", "valid_mask", "removed_mask"):
        assert result.payloads[name]["spatial"] == spatial
        assert result.payloads[name]["ticket"] == ticket
    assert source[0, 0] == -32768

    wrong_frame = ValuePacket(support, {"spatial": frame(source.shape[::-1], "rectified-right")})
    assert block.execute({"disparity": disparity, "valid_mask": wrong_frame}) is None
    assert "coordinate frame" in block.error


def test_diagonal_contacts_stay_separate_and_neighbor_threshold_is_inclusive():
    source = np.array([[10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 12, 13]], dtype=np.int16)
    block = FilterDisparitySpeckles(
        parameters={"fixed_point_scale": 1, "max_speckle_size": 1, "max_difference_px": 1}
    )
    result = block.execute({"disparity": source})
    assert result and result.passed, block.error
    expected = np.zeros(source.shape, np.uint8)
    expected[0, 0] = expected[1, 1] = 255
    np.testing.assert_array_equal(result.outputs["removed_mask"], expected)
    np.testing.assert_array_equal(result.outputs["disparity"][2, 2:], [12, 13])


def test_zero_size_disables_filter_but_normalizes_invalid_float_samples(monkeypatch):
    def unexpected(*args):
        pytest.fail("Bypass must skip native filtering")

    monkeypatch.setattr(cv2, "filterSpeckles", unexpected)
    source = np.array([[np.nan, np.inf, -700.0], [0.0, -0.1279, 2.123456789]], dtype=np.float64)
    block = FilterDisparitySpeckles(
        parameters={"max_speckle_size": 0, "invalid_value_enabled": True, "invalid_value": -700}
    )
    result = block.execute({"disparity": source})
    assert result and result.passed, block.error
    outputs = result.outputs
    assert outputs["disparity"].dtype == np.float64
    assert np.isnan(outputs["disparity"][0]).all()
    np.testing.assert_array_equal(outputs["disparity"][1], source[1])
    assert not outputs["removed_mask"].any()
    assert outputs["report"]["input_valid"] == 3


def test_all_valid_pixels_removed_still_emits_masks_and_failure_reason():
    source = np.array([[np.nan, 3.25, 0.0]], dtype=np.float32)
    block = FilterDisparitySpeckles(parameters={"max_speckle_size": 1, "max_difference_px": 0})
    result = block.execute({"disparity": source})
    assert result and not result.passed
    assert result.fail_reasons[0]["code"] == "disparity_speckles_no_valid_pixels"
    assert np.isnan(result.outputs["disparity"]).all()
    assert not result.outputs["valid_mask"].any()
    np.testing.assert_array_equal(result.outputs["removed_mask"], [[0, 255, 255]])


def test_full_signed_span_uses_unused_code_and_all_connected_fast_path():
    source = np.array([[-32768, 32767, 32767]], dtype=np.int16)
    for delta, removed in ((1, 1), (65535, 0)):
        block = FilterDisparitySpeckles(
            parameters={"fixed_point_scale": 1, "max_speckle_size": 1, "max_difference_px": delta}
        )
        result = block.execute({"disparity": source})
        assert result and result.passed, block.error
        assert result.outputs["report"]["removed"] == removed


@pytest.mark.parametrize(
    ("source", "parameters", "error"),
    [
        (DepthArray(np.ones((2, 2), np.float32)), {}, "monochrome"),
        (np.zeros((2, 2), np.uint16), {"invalid_value": -1}, "representable"),
        (np.zeros((2, 2), np.float32), {"invalid_value": 1e100}, "representable"),
        (np.array([[1, 10000.0]], dtype=np.float64), {}, "selected quantization"),
        (np.zeros((1, 32769), np.int16), {}, "per axis"),
        (
            np.array([[-32768, 32767]], dtype=np.int16),
            {"fixed_point_scale": 1, "max_difference_px": 32768},
            "native int16 threshold",
        ),
    ],
)
def test_rejects_incompatible_inputs_and_unsafe_native_ranges(source, parameters, error):
    block = FilterDisparitySpeckles(parameters={"invalid_value_enabled": True, **parameters})
    assert block.execute({"disparity": source}) is None
    assert error in block.error
