"""Qualify clean-depth rejection without changing accepted measurements."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import DepthArray
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock

PROJECT = Path(__file__).resolve().parents[1] / "examples/custom tools/tof_depth_cleanup"


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_recorded_samples_keep_the_step_thin_feature_and_capture_metadata(profile):
    graph = Graph.load(PROJECT / "tof_depth_cleanup.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        for index in range(3):
            graph.run()
            replay, clean, cloud = (graph.blocks[key] for key in ("replay", "clean", "cloud"))
            assert all(
                block.last_result and block.last_result.passed for block in (replay, clean, cloud)
            )
            raw = replay.last_result.outputs["depth"]
            result = clean.last_result
            outputs = result.outputs
            labels = cv2.imread(str(PROJECT / f"assets/reasons_{index}.png"), cv2.IMREAD_UNCHANGED)
            truth = cv2.imread(str(PROJECT / f"assets/truth_{index}.png"), cv2.IMREAD_UNCHANGED)
            keep = labels == 0
            np.testing.assert_array_equal(outputs["reason_codes"], labels)
            np.testing.assert_array_equal(outputs["valid_mask"] != 0, keep)
            np.testing.assert_array_equal(
                outputs["rejected_mask"] != 0, (labels > 0) & (labels < 4)
            )
            np.testing.assert_array_equal(outputs["raw_depth"], raw)
            np.testing.assert_array_equal(outputs["depth"][keep], raw[keep])
            np.testing.assert_allclose(outputs["depth"][keep], truth[keep] / 1000, atol=2e-7)
            assert np.isnan(outputs["depth"][~keep]).all()
            assert outputs["report"]["reason_counts"] == {
                "low_confidence": 20,
                "flying_pixel": 5,
                "isolated_outlier": 3,
                "invalid_input": 6,
            }
            assert outputs["report"]["retained_samples"] == 5342
            assert bool(result.graphics) == (profile == "debug")
            payload = replay.last_result.payloads["depth"]
            assert payload["recorded_rgbd"]["pair_index"] == index
            for name in ("depth", "raw_depth", "reason_codes", "valid_mask", "rejected_mask"):
                for key in ("ticket", "spatial", "optical_frame", "recorded_rgbd"):
                    assert result.payloads[name][key] == payload[key]
            points = cloud.last_result.outputs["cloud"]
            x, y = points.pixels.T
            assert len(points.points) == keep.sum() and keep[y, x].all()
            np.testing.assert_allclose(points.points[:, 2], outputs["depth"][y, x], atol=2e-7)
        assert graph.sources_exhausted()
    finally:
        graph.close()


def test_mask_and_region_limit_rejection_and_uint16_scale_is_explicit():
    depth = np.full((7, 7), 1000, np.uint16)
    depth[3, 3] = depth[0, 0] = depth[6, 6] = 1600
    mask = np.full(depth.shape, 255, np.uint8)
    mask[0, 0] = 0
    region = {
        "model": "image_region_v1",
        "image_size": [7, 7],
        "shape": "rectangle",
        "rectangle": [0, 0, 5, 5],
    }
    box = builtin_registry().create(
        "3d.clean_tof_depth", parameters={"input_units": "mm", "units_per_sample": 1}
    )
    result = box.execute({"depth": depth, "mask": mask, "region": region})
    assert result and result.passed, box.error
    expected = depth.copy()
    expected[3, 3] = 0
    assert result.outputs["depth"].dtype == np.uint16
    np.testing.assert_array_equal(result.outputs["depth"], expected)
    np.testing.assert_array_equal(result.outputs["raw_depth"], depth)
    assert depth[3, 3] == 1600
    assert result.outputs["reason_codes"][3, 3] == 3
    assert result.outputs["report"]["meters_per_sample"] == 0.001


@pytest.mark.parametrize("mismatch", ["ticket", "spatial", "recorded_rgbd"])
def test_incompatible_confidence_is_a_validation_error(mismatch):
    clock = TicketClock()
    payload = {
        "ticket": clock.issue("camera"),
        "spatial": frame([3, 3], "camera"),
        "recorded_rgbd": {"clock": "device", "timestamp_ns": 100, "pair_index": 0},
    }
    other = {
        "ticket": clock.issue("camera"),
        "spatial": frame([3, 3], "other-camera"),
        "recorded_rgbd": {"clock": "device", "timestamp_ns": 200, "pair_index": 1},
    }
    values = np.ones((3, 3), np.float32)
    box = builtin_registry().create("3d.clean_tof_depth")
    result = box.execute(
        {
            "depth": ValuePacket(DepthArray(values), payload),
            "confidence": ValuePacket(values, {**payload, mismatch: other[mismatch]}),
        }
    )
    assert result is None
    assert ("coordinate frames" if mismatch == "spatial" else "must refer to the same") in box.error


def test_content_failure_retains_raw_depth_and_geometry_checks_can_be_disabled():
    box = builtin_registry().create(
        "3d.clean_tof_depth",
        parameters={"reject_flying_pixels": False, "reject_isolated_outliers": False},
    )
    values = np.array([[1.0, 2.0], [0.0, np.inf]], np.float64)
    confidence = np.array([[0, 0.1], [1.0, 1.0]], np.float32)
    result = box.execute({"depth": values, "confidence": confidence})
    assert result is not None and not result.passed
    assert result.fail_reasons[0]["code"] == "no_trusted_depth_in_selection"
    assert result.outputs["depth"].dtype == np.float64
    np.testing.assert_array_equal(result.outputs["reason_codes"], [[1, 1], [4, 4]])
    np.testing.assert_array_equal(result.outputs["raw_depth"], values)
    np.testing.assert_array_equal(result.outputs["depth"], [[np.nan, np.nan], [0.0, np.inf]])
