"""Validate bounded depth-hole filling and reconstructed-pixel provenance."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import DepthArray
from opencv_dataflow.payloads import ValuePacket, frame


def test_fill_depth_holes_preserves_physical_units_and_marks_estimates():
    values = np.full((18, 20), 1500, np.float32)
    values[6:9, 6:9] = np.nan
    values[0, 13] = np.nan
    values[12:16, 11:17] = np.nan
    depth = DepthArray(values, "mm", 300, 3000)
    payload = {"spatial": frame([20, 18], "camera")}
    block = builtin_registry().create(
        "3d.fill_depth_holes", parameters={"maximum_hole_area": 10, "maximum_distance": 2}
    )
    result = block.execute({"depth": ValuePacket(depth, payload)})
    assert result is not None and result.passed, block.error
    output = result.outputs["depth"]
    assert isinstance(output, DepthArray)
    assert (output.units, output.display_min, output.display_max) == ("mm", 300, 3000)
    assert result.outputs["filled_count"] == 9
    assert np.all(output[6:9, 6:9] == 1500)
    assert np.isnan(output[0, 13]) and np.isnan(output[12:16, 11:17]).all()
    assert np.isnan(depth[6:9, 6:9]).all()
    assert result.payloads["filled_mask"]["spatial"] == payload["spatial"]
    np.testing.assert_array_equal(result.outputs["remaining_mask"] != 0, np.isnan(output))


@pytest.mark.parametrize("metric,filled_count", [("manhattan", 0), ("chessboard", 1)])
def test_depth_fill_distance_and_mask_limit_writes(metric, filled_count):
    depth = np.full((9, 9), 800, np.uint16)
    depth[3:6, 3:6] = 0
    depth[2, 2] = 1000
    selection = np.zeros((9, 9), np.uint8)
    selection[3, 3] = 255
    # A diagonal sample is within a distance of 1 only in the chessboard metric.
    depth[2, 3] = depth[3, 2] = 0
    block = builtin_registry().create(
        "3d.fill_depth_holes", parameters={"distance_metric": metric, "maximum_distance": 1}
    )
    result = block.execute({"depth": depth, "mask": selection})
    assert result is not None, block.error
    assert result.outputs["filled_count"] == filled_count
    assert result.outputs["depth"].dtype == np.uint16
    np.testing.assert_array_equal(result.outputs["depth"][selection == 0], depth[selection == 0])
    assert bool(result.outputs["depth"][3, 3]) == bool(filled_count)


def test_depth_fill_does_not_fabricate_samples_for_an_empty_depth_plane():
    block = builtin_registry().create("3d.fill_depth_holes", parameters={"include_border": True})
    result = block.execute({"depth": np.zeros((8, 8), np.uint16)})
    assert result is not None and not result.passed, block.error
    assert result.outputs["filled_count"] == 0
    assert (result.outputs["remaining_mask"] == 255).all()
