"""Check the scale reference, displacement measurements and result graphics."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import packet_for

_PROFILE = {
    "model": "opencv_pinhole_v1",
    "camera_matrix": [[32, 0, 11.5], [0, 32, 7.5], [0, 0, 1]],
    "distortion": [0.0] * 5,
    "image_size": [24, 16],
    "rms_pixels": 0.0,
    "units": "mm",
}


def _inputs(registry, **parameters):
    builder = registry.create("calibration.lens_maps", parameters=parameters)
    maps = builder.execute({"calibration": _PROFILE})
    assert maps is not None, builder.error
    return {port: packet_for(maps, port) for port in ("map_x", "map_y")}


def test_displacement_removes_the_expected_output_scaling():
    registry = builtin_registry()
    inputs = _inputs(registry, width=12, height=8)
    tool = registry.create("calibration.lens_map_displacement", parameters={"arrow_spacing": 4})
    result = tool.execute(inputs)
    assert result is not None, tool.error
    np.testing.assert_allclose(result.outputs["magnitude"], 0, atol=1e-5)
    assert result.outputs["valid_fraction"] == 1.0
    assert np.all(result.outputs["valid_mask"] == 255)
    assert not result.drawings


def test_displacement_measures_known_shift_and_marks_outside_coordinates():
    registry = builtin_registry()
    inputs = _inputs(
        registry,
        camera_matrix_mode="custom",
        new_camera_matrix=[[32, 0, 14.5], [0, 32, 3.5], [0, 0, 1]],
    )
    tool = registry.create(
        "calibration.lens_map_displacement",
        parameters={"colormap": "viridis", "heatmap_max": 3.0, "arrow_spacing": 10},
    )
    result = tool.execute(inputs)
    assert result is not None, tool.error
    np.testing.assert_allclose(result.outputs["magnitude"], 5.0)
    summary = result.outputs["summary"]
    assert summary["mean_source_pixels"] == summary["rms_source_pixels"] == 5.0
    assert summary["valid_count"] == 21 * 12
    mask = np.zeros((16, 24), np.uint8)
    mask[:12, 3:] = 255
    np.testing.assert_array_equal(result.outputs["valid_mask"], mask)
    full_scale = cv2.applyColorMap(np.full((16, 24), 255, np.uint8), cv2.COLORMAP_VIRIDIS)
    np.testing.assert_array_equal(result.outputs["heatmap"], full_scale)
    assert result.payloads["heatmap"]["spatial"] == inputs["map_x"].payload["spatial"]
    assert "lens_remap_map" not in result.payloads["heatmap"]
    arrow = result.drawings["Displacement vectors"].objects[0]
    assert arrow.points[:2] == ((5, 5), (2, 9))


def test_displacement_requires_tagged_matching_map_planes():
    registry = builtin_registry()
    tool = registry.create("calibration.lens_map_displacement")
    x = np.zeros((8, 8), np.float32)
    assert tool.execute({"map_x": x, "map_y": x}) is None
    assert "Build lens correction maps" in tool.error
    maps = _inputs(registry)
    assert tool.execute({"map_x": maps["map_y"], "map_y": maps["map_x"]}) is None
    assert "matching X and Y" in tool.error


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_saved_displacement_example_overlays_arrows_on_the_heatmap(profile):
    project = (
        Path(__file__).parents[1]
        / "examples/custom tools/lens_map_displacement/lens_map_displacement.json"
    )
    graph = Graph.load(project, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    result = graph.blocks["diagnostics"].last_result
    heatmap = result.outputs["heatmap"]
    overlay = graph.blocks["overlay"].last_result.outputs["image"]
    assert heatmap.shape == overlay.shape == (144, 192, 3)
    assert np.any(heatmap != overlay)
    assert result.outputs["maximum"] > 10
    assert result.outputs["valid_fraction"] == 1.0
