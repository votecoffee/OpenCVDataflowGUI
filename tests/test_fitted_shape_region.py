"""Verify fitted region geometry, provenance and downstream mask use."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import ValuePacket, frame, metadata

TYPE_ID = "region.fitted_shape"
_PROJECT = (
    Path(__file__).parents[1] / "examples/custom tools/fitted_shape_region/fitted_shape_region.json"
)


def test_rotated_ellipse_preserves_center_angle_and_full_axis_units():
    spatial = frame([120, 100], "camera")
    image = ImageArray(np.zeros((100, 120), np.uint8), "gray", {"spatial": spatial})
    fit = {
        "shape": "ellipse",
        "center": [59.25, 51.75],
        "axis_lengths": [50.0, 18.0],
        "angle_degrees": 30.0,
    }
    block = builtin_registry().create(TYPE_ID, parameters={"margin": 2.5})
    result = block.execute({"image": image, "fit": ValuePacket(fit, {"spatial": spatial})})
    assert result and result.passed, block.error
    region = result.outputs["region"]
    assert region["center"] == fit["center"]
    assert region["angle_degrees"] == 30
    assert region["axis_lengths"] == [55, 23]
    assert region["coordinate_context"] == spatial
    mask = result.outputs["mask"]
    assert metadata(mask)["spatial"] == spatial
    assert mask[52, 59] == mask[63, 79] == 255
    assert mask[68, 50] == 0
    assert set(np.unique(mask)) == {0, 255}
    assert len(result.drawings["region"].objects) == 1
    assert not image.any()


def test_circle_margin_inversion_and_invalid_geometry():
    image = np.zeros((100, 120), np.uint8)
    fit = {"shape": "enclosing_circle", "center": [45.5, 40.25], "radius": 10.5}
    block = builtin_registry().create(TYPE_ID, parameters={"margin": -1.5})
    result = block.execute({"image": image, "fit": fit})
    assert result and result.passed, block.error
    assert result.outputs["region"]["axis_lengths"] == [18, 18]
    mask = result.outputs["mask"]
    assert mask[40, 45] == 255 and mask[40, 57] == 0

    block.parameters["invert"] = True
    result = block.execute({"image": image, "fit": fit})
    np.testing.assert_array_equal(result.outputs["mask"], cv2.bitwise_not(mask))

    block.parameters["margin"] = -10.5
    assert block.execute({"image": image, "fit": fit}) is None
    assert "Margin collapses" in block.error
    assert block.execute({"image": image, "fit": {"shape": "line"}}) is None
    assert "ellipse or enclosing circle" in block.error


@pytest.mark.parametrize("spatial", (frame([120, 100], "other"), frame([60, 50])))
def test_rejects_fits_in_a_different_frame_or_size(spatial):
    image = ImageArray(
        np.zeros((100, 120), np.uint8), "gray", {"spatial": frame([120, 100], "camera")}
    )
    fit = {"shape": "enclosing_circle", "center": [10, 10], "radius": 5}
    block = builtin_registry().create(TYPE_ID)
    assert block.execute({"image": image, "fit": ValuePacket(fit, {"spatial": spatial})}) is None
    assert "target image coordinate frame" in block.error


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_example_measures_a_rotated_band_around_the_fitted_part(profile):
    graph = Graph.load(_PROJECT, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    outer = graph.blocks["outer"].last_result.outputs["mask"]
    inner = graph.blocks["inner"].last_result.outputs["mask"]
    band = graph.blocks["band"].last_result.outputs["mask"]
    np.testing.assert_array_equal(band, cv2.bitwise_and(outer, cv2.bitwise_not(inner)))
    assert band[64, 88] == 0
    assert 1200 < cv2.countNonZero(band) < 1800
    measured = graph.blocks["measure"].last_result.outputs
    assert measured["area"] == cv2.countNonZero(band)
    np.testing.assert_allclose(measured["centroid"], [[88.25, 63.75]], atol=1)
