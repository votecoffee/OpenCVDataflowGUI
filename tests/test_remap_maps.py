"""Verify reusable lens maps, frame separation, and calibration project wiring."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame, packet_for

_PROFILE = {
    "model": "opencv_pinhole_v1",
    "camera_matrix": [[135, 0, 95.5], [0, 137, 71.5], [0, 0, 1]],
    "distortion": [-0.28, 0.04, 0.001, -0.001, 0.0],
    "image_size": [192, 144],
    "rms_pixels": 0.08,
    "units": "mm",
}


def _run(block, inputs):
    result = block.execute(inputs)
    assert result is not None and result.passed, block.error
    return result


def test_scaled_lens_maps_match_native_coordinates_and_separate_image_frames():
    registry = builtin_registry()
    builder = registry.create("calibration.lens_maps", parameters={"width": 96, "height": 108})
    maps = _run(builder, {"calibration": _PROFILE})
    k = np.asarray(_PROFILE["camera_matrix"], np.float64)
    d = np.asarray(_PROFILE["distortion"])
    target = np.diag([0.5, 0.75, 1]) @ k
    expected = cv2.initUndistortRectifyMap(k, d, None, target, (96, 108), cv2.CV_32FC1)
    np.testing.assert_array_equal(maps.outputs["map_x"], expected[0])
    np.testing.assert_array_equal(maps.outputs["map_y"], expected[1])
    assert maps.outputs["map_x"].dtype == np.float32
    assert maps.outputs["rectified_calibration"]["camera_matrix"] == target.tolist()
    assert maps.outputs["rectified_calibration"]["distortion"] == [0] * 5
    assert maps.outputs["valid_roi"] is NO_VALUE

    remap = registry.create("cv.remap")
    source = np.random.default_rng(19).integers(0, 65535, (144, 192, 3), dtype=np.uint16)
    spatial = frame([192, 144], "camera")
    inputs = {
        "image": ValuePacket(source, {"spatial": spatial, "sample": "keep"}),
        "map_x": packet_for(maps, "map_x"),
        "map_y": packet_for(maps, "map_y"),
    }
    result = _run(remap, inputs)
    native = cv2.remap(source, *expected, cv2.INTER_LINEAR)
    np.testing.assert_array_equal(result.outputs["image"], native)
    payload = result.payloads["image"]
    assert payload["sample"] == "keep" and payload["spatial"]["image_size"] == [96, 108]
    assert payload["spatial"]["root_id"] != spatial["root_id"]
    assert _run(remap, inputs).payloads["image"]["spatial"] == payload["spatial"]
    inputs["image"] = ValuePacket(source, {"spatial": frame([192, 144], "other-camera")})
    assert _run(remap, inputs).payloads["image"]["spatial"] != payload["spatial"]


def test_optimal_intrinsics_roi_and_custom_rotated_maps():
    registry = builtin_registry()
    k = np.asarray(_PROFILE["camera_matrix"], np.float64)
    d = np.asarray(_PROFILE["distortion"])
    builder = registry.create(
        "calibration.lens_maps",
        parameters={"camera_matrix_mode": "optimal", "alpha": 0.65, "center_principal_point": True},
    )
    optimal = _run(builder, {"calibration": _PROFILE})
    target, roi = cv2.getOptimalNewCameraMatrix(
        k, d, (192, 144), 0.65, (192, 144), centerPrincipalPoint=True
    )
    assert optimal.outputs["valid_roi"] == list(roi)
    np.testing.assert_allclose(optimal.outputs["rectified_calibration"]["camera_matrix"], target)
    vector = [0.03, 0.08, -0.02]
    builder = registry.create(
        "calibration.lens_maps",
        parameters={
            "camera_matrix_mode": "custom",
            "new_camera_matrix": target.tolist(),
            "rotation_vector": vector,
        },
    )
    custom = _run(builder, {"calibration": _PROFILE})
    rotation = cv2.Rodrigues(np.asarray(vector))[0]
    expected_x, _ = cv2.initUndistortRectifyMap(k, d, rotation, target, (192, 144), cv2.CV_32FC1)
    np.testing.assert_array_equal(custom.outputs["map_x"], expected_x)
    assert custom.outputs["valid_roi"] is NO_VALUE


def test_remap_rejects_swapped_mixed_and_wrong_size_tagged_maps():
    registry = builtin_registry()
    builder = registry.create("calibration.lens_maps")
    first = _run(builder, {"calibration": _PROFILE})
    second = _run(builder, {"calibration": {**_PROFILE, "distortion": [0.0] * 5}})
    x, y = (packet_for(first, port) for port in ("map_x", "map_y"))
    remap = registry.create("cv.remap")
    image = np.zeros((144, 192), np.uint8)
    for source, mx, my, message in (
        (image, y, x, "matching X and Y"),
        (image, x, packet_for(second, "map_y"), "matching X and Y"),
        (image, x, first.outputs["map_y"].view(np.ndarray), "matching X and Y"),
        (image[:-1], x, y, "Source image dimensions"),
    ):
        assert remap.execute({"image": source, "map_x": mx, "map_y": my}) is None
        assert message in remap.error


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"rms_pixels": 1.5}, "allowed reprojection error"),
        ({"image_size": [192, 0]}, "valid [width, height]"),
    ],
)
def test_lens_map_builder_rejects_unusable_profiles(change, message):
    builder = builtin_registry().create("calibration.lens_maps")
    assert builder.execute({"calibration": {**_PROFILE, **change}}) is None
    assert message in builder.error


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_saved_lens_map_example_applies_a_project_relative_profile(profile):
    project = Path(__file__).parents[1] / "examples/lens_correction_maps/lens_correction_maps.json"
    graph = Graph.load(project, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    source = graph.blocks["source"].last_result
    corrected = graph.blocks["remap"].last_result
    image = corrected.outputs["image"]
    assert image.shape == (144, 192) and image.dtype == np.uint8
    assert corrected.payloads["image"]["spatial"] != source.payloads["image"]["spatial"]
    assert corrected.payloads["image"]["ticket"] == source.payloads["image"]["ticket"]
    # The off-center vertical line is straighter and darker after correction.
    assert np.mean(image[22:122, 24]) < np.mean(source.outputs["image"][22:122, 24]) - 50
