"""Independent full-sensor projection references for reduced ROS image grids."""

import copy

import cv2
import numpy as np
import pytest
from test_ros2_rectification import calibration, image, run

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import ValuePacket, frame, packet_for
from opencv_dataflow.ros2_calibration import camera_from_info
from opencv_dataflow.ros2_rectification import RectifyROSImage


def acquisition(fisheye=False, bins=(2, 2), roi=(8, 4, 16, 12), identity=False):
    fields = calibration(fisheye, identity=identity)
    fields["binning_x"], fields["binning_y"] = bins
    x, y, width, height = roi
    fields["roi"].update(x_offset=x, y_offset=y, width=width, height=height)
    full = image()
    bx, by = bins
    source_size = (width // bx, height // by)
    pixels = np.asarray(full.value)[
        y : y + source_size[1] * by : by, x : x + source_size[0] * bx : bx
    ]
    payload = {
        "spatial": frame(list(source_size), "received_reduced_grid"),
        "ros2": {
            **copy.deepcopy(full.value.metadata["ros2"]),
            "width": source_size[0],
            "height": source_size[1],
            "encoding": "mono8",
        },
    }
    return fields, ValuePacket(
        ImageArray(pixels.copy(), "gray", payload), copy.deepcopy(full.payload)
    )


@pytest.mark.parametrize("fisheye", [False, True])
@pytest.mark.parametrize("bins", [(2, 2), (2, 3)])
@pytest.mark.parametrize("grid", ["camera_info_roi", "full_sensor", "custom_roi"])
def test_reduced_maps_match_full_sensor_native_projection(fisheye, bins, grid):
    fields, value = acquisition(fisheye, bins)
    original = copy.deepcopy(fields)
    parameters = {"rectified_grid": grid}
    roi = [8, 4, 16, 12]
    if grid == "full_sensor":
        roi = [0, 0, 32, 24]
        fields["roi"]["do_rectify"] = True  # Explicit canvas selection avoids guessing a crop.
    elif grid == "custom_roi":
        roi = [4, 2, 24, 18]
        parameters.update(rectified_roi=roi, width=18, height=9)
        fields["roi"]["do_rectify"] = True
    block = RectifyROSImage(parameters=parameters)
    result = run(block, image=value, camera_info=fields)
    bx, by = bins
    reduced = (roi[2] // bx, roi[3] // by)
    size = (18, 9) if grid == "custom_roi" else reduced
    p = np.array(fields["p"]).reshape(3, 4)[:, :3]
    # The full-resolution virtual grid is chosen independently of the raw crop.
    full_from_output = np.array(
        [[bx * reduced[0] / size[0], 0, roi[0]], [0, by * reduced[1] / size[1], roi[1]], [0, 0, 1]]
    )
    new_k = np.linalg.solve(full_from_output, p)
    full_k = np.array(fields["k"]).reshape(3, 3)
    rotation = np.array(fields["r"]).reshape(3, 3)
    native = cv2.fisheye.initUndistortRectifyMap if fisheye else cv2.initUndistortRectifyMap
    full_x, full_y = native(full_k, np.array(fields["d"]), rotation, new_k, size, cv2.CV_32FC1)
    expected_x, expected_y = (full_x - 8) / bx, (full_y - 4) / by
    np.testing.assert_allclose(result.outputs["map_x"], expected_x, atol=2e-6)
    np.testing.assert_allclose(result.outputs["map_y"], expected_y, atol=2e-6)
    expected = cv2.remap(
        np.asarray(value.value),
        expected_x,
        expected_y,
        cv2.INTER_LINEAR,
    )
    np.testing.assert_array_equal(result.outputs["image"], expected)
    np.testing.assert_allclose(result.outputs["camera"]["camera_matrix"], new_k)
    np.testing.assert_allclose(result.outputs["info"]["rectified_output_to_full"], full_from_output)
    assert not result.outputs["info"]["rectified_roi_estimated"]
    assert result.outputs["info"]["source_geometry"]["binning"] == list(bins)
    assert original["k"] == fields["k"] and original["p"] == fields["p"]
    canonical, _ = camera_from_info(result.outputs["camera_info"])
    assert canonical["image_size"] == list(size)
    assert result.outputs["camera_info"]["binning_x"] == 0
    assert result.outputs["camera_info"]["roi"]["width"] == 0


def test_identity_capture_roi_is_exact_and_full_canvas_marks_uncaptured_pixels_invalid():
    fields, value = acquisition(identity=True)
    captured = run(RectifyROSImage(), image=value, camera_info=fields)
    np.testing.assert_array_equal(captured.outputs["image"], value.value)
    assert np.all(captured.outputs["valid_mask"] == 255)
    full = run(
        RectifyROSImage(parameters={"rectified_grid": "full_sensor"}),
        image=value,
        camera_info=fields,
    )
    assert full.outputs["image"].shape == (12, 16)
    np.testing.assert_array_equal(full.outputs["image"][2:8, 4:12], value.value)
    assert np.count_nonzero(full.outputs["valid_mask"]) == 48
    assert full.outputs["image"][0, 0] == 0


def test_partial_bins_are_reported_without_inventing_extra_source_or_output_pixels():
    fields, value = acquisition(bins=(2, 3), roi=(8, 4, 17, 13))
    result = run(RectifyROSImage(), image=value, camera_info=fields)
    assert result.outputs["image"].shape == (4, 8)
    info = result.outputs["info"]
    assert info["source_geometry"]["discarded_partial_bins"] == [1, 1]
    assert info["discarded_rectified_partial_bins"] == [1, 1]


def test_roi_request_is_not_silently_treated_as_an_estimated_rectified_crop():
    fields, value = acquisition()
    fields["roi"]["do_rectify"] = True
    block = RectifyROSImage()
    assert block.execute({"image": value, "camera_info": fields}) is None
    assert block.error_kind == "invalid" and "Full sensor" in block.error
    block.parameters["rectified_grid"] = "full_sensor"
    result = run(block, image=value, camera_info=fields)
    assert result.outputs["info"]["rectified_roi_full"] == [0, 0, 32, 24]


@pytest.mark.parametrize(
    "roi",
    [
        [0, 0, 0, 10],
        [0, 0, 33, 24],
        [-1, 0, 10, 10],
        [0, 0, 10.0, 10],
        [False, 0, 10, 10],
        [0, 0, 1, 10],
    ],
)
def test_custom_rectified_roi_is_validated(roi):
    fields, value = acquisition()
    block = RectifyROSImage(parameters={"rectified_grid": "custom_roi", "rectified_roi": roi})
    assert block.execute({"image": value, "camera_info": fields}) is None
    assert block.error_kind == "invalid" and not block.current_emissions


def test_known_crop_binning_affine_is_accepted_but_unrelated_spatial_mapping_is_rejected():
    fields, received = acquisition(identity=True)
    warp = builtin_registry().create(
        "cv.warp_affine",
        parameters={
            "matrix": [[0.5, 0, -4], [0, 0.5, -2]],
            "width": 8,
            "height": 6,
            "interpolation": "nearest",
        },
    )
    reduced = run(warp, image=image())
    source = packet_for(reduced, "image")
    np.testing.assert_array_equal(source.value, received.value)
    result = run(RectifyROSImage(), image=source, camera_info=fields)
    np.testing.assert_array_equal(result.outputs["image"], received.value)
    remap = builtin_registry().create("cv.remap")
    replay = run(
        remap, image=source, map_x=packet_for(result, "map_x"), map_y=packet_for(result, "map_y")
    )
    np.testing.assert_array_equal(replay.outputs["image"], result.outputs["image"])
    source.payload["spatial"]["to_root"][0][2] += 1
    block = RectifyROSImage()
    assert block.execute({"image": source, "camera_info": fields}) is None
    assert block.error_kind == "invalid"


def test_grid_changes_rebuild_maps_when_output_intrinsics_change():
    fields, value = acquisition()
    block = RectifyROSImage()
    first = run(block, image=value, camera_info=fields)
    maps = block._maps
    block.parameters["rectified_grid"] = "full_sensor"
    second = run(block, image=value, camera_info=fields)
    assert block._maps is not maps and first.outputs["info"]["id"] != second.outputs["info"]["id"]
