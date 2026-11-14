"""Native and independent projection references for explicit ROS intensity geometry."""

import copy

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import camera_record
from opencv_dataflow.images import DepthArray, ImageArray
from opencv_dataflow.parameters import BORDERS
from opencv_dataflow.payloads import ValuePacket, frame, packet_for
from opencv_dataflow.ros2_calibration import camera_from_info, camera_to_info
from opencv_dataflow.ros2_camera_cache import CacheROSCamera
from opencv_dataflow.ros2_context import AttachAcquisition, RunIdentity
from opencv_dataflow.ros2_images import encode_image
from opencv_dataflow.ros2_rectification import RectifyROSImage
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry


def calibration(fisheye=False, identity=False):
    model = camera_record(
        [[30, 0, 15.5], [0, 31, 11.5], [0, 0, 1]],
        [0] * (4 if fisheye else 5)
        if identity
        else ([0.03, -0.005, 0.001, -0.0002] if fisheye else [0.08, -0.015, 0.002, -0.001, 0.003]),
        (32, 24),
        frame="camera_optical",
        lens="fisheye" if fisheye else "pinhole",
    )
    result = camera_to_info(model)
    result["header"]["stamp"] = {"sec": 3, "nanosec": 0}
    result["r"] = (
        (np.eye(3) if identity else cv2.Rodrigues(np.array([0.03, -0.08, 0.04]))[0])
        .ravel()
        .tolist()
    )
    return result


def image(dtype=np.uint8, color="gray", header=True):
    pixels = np.arange(32 * 24).reshape(24, 32) % 200
    if color != "gray":
        pixels = np.dstack(
            [pixels, pixels // 2, pixels // 3] + ([pixels * 0 + 255] if color.endswith("a") else [])
        )
    metadata = {"spatial": frame([32, 24], "raw_pixels")}
    if header:
        metadata["ros2"] = {
            "header": {"frame_id": "camera_optical", "stamp": {"sec": 17, "nanosec": 9}}
        }
    return ValuePacket(
        ImageArray(pixels.astype(dtype), color, metadata),
        {"ticket": TicketClock("rectify-test").issue("camera")},
    )


def run(block, **inputs):
    result = block.execute(inputs)
    assert result is not None, block.error
    return result


@pytest.mark.parametrize(
    "fisheye,dtype,color,interpolation",
    [
        (False, np.uint8, "rgb", "nearest"),
        (False, np.uint16, "bgr", "linear"),
        (False, np.float32, "gray", "cubic"),
        (False, np.float64, "gray", "lanczos4"),
        (False, np.int16, "gray", "nearest"),
        (True, np.uint8, "bgra", "linear"),
        (True, np.uint16, "gray", "nearest"),
    ],
)
def test_rectification_matches_native_maps_and_pixels(fisheye, dtype, color, interpolation):
    fields, value = calibration(fisheye), image(dtype, color)
    original = copy.deepcopy(fields)
    block = RectifyROSImage(
        parameters={
            "interpolation": interpolation,
            "width": 24,
            "height": 18,
            "scalar_semantics": "intensity",
        }
    )
    result = run(block, image=value, camera_info=fields)
    k = np.array(fields["k"]).reshape(3, 3)
    r = np.array(fields["r"]).reshape(3, 3)
    new_k = np.array(fields["p"]).reshape(3, 4)[:, :3].copy()
    new_k[:2] *= 0.75
    native = cv2.fisheye.initUndistortRectifyMap if fisheye else cv2.initUndistortRectifyMap
    mx, my = native(k, np.array(fields["d"]), r, new_k, (24, 18), cv2.CV_32FC1)
    np.testing.assert_array_equal(result.outputs["map_x"], mx)
    np.testing.assert_array_equal(result.outputs["map_y"], my)
    expected = cv2.remap(
        np.asarray(value.value),
        mx,
        my,
        {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "lanczos4": cv2.INTER_LANCZOS4,
        }[interpolation],
    )
    np.testing.assert_array_equal(result.outputs["image"], expected)
    assert result.outputs["image"].dtype == np.dtype(dtype)
    assert result.outputs["image"].color_format == color
    assert fields == original
    camera, _ = camera_from_info(result.outputs["camera_info"])
    np.testing.assert_allclose(camera["camera_matrix"], new_k)
    np.testing.assert_allclose(np.array(result.outputs["pose"]["source_to_target"])[:3, :3], r)
    assert camera["frame"] == "camera_optical_rectified"
    assert camera["ros2"]["header"]["stamp"] == {"sec": 17, "nanosec": 9}
    assert result.outputs["info"]["source_camera_info"]["header"]["stamp"]["sec"] == 3


def test_independent_point_projection_matches_destination_to_source_maps():
    fields = calibration()
    result = run(RectifyROSImage(), image=image(), camera_info=fields)
    samples = np.array([[8, 6], [16, 12], [23, 17]], np.float64)
    k = np.array(fields["k"]).reshape(3, 3)
    r = np.array(fields["r"]).reshape(3, 3)
    rays = (r.T @ np.linalg.inv(k) @ np.c_[samples, np.ones(3)].T).T
    source = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), k, np.array(fields["d"]))[0].reshape(
        -1, 2
    )
    measured = np.array(
        [
            [result.outputs["map_x"][int(y), int(x)], result.outputs["map_y"][int(y), int(x)]]
            for x, y in samples
        ]
    )
    np.testing.assert_allclose(measured, source, atol=2e-6)


def test_big_endian_samples_are_normalized_before_the_native_remap():
    value = image(np.dtype(">u2"))
    result = run(RectifyROSImage(), image=value, camera_info=calibration(identity=True))
    np.testing.assert_array_equal(result.outputs["image"], value.value)
    assert result.outputs["image"].dtype.isnative


def test_identity_and_missing_header_use_no_calibration_timestamp_as_image_time():
    value, fields = image(header=False), calibration(identity=True)
    block = RectifyROSImage(parameters={"camera_info": fields})
    assert block.execute({"image": value}) is None
    block.parameters["header_mode"] = "zero_if_missing"
    result = run(block, image=value)
    np.testing.assert_array_equal(result.outputs["image"], value.value)
    assert np.all(result.outputs["valid_mask"] == 255)
    assert result.outputs["camera_info"]["header"]["stamp"] == {"sec": 0, "nanosec": 0}
    assert result.outputs["info"]["source_camera_info"]["header"]["stamp"]["sec"] == 3


@pytest.mark.parametrize("dtype,encoding", [(np.float32, "32FC1"), (np.uint16, "16UC1")])
def test_ambiguous_scalar_input_requires_an_explicit_intensity_declaration(dtype, encoding):
    value, fields = image(dtype), calibration()
    value.value.metadata["ros2"]["encoding"] = encoding
    block = RectifyROSImage()
    assert block.execute({"image": value, "camera_info": fields}) is None
    assert block.error_kind == "invalid"
    block.parameters["scalar_semantics"] = "intensity"
    assert run(block, image=value, camera_info=fields).passed


def test_already_rectified_camera_model_is_rejected_and_bound_pixel_grid_is_checked():
    fields, value = calibration(identity=True), image()
    model = camera_from_info(fields, space="rectified")[0]
    assert RectifyROSImage().execute({"image": value, "camera_info": model}) is None
    value.value.metadata["ros2"].update(width=32, height=24, encoding="mono8")
    cache = CacheROSCamera()
    run(cache, camera=camera_from_info(fields)[0])
    bound = run(cache, image=value)
    model_packet = packet_for(bound, "camera")
    model_packet.payload["spatial"]["root_id"] = "old_grid"
    assert (
        RectifyROSImage().execute(
            {"image": packet_for(bound, "image"), "camera_info": model_packet}
        )
        is None
    )


@pytest.mark.parametrize(
    "change",
    [
        "depth",
        "frame",
        "grid",
        "roi",
        "binning",
        "baseline",
        "rotation",
        "same_frame",
        "width",
        "size_limit",
        "int32",
        "projection",
    ],
)
def test_invalid_input_is_contained(change):
    fields, value, parameters = calibration(), image(), {}
    if change == "depth":
        value.value = DepthArray(value.value, "m")
    elif change == "frame":
        value.value.metadata["ros2"]["header"]["frame_id"] = "other"
    elif change == "grid":
        value.value.metadata["spatial"]["to_root"][0][2] = 1
    elif change == "roi":
        fields["roi"].update(x_offset=1, width=31, height=24)
    elif change == "binning":
        fields["binning_x"] = 2
    elif change == "baseline":
        fields["p"][3] = -3
    elif change == "rotation":
        fields["r"][0] = 2
    elif change == "same_frame":
        parameters["rectified_frame"] = "camera_optical"
    elif change == "width":
        parameters["width"] = 16
    elif change == "size_limit":
        parameters.update(width=16384, height=16384)
    elif change == "projection":
        fields["p"][0] = -1
    else:
        value.value = ImageArray(value.value.astype(np.int32), "gray", value.value.metadata)
    block = RectifyROSImage(parameters=parameters)
    assert block.execute({"image": value, "camera_info": fields}) is None
    assert not block.current_emissions and block.error_kind == "invalid"


def test_no_front_facing_pixels_is_a_contained_failed_result_even_with_replicated_border():
    fields = calibration(identity=True)
    fields["r"] = np.diag([1.0, -1.0, -1.0]).ravel().tolist()
    block = RectifyROSImage(parameters={"border": "replicate", "border_value": [10, 20, 30]})
    result = run(block, image=image(color="rgb"), camera_info=fields)
    assert not result.passed and result.fail_reasons
    assert not result.outputs["valid_mask"].any()
    np.testing.assert_array_equal(result.outputs["image"], np.tile([10, 20, 30], (24, 32, 1)))


def test_maps_reuse_geometry_but_never_share_output_buffers_or_old_pixel_frame():
    block, fields, value = RectifyROSImage(), calibration(), image()
    first = run(block, image=value, camera_info=fields)
    cached = block._maps
    assert first.payloads["image"]["spatial"] != value.value.metadata["spatial"]
    first.outputs["map_x"][:] = 999
    fields["header"]["stamp"]["sec"] = 4
    second = run(block, image=value, camera_info=fields)
    assert block._maps is cached and np.max(second.outputs["map_x"]) < 999
    other = RectifyROSImage()
    other.copy_runtime_from(block)
    assert other._maps is None
    remap = builtin_registry().create("cv.remap")
    replay = run(
        remap, image=value, map_x=packet_for(second, "map_x"), map_y=packet_for(second, "map_y")
    )
    np.testing.assert_array_equal(replay.outputs["image"], second.outputs["image"])
    assert replay.payloads["image"]["spatial"] == second.payloads["image"]["spatial"]
    assert second.payloads["valid_mask"]["ros2"]["encoding"] == "mono8"


def test_static_calibration_binding_and_ticket_checks_are_preserved():
    fields, value = calibration(), image()
    camera = camera_from_info(fields)[0]
    # Full-grid cache binds the raw model before rectification changes pixel coordinates.
    value.value.metadata["ros2"].update(width=32, height=24, encoding="mono8")
    cache = CacheROSCamera()
    run(cache, camera=camera)
    bound = run(cache, image=value)
    result = run(
        RectifyROSImage(), image=packet_for(bound, "image"), camera_info=packet_for(bound, "camera")
    )
    assert "ros2_static_binding" not in result.payloads["image"]
    assert result.payloads["image"]["ticket"] == value.payload["ticket"]
    assert result.outputs["camera"]["ros2"]["header"]["frame_id"] == "camera_optical_rectified"
    assert encode_image(result.outputs["image"]).header["frame_id"] == "camera_optical_rectified"
    block = RectifyROSImage(parameters={"require_matching_ticket": True})
    assert block.execute({"image": value, "camera_info": fields}) is None


@pytest.mark.parametrize("border", list(BORDERS))
def test_border_choices_match_native_without_marking_extrapolation_valid(border):
    fields, value = calibration(), image(np.uint16, "rgb")
    result = run(
        RectifyROSImage(parameters={"border": border, "border_value": [-1, 65536, 7]}),
        image=value,
        camera_info=fields,
    )
    expected = cv2.remap(
        np.asarray(value.value),
        np.asarray(result.outputs["map_x"]),
        np.asarray(result.outputs["map_y"]),
        cv2.INTER_LINEAR,
        borderMode=BORDERS[border],
        borderValue=(-1, 65536, 7),
    )
    np.testing.assert_array_equal(result.outputs["image"], expected)
    assert np.count_nonzero(result.outputs["valid_mask"]) < result.outputs["valid_mask"].size


def test_tagged_calibration_cannot_cross_run_generations_and_recipes_must_agree():
    source, attach = RunIdentity(), AttachAcquisition()
    old = run(source).outputs["context"]
    fields = calibration()
    calibration_packet = ValuePacket(fields, {"ticket": TicketClock().issue("calibration")})
    old_calibration = run(attach, value=calibration_packet, context=old)
    current = run(source, reset=True).outputs["context"]
    current_image = run(attach, value=image(), context=current)
    block = RectifyROSImage()
    assert (
        block.execute(
            {
                "image": packet_for(current_image, "value"),
                "camera_info": packet_for(old_calibration, "value"),
            }
        )
        is None
    )
    assert block.error_kind == "invalid" and not block.current_emissions
    value = image()
    ref = {"id": "inspection", "revision": 1, "digest": "a" * 64}
    value.payload["recipe_ref"] = ref
    conflict = ValuePacket(fields, {"recipe_ref": {**ref, "revision": 2}})
    assert block.execute({"image": value, "camera_info": conflict}) is None
    assert block.error_kind == "invalid"
    matched = ValuePacket(fields, {"recipe_ref": ref, "ticket": value.payload["ticket"]})
    block.parameters["require_matching_ticket"] = True
    result = run(block, image=value, camera_info=matched)
    assert result.payloads["camera"]["recipe_ref"] == ref


def test_registry_xml_and_offscreen_controls(app, tmp_path):
    from PySide6.QtWidgets import QComboBox

    from opencv_dataflow.app import MainWindow

    graph = Graph()
    block = graph.add(
        RectifyROSImage(
            parameters={
                "camera_info": calibration(),
                "rectified_grid": "custom_roi",
                "rectified_roi": [4, 2, 24, 18],
                "width": 18,
                "height": 9,
            }
        )
    )
    run(block, image=image())
    path = tmp_path / "rectify.xml"
    graph.save(path)
    restored = Graph.load(path, builtin_registry())
    loaded = next(iter(restored.blocks.values()))
    assert loaded._maps is None and loaded.parameters == block.parameters
    assert block_entry(loaded).path == ("Industrial integration", "ROS 2")
    window = MainWindow()
    try:
        window.replace_graph(graph)
        window.scene.nodes[block.id].setSelected(True)
        for name in (
            "interpolation",
            "border",
            "header_mode",
            "scalar_semantics",
            "rectified_grid",
        ):
            assert isinstance(window.inspector.fields[name], QComboBox)
        assert window.inspector.can_preview()
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
        restored.close()
