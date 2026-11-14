"""Independent camera geometry and ticket/frame/time checks without ROS imports."""

import copy

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import camera_record
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket, frame
from opencv_dataflow.ros2_calibration import (
    AssociateROSRGBD,
    DecodeROSCamera,
    EncodeROSCamera,
    camera_from_info,
    camera_to_info,
)
from opencv_dataflow.ros2_images import decode_image, encode_image
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

HEADER = {"stamp": {"sec": 17, "nanosec": 987654321}, "frame_id": "camera_optical"}


def model(size=(6, 4), distortion=(), lens="pinhole"):
    return camera_record(
        [[10, 0, 2], [0, 12, 1], [0, 0, 1]],
        distortion,
        size,
        frame="camera_optical",
        lens=lens,
        **({"xi": 1.0} if lens == "omnidir" else {}),
        ros2={"header": HEADER},
    )


def group(zero=False, color=True):
    header = copy.deepcopy(HEADER)
    if zero:
        header["stamp"] = {"sec": 0, "nanosec": 0}
    calibration = model()
    calibration["ros2"]["header"] = header
    calibration = camera_from_info(camera_to_info(calibration))[0]
    ticket = TicketClock("ros-test").issue("sensor", "external", "part-17")
    rgb = ImageArray(
        np.tile(np.array([5, 10, 20], np.uint8), (4, 6, 1)), "bgr", {"ros2": {"header": header}}
    )
    depth = ImageArray(np.full((4, 6), 1000, np.uint16), "gray", {"ros2": {"header": header}})
    depth[0, 0] = 0
    values = [decode_image(encode_image(rgb)), decode_image(encode_image(depth)), calibration]
    entries = [ValuePacket(v, {"ticket": copy.deepcopy(ticket)}) for v in values]
    return FlowGroup(
        tuple(entries if color else entries[1:]),
        ("input_1", "input_2", "input_3") if color else ("input_2", "input_3"),
    )


@pytest.mark.parametrize(
    "lens,d",
    [
        ("pinhole", []),
        ("pinhole", [0.1, -0.1, 0, 0]),
        ("pinhole", [0.1, -0.1, 0, 0, 0.02]),
        ("pinhole", [0.01] * 8),
        ("fisheye", [0.01, -0.02, 0.03, 0.04]),
    ],
)
def test_camera_roundtrip_preserves_current_grid_and_lens(lens, d):
    source = model(distortion=d, lens=lens)
    fields = camera_to_info(source)
    decoded, info = camera_from_info(fields)
    np.testing.assert_array_equal(decoded["camera_matrix"], source["camera_matrix"])
    assert decoded["image_size"] == source["image_size"] and decoded["lens"] == lens
    if len(d) == 4 and lens == "pinhole":
        assert decoded["distortion"] == d + [0.0]
    elif d:
        assert decoded["distortion"] == d
    else:
        assert not np.any(decoded["distortion"])
    assert info["header"] == HEADER
    assert fields["roi"] == {
        "x_offset": 0,
        "y_offset": 0,
        "width": 0,
        "height": 0,
        "do_rectify": False,
    }


def test_roi_binning_geometry_matches_independent_projected_pixel_mapping():
    source = model((640, 480), distortion=[0.01, 0, 0, 0, 0])
    fields = camera_to_info(source)
    fields.update(
        binning_x=2,
        binning_y=4,
        roi={"x_offset": 10, "y_offset": 20, "width": 401, "height": 303, "do_rectify": False},
    )
    result, info = camera_from_info(fields)
    assert result["image_size"] == [200, 75] and info["discarded_partial_bins"] == [1, 3]
    points = np.array([[0.1, 0.2, 2.0], [0.2, -0.3, 3]], np.float64)
    expected, _ = cv2.projectPoints(
        points,
        np.zeros(3),
        np.zeros(3),
        np.array(source["camera_matrix"], float),
        np.array(source["distortion"]),
    )
    actual, _ = cv2.projectPoints(
        points,
        np.zeros(3),
        np.zeros(3),
        np.array(result["camera_matrix"]),
        np.array(result["distortion"]),
    )
    np.testing.assert_allclose(actual.reshape(-1, 2), (expected.reshape(-1, 2) - [10, 20]) / [2, 4])
    assert result["ros2"]["camera_info"] == fields
    assert source["image_size"] == [640, 480]


def test_rectified_selection_uses_projection_intrinsics_and_zero_distortion():
    fields = camera_to_info(model(distortion=[0.1, 0, 0, 0, 0]))
    fields["p"] = [20, 0, 3, 0, 0, 22, 2, 0, 0, 0, 1, 0]
    rectified, _ = camera_from_info(fields, space="rectified")
    assert rectified["camera_matrix"] == [[20, 0, 3], [0, 22, 2], [0, 0, 1]]
    assert rectified["distortion"] == []
    assert camera_from_info(fields)[0]["distortion"][0] == 0.1


@pytest.mark.parametrize(
    "name,count", [("plumb_bob", 5), ("rational_polynomial", 8), ("equidistant", 4)]
)
def test_recognized_models_can_declare_ideal_empty_distortion(name, count):
    fields = camera_to_info(model())
    fields.update(distortion_model=name, d=[])
    decoded, _ = camera_from_info(fields)
    assert decoded["distortion"] == [0.0] * count
    assert decoded["ros2"]["camera_info"]["d"] == []


@pytest.mark.parametrize(
    "change",
    [
        {"width": 0},
        {"binning_x": -1},
        {"header": {**HEADER, "frame_id": ""}},
        {"k": [0] * 9},
        {"d": [True] * 5},
        {"distortion_model": "unknown"},
        {"r": [0] * 8},
        {"p": [0] * 11},
        {"k": [np.nan] * 9},
        {"roi": {"x_offset": 0, "y_offset": 0, "width": 7, "height": 4, "do_rectify": False}},
    ],
)
def test_invalid_calibration_rejected(change):
    fields = camera_to_info(model())
    with pytest.raises(ValueError):
        camera_from_info({**fields, **change})


@pytest.mark.parametrize("kind", ["rotation", "baseline", "rectified_roi"])
def test_rectification_cannot_silently_relabel_physical_frames(kind):
    fields = camera_to_info(model())
    if kind == "rotation":
        fields["r"] = [0, -1, 0, 1, 0, 0, 0, 0, 1]
    elif kind == "baseline":
        fields["p"][3] = -0.2
    else:
        fields["roi"]["do_rectify"] = True
    with pytest.raises(ValueError):
        camera_from_info(fields, space="rectified")
    camera_from_info(fields, space="raw")  # Unused R/P remain inspectable in the original message.


@pytest.mark.parametrize("lens,d", [("omnidir", [0] * 4), ("pinhole", [0] * 12)])
def test_unsupported_ros_distortion_encoding_is_explicit(lens, d):
    source = model(distortion=d, lens=lens)
    with pytest.raises(ValueError):
        camera_to_info(source)


def test_registered_group_converts_metric_depth_without_mutating_source():
    source = group()
    before = source.items[1].value.copy()
    block = AssociateROSRGBD()
    result = block.execute({"group": source})
    assert result is not None, block.error
    assert result.passed and result.outputs["depth"].units == "m"
    assert result.outputs["info"]["valid_depth_samples"] == 23
    assert not result.outputs["info"]["clock_domain_verified"]
    assert np.isnan(result.outputs["depth"][0, 0])
    np.testing.assert_allclose(result.outputs["depth"][1:], 1)
    np.testing.assert_array_equal(source.items[1].value, before)
    assert (
        result.outputs["color"].metadata["spatial"] == result.outputs["depth"].metadata["spatial"]
    )
    cloud = (
        builtin_registry()
        .create("3d.depth_cloud", parameters={"sample_step": 1})
        .execute(
            {
                "depth": result.outputs["depth"],
                "camera": result.outputs["camera"],
                "image": result.outputs["color"],
            }
        )
    )
    assert cloud is not None and len(cloud.outputs["cloud"].points) == 23
    np.testing.assert_allclose(cloud.outputs["cloud"].points[0], [-0.1, -1 / 12, 1], rtol=1e-6)


@pytest.mark.parametrize(
    "kind", ["frame", "stamp", "ticket", "size", "transform", "distortion", "missing_ticket"]
)
def test_group_mismatches_emit_no_results(kind):
    source = group()
    image = source.items[0].value
    if kind == "frame":
        image.metadata["ros2"]["header"]["frame_id"] = "other"
    elif kind == "stamp":
        image.metadata["ros2"]["header"]["stamp"]["nanosec"] += 1
    elif kind == "ticket":
        source.items[0].payload["ticket"]["value"] = "other"
    elif kind == "missing_ticket":
        source.items[0].payload.pop("ticket")
    elif kind == "size":
        image.metadata["ros2"]["width"] += 1
    elif kind == "transform":
        image.metadata["spatial"] = frame([6, 4], "camera")
        image.metadata["spatial"]["to_root"][0][2] = 1
    else:
        source.items[2].value["distortion"] = [0.1, 0, 0, 0, 0]
    block = AssociateROSRGBD()
    assert block.execute({"group": source}) is None
    assert not block.current_emissions


def test_zero_stamp_mode_and_nan_depth_are_reported_without_false_timing_claims():
    source = group(zero=True)
    assert AssociateROSRGBD().execute({"group": source}) is None
    block = AssociateROSRGBD(parameters={"allow_zero_stamps": True})
    source.items[1].value[:] = 0
    result = block.execute({"group": source})
    assert result is not None and not result.passed
    assert not result.outputs["info"]["timestamps_nonzero"]
    assert result.fail_reasons and np.isnan(result.outputs["depth"]).all()


def test_nanosecond_tolerance_and_float_depth_units_are_explicit():
    source = group()
    source.items[0].value.metadata["ros2"]["header"]["stamp"]["nanosec"] += 5
    image = source.items[1].value.astype(np.float32)
    image.metadata["ros2"]["encoding"] = "32FC1"
    source.items[1].value = image
    block = AssociateROSRGBD(parameters={"max_skew_ms": 0.000005, "float_depth_units": "mm"})
    result = block.execute({"group": source})
    assert result is not None, block.error
    assert result.outputs["info"]["header_spread_ns"] == 5
    assert result.outputs["depth"][1, 1] == 1


def test_depth_only_group_and_partial_snapshot_validation():
    block = AssociateROSRGBD(parameters={"use_color": False})
    result = block.execute({"group": group(color=False)})
    assert result is not None, block.error
    assert result.outputs["color"] is NO_VALUE
    partial = group()
    partial = FlowGroup(partial.items[:2], partial.labels[:2])
    assert block.execute({"group": partial}) is None


@pytest.mark.parametrize(
    "parameters",
    [
        {"max_skew_ms": True},
        {"uint16_m_per_sample": "0.001"},
        {"float_depth_units": "inches"},
        {"allow_zero_stamps": "yes"},
        {"color_label": "input_2"},
        {"depth_label": ""},
    ],
)
def test_invalid_association_configuration_is_rejected_before_processing(parameters):
    with pytest.raises(ValueError):
        AssociateROSRGBD(parameters=parameters).validate_parameters()


def test_camera_blocks_registry_xml_and_original_message_are_preserved(tmp_path):
    registry = builtin_registry()
    source = camera_to_info(model())
    decoder = DecodeROSCamera()
    result = decoder.execute({"message": source})
    assert result.outputs["message"] == source
    result.outputs["message"]["k"][0] += 1
    assert source["k"][0] == 10
    assert EncodeROSCamera().execute({"camera": model()}).outputs["message"]["header"] == HEADER
    graph = Graph()
    for cls in (DecodeROSCamera, EncodeROSCamera, AssociateROSRGBD):
        block = graph.add(registry.create(cls.type_id))
        block.validate_parameters()
        assert block_entry(block).path == ("Industrial integration", "ROS 2")
    path = tmp_path / "ros_calibration.xml"
    graph.save(path)
    loaded = Graph.load(path, registry)
    assert len(loaded.blocks) == 3
    loaded.close()
    graph.close()
