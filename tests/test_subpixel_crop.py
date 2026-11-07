"""Check fractional sampling, border coverage, and graph coordinate contracts."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import ValuePacket, frame, metadata
from opencv_dataflow.tickets import TicketClock

TYPE_ID = "cv.crop_subpixel"
_PROJECT = Path(__file__).parents[1] / "examples/opencv tools/subpixel_crop/subpixel_crop.json"


def test_bilinear_ramp_and_transform_for_even_width_and_odd_height():
    y, x = np.indices((9, 10), dtype=np.float32)
    pixels = 2 * x + 11 * y
    block = builtin_registry().create(
        TYPE_ID,
        parameters={
            "width": 4,
            "height": 3,
            "center_mode": "custom",
            "center_x": 4.125,
            "center_y": 3.25,
        },
    )
    block.exposed_outputs.append("result_graphics")
    result = block.execute({"image": pixels})
    assert result and result.passed, block.error
    yy, xx = np.indices((3, 4), dtype=np.float32)
    np.testing.assert_array_equal(result.outputs["image"], 2 * (xx + 2.625) + 11 * (yy + 2.25))
    np.testing.assert_array_equal(result.outputs["valid_mask"], 255)
    matrix = [[1, 0, 2.625], [0, 1, 2.25], [0, 0, 1]]
    assert result.outputs["geometry"]["output_to_input"] == matrix
    assert metadata(result.outputs["image"])["spatial"]["to_root"] == matrix
    assert metadata(result.outputs["valid_mask"])["spatial"]["to_root"] == matrix
    outline, cross = result.outputs["result_graphics"].layers[0].objects
    assert outline.points[0] == (2.625, 2.25) and cross.points[0] == (4.125, 3.25)


def test_replicated_border_coverage_and_raw_float_promotion():
    y, x = np.indices((5, 7))
    image = (10 * y + 3 * x).astype(np.uint8)
    inputs = {"image": image, "center": [[0.25, 0.75]]}
    registry = builtin_registry()
    parameters = {"width": 4, "height": 4, "output_depth": "float32"}
    strict = registry.create(TYPE_ID, parameters=parameters)
    assert strict.execute(inputs) is None and "Strict mode" in strict.error
    crop = registry.create(TYPE_ID, parameters={**parameters, "bounds_policy": "replicate"})
    result = crop.execute(inputs)
    assert result and result.passed, crop.error
    y, x = np.indices((4, 4), dtype=np.float32)
    expected = 10 * np.clip(y - 0.75, 0, 4) + 3 * np.clip(x - 1.25, 0, 6)
    np.testing.assert_array_equal(result.outputs["image"], expected)
    assert result.outputs["image"].dtype == np.float32
    np.testing.assert_array_equal(result.outputs["valid_mask"] != 0, (x >= 2) & (y >= 1))
    assert crop.execute({"image": image, "center": [6.25, 4]}) is None
    assert "outermost input pixel centers" in crop.error


def test_uint16_rgba_preserves_stored_color_and_alpha_channels():
    y, x = np.indices((8, 10))
    bases = np.array([50000, 12000, 36000, 62500])
    scales = np.array([4, 8, 12, 16])
    pixels = (bases + (x + 2 * y)[..., None] * scales).astype(np.uint16)
    source = ImageArray(pixels, "rgba")
    block = builtin_registry().create(TYPE_ID, parameters={"width": 3, "height": 2})
    result = block.execute({"image": source, "center": [5.25, 4.25]})
    assert result and result.passed, block.error
    output = result.outputs["image"]
    assert output.dtype == np.uint16 and image_format(output) == "rgba"
    y, x = np.indices((2, 3))
    expected = bases + (x + 4.25 + 2 * (y + 3.75))[..., None] * scales
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("shape", ((1, 1), (1, 7), (8, 1), (2, 2)))
def test_large_replicated_patch_from_tiny_source(shape):
    rows, columns = np.indices(shape)
    image = (17 + 7 * rows + 3 * columns).astype(np.int16)
    block = builtin_registry().create(
        TYPE_ID,
        parameters={
            "width": 13,
            "height": 11,
            "bounds_policy": "replicate",
            "output_depth": "float32",
        },
    )
    result = block.execute({"image": image, "center": [0, 0]})
    assert result and result.passed, block.error
    rows, columns = np.indices((11, 13))
    expected = (
        17 + 7 * np.clip(rows - 5, 0, shape[0] - 1) + 3 * np.clip(columns - 6, 0, shape[1] - 1)
    )
    np.testing.assert_array_equal(result.outputs["image"], expected)


def test_pose_origin_is_mapped_to_selected_pyramid_level():
    registry = builtin_registry()
    size = [80, 64]
    ticket = TicketClock(session="test").issue("camera", "fixed", 10)
    source = ImageArray(
        np.zeros((64, 80), np.uint8),
        "gray",
        {"spatial": frame(size, "camera"), "ticket": ticket},
    )
    pose_block = registry.create(
        "pose.define", parameters={"x": 39.25, "y": 29.5, "angle_degrees": 67, "scale": 1.3}
    )
    pose = pose_block.execute({"image": source}).outputs["pose"]
    builder = registry.create("cv.image_pyramid", parameters={"levels": 2})
    pyramid = builder.execute({"image": source}).outputs["images"]
    crop = registry.create(TYPE_ID, parameters={"width": 5, "height": 3})
    crop.input_levels["image"] = "1"
    result = crop.execute({"image": pyramid, "pose": pose})
    assert result and result.passed, crop.error
    payload = metadata(result.outputs["image"])
    assert payload["ticket"] == ticket
    assert payload["pyramid"]["level"] == 1
    assert payload["spatial"]["to_root"] == [[2, 0, 35.25], [0, 2, 27.5], [0, 0, 1]]
    assert payload["pyramid"]["to_largest"] == payload["spatial"]["to_root"]
    np.testing.assert_array_equal(
        np.asarray(payload["spatial"]["to_root"]) @ [2, 1, 1], [39.25, 29.5, 1]
    )


def test_invalid_center_inputs_and_output_size_are_rejected():
    source = ImageArray(np.zeros((12, 14), np.uint8), "gray", {"spatial": frame([14, 12], "a")})
    block = builtin_registry().create(TYPE_ID, parameters={"width": 3, "height": 3})
    wrong_frame = ValuePacket([5, 5], {"spatial": frame([14, 12], "b")})
    for values, message in (
        ({"center": wrong_frame}, "coordinate frame"),
        ({"center": [5, 5], "pose": {}}, "either a center or a pose"),
        ({"center": [[5, 5], [6, 6]]}, "finite [x, y] point"),
    ):
        assert block.execute({"image": source, **values}) is None
        assert message in block.error
    large = builtin_registry().create(TYPE_ID, parameters={"width": 8192, "height": 4096})
    assert large.execute({"image": source}) is None
    assert "16,777,216" in large.error
    block.parameters["output_depth"] = "float32"
    hsv = ImageArray(np.zeros((12, 14, 3), np.uint8), "hsv")
    assert block.execute({"image": hsv}) is None
    assert "Convert HSV or Lab" in block.error


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_saved_project_remaps_the_subpixel_center_and_waits_after_rejection(profile):
    graph = Graph.load(_PROJECT, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    match, patch, mapped = (graph.blocks[k] for k in ("match", "patch", "remap_center"))
    center = match.last_result.outputs["center"][0]
    assert center == pytest.approx([102.25, 64.375], abs=0.01)
    assert patch.last_result.outputs["image"].shape == (31, 37)
    assert mapped.last_result.outputs["points"][0] == pytest.approx(center, abs=1e-5)
    match.parameters["min_score"] = 1
    graph.run()
    assert match.state == "fail" and patch.input_rejected
    assert mapped.state == "inactive" and mapped.last_result is None
