"""Verify path-strip geometry, pixel preservation, selection, and metadata."""

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock


def test_horizontal_strip_preserves_rgb16_ticket_and_exact_pixel_mapping():
    pixels = np.arange(12 * 14 * 3, dtype=np.uint16).reshape(12, 14, 3)
    image = ImageArray(pixels, "rgb")
    source_frame = frame([14, 12], "camera")
    ticket = TicketClock().issue("source")
    tool = builtin_registry().create("cv.unwrap_path_strip", parameters={
        "vertices": [[2, 5], [10, 5]], "half_width": 2, "cross_samples": 5,
    })
    result = tool.execute({"image": ValuePacket(image, {
        "spatial": source_frame, "ticket": ticket, "pyramid": {
            "level": 1, "largest_size": [28, 24], "to_largest": np.diag([2, 2, 1]).tolist(),
        },
    })})
    assert result is not None, tool.error
    np.testing.assert_array_equal(result.outputs["image"], pixels[3:8, 2:11])
    assert result.outputs["image"].dtype == np.uint16
    assert image_format(result.outputs["image"]) == "rgb"
    np.testing.assert_array_equal(result.outputs["map_x"], np.tile(np.arange(2, 11), (5, 1)))
    np.testing.assert_array_equal(
        result.outputs["map_y"], np.tile(np.arange(3, 8)[:, None], (1, 9))
    )
    assert np.all(result.outputs["valid_mask"] == 255)
    sampling = result.outputs["sampling"]
    assert sampling["source_spatial"] == source_frame
    assert sampling["strip_spatial"]["root_id"] != source_frame["root_id"]
    assert sampling["along_step"] == sampling["cross_step"] == 1
    assert sampling["center_row"] == 2
    for name in result.outputs:
        assert result.payloads[name]["ticket"] == ticket
        assert result.payloads[name]["spatial"] == sampling["strip_spatial"]
        assert "pyramid" not in result.payloads[name]


def test_strip_excludes_unselected_interpolation_neighbors_and_pixels_outside_frame():
    image = np.full((7, 8), 100, np.uint8)
    mask = np.full_like(image, 255)
    mask[3, 3] = 0
    tool = builtin_registry().create("cv.unwrap_path_strip", parameters={
        "vertices": [[-0.5, 3], [6.5, 3]], "half_width": 1, "cross_samples": 3,
    })
    region = {"model": "image_region_v1", "image_size": [8, 7],
              "shape": "rectangle", "rectangle": [0, 2, 8, 2]}
    result = tool.execute({"image": image, "mask": mask, "region": region})
    assert result is not None, tool.error
    invalid = result.outputs["valid_mask"] == 0
    assert invalid[1].tolist() == [True, False, False, True, True, False, False, False]
    assert invalid[2].all()
    assert np.all(result.outputs["image"][invalid] == 0)
    for name in ("map_x", "map_y"):
        assert np.all(result.outputs[name][invalid] == -1)


def test_closed_strip_uses_corner_normal_without_duplicate_column_and_tracks_frame():
    image = np.arange(256, dtype=np.int32).reshape(16, 16)
    space = {"spatial": frame([16, 16], "scene")}
    path = [[4, 4], [10, 4], [10, 10], [4, 10], [4, 4]]
    tool = builtin_registry().create("cv.unwrap_path_strip", parameters={
        "closed": True, "half_width": 1, "cross_samples": 3, "interpolation": "nearest",
    })
    data = {"image": ValuePacket(image, space), "path": ValuePacket(path, space)}
    result = tool.execute(data)
    assert result is not None, tool.error
    assert result.outputs["image"].shape == (3, 24)
    assert result.outputs["image"].dtype == np.int32
    np.testing.assert_allclose(result.outputs["map_x"][:, 0], 4 + np.arange(-1, 2) / np.sqrt(2))
    np.testing.assert_allclose(result.outputs["map_y"][:, 0], 4 + np.arange(-1, 2) / np.sqrt(2))
    assert list(result.outputs["image"][1, [0, 6, 12, 18, 23]]) == [68, 74, 170, 164, 84]
    expected = cv2.remap(
        image, result.outputs["map_x"], result.outputs["map_y"], cv2.INTER_NEAREST
    )
    np.testing.assert_array_equal(result.outputs["image"], expected)
    first_frame = result.payloads["image"]["spatial"]
    assert tool.execute(data).payloads["image"]["spatial"] == first_frame
    shifted = [[x + 1, y] for x, y in path]
    changed = tool.execute({**data, "path": ValuePacket(shifted, space)})
    assert changed.payloads["image"]["spatial"] != first_frame
    other_frame = {"spatial": frame([16, 16], "other")}
    assert tool.execute({**data, "path": ValuePacket(path, other_frame)}) is None
    assert "coordinate frame" in tool.error


def test_strip_rejects_undefined_tangent_and_marks_empty_coverage_failed():
    image = np.ones((16, 16), np.uint8)
    tool = builtin_registry().create("cv.unwrap_path_strip", parameters={
        "vertices": [[3, 4], [7, 4], [3, 4]], "half_width": 1,
    })
    assert tool.execute({"image": image}) is None
    assert "doubles back" in tool.error
    tool.parameters["vertices"] = [[3, 4], [7, 4]]
    result = tool.execute({"image": image, "mask": np.zeros_like(image)})
    assert not result.passed and not result.outputs["valid_mask"].any()
    assert result.fail_reasons[0]["code"] == "path_strip_empty_coverage"


def test_strip_enforces_odd_cross_sample_count():
    tool = builtin_registry().create("cv.unwrap_path_strip", parameters={"cross_samples": 4})
    assert tool.execute({"image": np.ones((16, 16), np.uint8)}) is None
    assert "odd integer" in tool.error
