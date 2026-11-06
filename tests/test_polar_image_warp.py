"""Verify native polar sampling, validity and frame provenance."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import frame, metadata, moved_payload
from opencv_dataflow.tickets import TicketClock


def test_unwrap_clockwise_with_wired_center_and_radius_and_exact_labels():
    labels = np.arange(49, dtype=np.int32).reshape(7, 7) + 2**30
    labels[3, 1] = -1
    block = builtin_registry().create(
        "cv.warp_polar", parameters={"interpolation": "nearest", "width": 3, "height": 4}
    )
    result = block.execute({"image": labels, "center_x": 3, "center_y": 3, "radius": 3})
    assert result and result.passed, block.error
    expected = np.stack((labels[3, 3:6], labels[3:6, 3], labels[3, 3:0:-1], labels[3:0:-1, 3]))
    np.testing.assert_array_equal(result.outputs["image"], expected)
    assert result.outputs["image"].dtype == np.int32
    assert result.outputs["valid_fraction"] == 1


def test_semilog_radial_samples_preserve_rgb_order_and_uint16_depth():
    ramp = np.tile(np.arange(32, dtype=np.uint16), (32, 1))
    rgb = ImageArray(np.stack((ramp, ramp + 256, ramp + 512), axis=-1), "rgb")
    block = builtin_registry().create(
        "cv.warp_polar",
        parameters={
            "center_mode": "custom",
            "center_x": 16,
            "center_y": 16,
            "radius": 16,
            "width": 4,
            "height": 4,
            "radial_scale": "semilog",
            "interpolation": "nearest",
        },
    )
    result = block.execute({"image": rgb})
    assert result and result.passed, block.error
    assert image_format(result.outputs["image"]) == "rgb"
    assert result.outputs["image"].dtype == np.uint16
    # Native OpenCV 5 uses exp(column * ln(16) / 4) - 1 for source radius.
    np.testing.assert_array_equal(result.outputs["image"][0], rgb[16, [16, 17, 19, 23]])


def test_input_mask_and_region_restrict_only_validity_and_can_report_empty():
    pixels = np.full((7, 7), 18, np.uint8)
    mask = np.ones_like(pixels)
    mask[3, 4] = 0
    region = {
        "model": "image_region_v1",
        "image_size": [7, 7],
        "shape": "rectangle",
        "rectangle": [4, 3, 2, 1],
    }
    block = builtin_registry().create(
        "cv.warp_polar", parameters={"width": 3, "height": 4, "radius": 3}
    )
    result = block.execute({"image": pixels, "mask": mask, "region": region})
    assert result and result.passed, block.error
    np.testing.assert_array_equal(result.outputs["image"], 18)
    valid = np.zeros((4, 3), dtype=np.uint8)
    valid[0, 2] = 255
    np.testing.assert_array_equal(result.outputs["valid_mask"], valid)
    assert result.outputs["valid_fraction"] == pytest.approx(1 / 12)
    empty = block.execute({"image": pixels, "mask": np.zeros_like(mask)})
    assert empty and not empty.passed, block.error
    assert empty.fail_reasons[0]["code"] == "polar_empty_coverage"


def test_rewrap_restores_ticket_and_pyramid_frame_and_rejects_changed_grid():
    registry = builtin_registry()
    matrix = [[2, 0, 3], [0, 2, 5], [0, 0, 1]]
    spatial = moved_payload({"spatial": frame([32, 32], "camera")}, matrix, [8, 8])["spatial"]
    pyramid = {"level": 1, "largest_size": [32, 32], "to_largest": matrix}
    ticket = TicketClock(session="test").issue("camera", "fixed", "a")
    image = ImageArray(
        np.full((8, 8), 205, dtype=np.uint8),
        "gray",
        {"spatial": spatial, "pyramid": pyramid, "ticket": ticket},
    )
    unwrap = registry.create("cv.warp_polar", parameters={"width": 8, "height": 8, "radius": 3})
    forward = unwrap.execute({"image": image})
    assert forward and forward.passed, unwrap.error
    polar = forward.outputs["image"]
    polar_metadata = metadata(polar)
    assert polar_metadata["spatial"]["root_id"] != "camera"
    assert "pyramid" not in polar_metadata
    assert polar_metadata["ticket"]["value"] == "a"
    assert forward.outputs["mapping"]["cartesian_spatial"] == spatial
    rewrap = registry.create("cv.warp_polar", parameters={"direction": "rewrap"})
    reverse = rewrap.execute({"image": polar, "mask": forward.outputs["valid_mask"]})
    assert reverse and reverse.passed, rewrap.error
    assert metadata(reverse.outputs["image"])["spatial"] == spatial
    assert metadata(reverse.outputs["valid_mask"])["pyramid"] == pyramid
    assert metadata(reverse.outputs["image"])["ticket"]["value"] == "a"
    assert reverse.outputs["image"][3, 3] == 205
    assert reverse.outputs["valid_mask"][0, 0] == reverse.outputs["image"][0, 0] == 0

    crop = registry.create("cv.crop", parameters={"y": 1})
    cropped = crop.execute({"image": polar})
    assert cropped and cropped.passed, crop.error
    assert rewrap.execute({"image": cropped.outputs["image"]}) is None
    assert "original polar grid" in rewrap.error


def test_untagged_rewrap_and_invalid_parameters():
    registry = builtin_registry()
    source = np.full((6, 4), 200, np.uint8)
    reverse = registry.create(
        "cv.warp_polar",
        parameters={"direction": "rewrap", "width": 7.0, "height": 5, "interpolation": "nearest"},
    )
    result = reverse.execute({"image": source})
    assert result and result.passed, reverse.error
    assert result.outputs["image"].shape == (5, 7)
    assert result.outputs["image"][2, 3] == 200
    default_reverse = registry.create("cv.warp_polar", parameters={"direction": "rewrap"})
    assert default_reverse.execute({"image": source}) is None
    assert "Cartesian width and height" in default_reverse.error
    semilog = registry.create("cv.warp_polar", parameters={"radial_scale": "semilog", "radius": 1})
    assert semilog.execute({"image": source}) is None
    assert "above one for semilog" in semilog.error
    large = registry.create("cv.warp_polar", parameters={"width": 6000, "height": 6000})
    assert large.execute({"image": source}) is None
    assert "image limits" in large.error


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_example_unwraps_the_notch_and_rewraps_on_original_grid(profile):
    example = (
        Path(__file__).parents[1]
        / "examples/image_processing/polar_image_warp/polar_image_warp.json"
    )
    graph = Graph.load(example, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    original = graph.blocks["source"].last_result.outputs["image"]
    polar = graph.blocks["unwrap"].last_result.outputs["image"]
    assert polar.shape == (360, 90, 3)
    np.testing.assert_array_equal(polar[28, 65], original[0, 0])
    np.testing.assert_array_equal(polar[90, 65], original[180, 140])
    rewrapped = graph.blocks["rewrap"].last_result.outputs["image"]
    assert rewrapped.shape == original.shape
    assert metadata(rewrapped)["spatial"] == metadata(original)["spatial"]
