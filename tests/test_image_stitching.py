"""Check image stitching coverage, frame mapping, and failure behavior."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.mosaic_seams import _cost_images
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame, metadata, moved_payload
from opencv_dataflow.tickets import TicketClock

TYPE = "cv.stitch_image_pair"


def test_shared_root_reconstructs_crops_and_handles_left_and_top_extension():
    original = np.arange(48, dtype=np.uint16).reshape(6, 8)
    payload = {"spatial": frame([8, 6], "camera")}
    a_to_root = np.array([[1, 0, 2], [0, 1, 1], [0, 0, 1]])
    a = ImageArray(original[1:, 2:], "gray", moved_payload(payload, a_to_root, [6, 5]))
    b = ImageArray(original[:, :5], "gray", moved_payload(payload, np.eye(3), [5, 6]))
    block = builtin_registry().create(TYPE)
    result = block.execute({"image_a": a, "image_b": b})
    assert result is not None and result.passed, block.error
    output = result.outputs["image"]
    assert output.dtype == np.uint16
    expected = original.copy()
    expected[0, 5:] = 0
    np.testing.assert_array_equal(output, expected)
    coverage = np.full((6, 8), 255, np.uint8)
    coverage[0, 5:] = 0
    np.testing.assert_array_equal(result.outputs["coverage"], coverage)
    assert result.outputs["layout"]["overlap_pixels"] == 15
    assert metadata(output)["spatial"] == payload["spatial"]
    assert metadata(result.outputs["coverage"])["spatial"] == payload["spatial"]
    np.testing.assert_array_equal(
        result.outputs["layout"]["views"][0]["source_to_mosaic"], a_to_root
    )


@pytest.mark.parametrize(
    ("blend", "expected"), [("prefer_a", 10), ("prefer_b", 30), ("average", 20)]
)
def test_wired_registration_direction_and_blending_preserve_encoding(blend, expected):
    a = ImageArray(np.full((4, 5, 3), 10, np.uint8), "rgb")
    b = ImageArray(np.full((4, 6, 3), 30, np.uint8), "rgb")
    transform = {
        "source_to_target": [[1, 0, 3], [0, 1, 0], [0, 0, 1]],
        "source_image_size": [6, 4],
        "target_image_size": [5, 4],
    }
    block = builtin_registry().create(TYPE, parameters={"blend": blend})
    result = block.execute({"image_a": a, "image_b": b, "transform": transform})
    assert result is not None and result.passed, block.error
    image = result.outputs["image"]
    assert image_format(image) == "rgb" and image.shape == (4, 9, 3)
    np.testing.assert_array_equal(image[:, :3], 10)
    np.testing.assert_array_equal(image[:, 3:5], expected)
    np.testing.assert_array_equal(image[:, 5:], 30)
    assert result.outputs["layout"]["overlap_pixels"] == 8
    transform["source_image_size"], transform["target_image_size"] = [5, 4], [6, 4]
    assert block.execute({"image_a": a, "image_b": b, "transform": transform}) is None
    assert block.error_kind == "invalid" and "matching sizes" in block.error


def test_linear_sampling_excludes_pixels_with_unselected_neighbors():
    a = np.full((3, 4), 8, np.uint8)
    b = np.tile(np.array([250, 30, 50, 70], np.uint8), (3, 1))
    b_mask = np.full_like(b, 255)
    b_mask[:, 0] = 0
    block = builtin_registry().create(TYPE, parameters={"blend": "prefer_b"})
    transform = [[1, 0, 1.5], [0, 1, 0], [0, 0, 1]]
    inputs = {"image_a": a, "image_b": b, "transform": transform, "mask_b": b_mask}
    result = block.execute(inputs)
    assert result is not None and result.passed, block.error
    np.testing.assert_array_equal(result.outputs["image"][0], [8, 8, 8, 40, 60, 0])
    np.testing.assert_array_equal(result.outputs["coverage"][0], [255, 255, 255, 255, 255, 0])


def test_disjoint_views_withhold_image_but_report_coverage():
    a = np.ones((2, 3), np.uint8)
    block = builtin_registry().create(TYPE)
    inputs = {"image_a": a, "image_b": a, "transform": [[1, 0, 5], [0, 1, 0]]}
    result = block.execute(inputs)
    assert result is not None and not result.passed, block.error
    assert result.outputs["image"] is NO_VALUE
    assert result.outputs["layout"]["overlap_pixels"] == 0
    assert result.fail_reasons[0]["code"] == "stitch_insufficient_overlap"
    block.parameters["min_overlap_pixels"] = 0
    result = block.execute(inputs)
    assert result is not None and result.passed, block.error
    np.testing.assert_array_equal(result.outputs["image"][0], [1, 1, 1, 0, 0, 1, 1, 1])


def test_magnified_view_keeps_the_entire_pixel_footprint():
    a = np.ones((2, 2), np.uint8)
    block = builtin_registry().create(
        TYPE, parameters={"blend": "prefer_b", "interpolation": "nearest"}
    )
    result = block.execute({"image_a": a, "image_b": a * 2, "transform": [[3, 0, 0], [0, 3, 0]]})
    assert result is not None and result.passed, block.error
    assert result.outputs["image"].shape == (6, 6)
    np.testing.assert_array_equal(result.outputs["image"], 2)
    assert result.outputs["layout"]["covered_pixels"] == 36


@pytest.mark.parametrize(
    ("transform", "error"),
    [
        ([[1, 0, 20000], [0, 1, 0]], "dimension limit"),
        ([[1, 0, 0], [0, 1, 0], [1, 0, -1]], "crosses infinity"),
    ],
)
def test_unbounded_canvas_is_rejected(transform, error):
    a = np.zeros((4, 4), np.uint8)
    block = builtin_registry().create(TYPE)
    assert block.execute({"image_a": a, "image_b": a, "transform": transform}) is None
    assert block.error_kind == "invalid" and error in block.error


def test_ticket_policy_requires_matching_views_unless_explicitly_relaxed():
    clock = TicketClock()
    a_ticket, b_ticket = clock.issue("a"), clock.issue("b")
    a = np.zeros((4, 4), np.uint8)
    inputs = {
        "image_a": ValuePacket(a, {"ticket": a_ticket}),
        "image_b": ValuePacket(a, {"ticket": b_ticket}),
        "transform": np.eye(3).tolist(),
    }
    block = builtin_registry().create(TYPE)
    assert block.execute(inputs) is None
    assert block.error_kind == "invalid" and "same ticket" in block.error
    block.parameters["ticket_policy"] = "anchor_a"
    result = block.execute(inputs)
    assert result is not None and result.passed, block.error
    assert metadata(result.outputs["image"])["ticket"] == a_ticket
    assert result.outputs["layout"]["views"][1]["source_ticket"] == b_ticket


def test_saved_example_reconstructs_original_pixel_for_pixel():
    path = (
        Path(__file__).parents[1]
        / "examples/custom tools/stitch_image_pair/stitch_image_pair.json"
    )
    graph = Graph.load(path, builtin_registry())
    try:
        graph.set_run_profile("lean")
        graph.run()
        assert all(block.state == "pass" for block in graph.blocks.values())
        assert graph.blocks["compare"].last_result.outputs["value"] == 0
        layout = graph.blocks["stitch"].last_result.outputs["layout"]
        assert layout["image_size"] == [480, 360]
        assert layout["overlap_pixels"] == 120 * 360
    finally:
        graph.close()


@pytest.mark.parametrize(
    ("project", "scale", "work_factor"),
    [("stitch_seam", 1, 1), ("stitch_large_seam", 8, 2)],
)
def test_saved_graph_cut_selects_original_samples_and_retains_both_marks(
    project, scale, work_factor
):
    folder = Path(__file__).parents[1] / "examples/custom tools/stitch_seam"
    path = folder / f"{project}.json"
    graph = Graph.load(path, builtin_registry())
    try:
        graph.set_run_profile("lean")
        graph.run()
        assert all(block.state == "pass" for block in graph.blocks.values())
        suffix = "_full" if scale > 1 else ""
        a = graph.blocks["left" + suffix].last_result.outputs["image"]
        b = graph.blocks["right" + suffix].last_result.outputs["image"]
        cut = graph.blocks["cut"].last_result.outputs
        average = graph.blocks["average"].last_result.outputs
        assert cut["image"].dtype == np.uint16
        for mask in ("coverage", "overlap"):
            np.testing.assert_array_equal(cut[mask], average[mask])
        overlap = cut["image"][:, 45 * scale : 140 * scale]
        assert np.all((overlap == a[:, 45 * scale :]) | (overlap == b[:, : 95 * scale]))
        for y, x in ((45, 75), (50, 124)):
            point = y * scale, x * scale
            assert cut["image"][point] == 53970 > average["image"][point]
        seam = cut["layout"]["seam"]
        assert seam["overlap_from_existing"] > 0
        assert seam["overlap_from_incoming"] > 0
        assert seam["overlap_pixels"] == 9120 * scale**2
        assert seam["downsample_factor"] == work_factor
        assert seam["work_pixels"] <= 524_288
    finally:
        graph.close()


@pytest.mark.parametrize(("resolution", "factor"), [("auto", 1), ("quarter", 4)])
def test_graph_cut_uses_authoritative_support_and_preserves_hdr_color_and_alpha(resolution, factor):
    a = np.broadcast_to([-1e308, 0, 2, 0.25], (48, 80, 4)).copy()
    b = np.broadcast_to([1e308, 3, 0, 0.75], a.shape).copy()
    mask_a = np.full(a.shape[:2], 255, np.uint8)
    mask_b = np.full_like(mask_a, 255)
    mask_a[5:12, 48:60] = 0
    mask_b[5:12, 33:45] = 0
    block = builtin_registry().create(
        TYPE,
        parameters={
            "blend": "graph_cut",
            "interpolation": "nearest",
            "seam_resolution": resolution,
        },
    )
    result = block.execute(
        {
            "image_a": a,
            "image_b": b,
            "mask_a": mask_a,
            "mask_b": mask_b,
            "transform": [[1, 0, 22], [0, 1, 0]],
        }
    )
    assert result is not None and result.passed, block.error
    output = result.outputs["image"]
    assert output.dtype == np.float64 and image_format(output) == "bgra"
    np.testing.assert_array_equal(output[7, 50], b[7, 28])
    np.testing.assert_array_equal(output[7, 57], [0, 0, 0, 0])
    np.testing.assert_array_equal(output[7, 63], a[7, 63])
    assert result.outputs["coverage"][7, 50:66].tolist() == [255] * 5 + [0] * 5 + [255] * 6
    assert result.outputs["layout"]["seam"]["downsample_factor"] == factor


def test_seam_decimation_averages_only_valid_samples_and_keeps_thin_support():
    a = np.array([[10, 255, 20, 255, 255, 255], [30, 255, 255, 255, 255, 255]], np.uint8)
    mask = np.zeros(a.shape, dtype=bool)
    mask[:, 0] = mask[0, 2] = True
    cost_images, masks = _cost_images((a, a), (mask, mask), (3, 1))
    for cost, support in zip(cost_images, masks, strict=True):
        np.testing.assert_allclose(cost, [[[20, 20, 20], [20, 20, 20], [0, 0, 0]]])
        np.testing.assert_array_equal(support, [[1, 1, 0]])


def test_graph_cut_skips_disjoint_views_and_bounds_native_workspace():
    block = builtin_registry().create(
        TYPE, parameters={"blend": "graph_cut", "min_overlap_pixels": 0, "seam_resolution": "full"}
    )
    small = np.ones((4, 8), np.uint8)
    result = block.execute(
        {"image_a": small, "image_b": small, "transform": [[1, 0, 10], [0, 1, 0]]}
    )
    assert result is not None and result.passed
    assert result.outputs["layout"]["seam"]["overlap_pixels"] == 0
    np.testing.assert_array_equal(result.outputs["image"][0], [1] * 8 + [0, 0] + [1] * 8)
    large = np.ones((720, 720), np.uint8)
    inputs = {"image_a": large, "image_b": large, "transform": np.eye(3).tolist()}
    assert block.execute(inputs) is None
    assert block.error_kind == "invalid" and "Graph-cut work area exceeds" in block.error
