"""Verify bounded mosaic windows against the dense renderer and source coordinates."""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow import tiled_mosaic
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import FlowGroup, ValuePacket, frame, metadata

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/tiled_scan_mosaic"


def _scene(width, height=128):
    rng = np.random.default_rng(1024)
    image = cv2.GaussianBlur(rng.integers(0, 256, (height, width), np.uint8), (3, 3), 0)
    for x in range(10, width, 31):
        cv2.circle(image, (x, 32 + x % 65), 5 + x % 7, 240, 2)
    return image


def _group(images, *, spatial=False):
    return FlowGroup(
        tuple(
            ValuePacket(
                image, {"spatial": frame(image.shape[1::-1], f"view-{i}")} if spatial else {}
            )
            for i, image in enumerate(images)
        ),
        tuple(f"view-{i}" for i in range(len(images))),
    )


def _layout(group, masks=None):
    block = builtin_registry().create("cv.register_scan_layout")
    inputs = {"views": group}
    if masks is not None:
        inputs["masks"] = masks
    result = block.execute(inputs)
    assert result is not None and result.passed, block.error or getattr(result, "message", "")
    return result.outputs["layout"]


def _render(group, layout, *, masks=None, parameters=None, index=None):
    block = builtin_registry().create("cv.render_mosaic_tile", parameters=parameters or {})
    inputs = {"views": group, "layout": layout}
    if masks is not None:
        inputs["masks"] = masks
    if index is not None:
        inputs["tile_index"] = index
    result = block.execute(inputs)
    assert result is not None and result.passed, block.error
    return result.outputs


@pytest.mark.parametrize("blend", ["prefer_first", "prefer_last", "feather"])
@pytest.mark.parametrize("kind", ["gray8", "rgb16", "bgra_float", "gray_float64"])
@pytest.mark.parametrize("interpolation", ["nearest", "linear"])
def test_tiles_reassemble_to_dense_pixels_and_masks_across_boundaries(blend, kind, interpolation):
    scene = _scene(350)
    raw = [scene[:, x : x + 180].copy() for x in (0, 80, 160)]
    if kind == "rgb16":
        images = [
            ImageArray(cv2.cvtColor(v.astype(np.uint16) * 200, cv2.COLOR_GRAY2RGB), "rgb")
            for v in raw
        ]
    elif kind == "bgra_float":
        images = [cv2.cvtColor(v.astype(np.float32) / 255, cv2.COLOR_GRAY2BGRA) for v in raw]
    elif kind == "gray_float64":
        images = [v.astype(np.float64) / 255 for v in raw]
    else:
        images = raw
    # Different overlap intensities exercise both source priorities and feather weights.
    images[-1] = images[-1].copy()
    signal = images[-1][..., :3] if images[-1].ndim == 3 else images[-1]
    signal += 0.01 if images[-1].dtype.kind == "f" else 5
    masks = [np.full(v.shape[:2], 255, np.uint8) for v in raw]
    masks[1][35:85, 25:85] = 0
    masks[0][60:75, 10:30] = 0
    group, selections = _group(images), _group(masks)
    settings = {"interpolation": interpolation, "blend": blend, "feather_width": 19}
    block = builtin_registry().create("cv.stitch_scan_sequence", parameters=settings)
    dense = block.execute({"views": group, "masks": selections})
    assert dense is not None and dense.passed, block.error
    layout = _layout(group, selections)
    assert layout["image_size"] == dense.outputs["layout"]["image_size"]
    for actual, expected in zip(layout["views"], dense.outputs["layout"]["views"], strict=True):
        np.testing.assert_allclose(
            actual["source_to_mosaic"], expected["source_to_mosaic"], atol=1e-9
        )
    width, height = layout["image_size"]
    assembled = {
        name: np.zeros_like(dense.outputs[name]) for name in ("image", "coverage", "overlap")
    }
    side = 64
    for index in range(math.ceil(width / side) * math.ceil(height / side)):
        output = _render(
            group, layout, masks=selections, parameters={"tile_size": side, **settings}, index=index
        )
        tile = output["tile"]
        x, y = tile["origin"]
        w, h = tile["image_size"]
        assert 0 < w <= side and 0 < h <= side
        for name in assembled:
            assembled[name][y : y + h, x : x + w] = output[name]
        assert output["image"].dtype == images[0].dtype
    # Window-relative inversion may change the final interpolation rounding.
    tolerance = 0 if interpolation == "nearest" else (1 if images[0].dtype.kind == "u" else 2e-5)
    if interpolation == "nearest" and blend == "feather" and images[0].dtype.kind == "f":
        tolerance = 1e-8
    np.testing.assert_allclose(assembled["image"], dense.outputs["image"], atol=tolerance, rtol=0)
    for name in ("coverage", "overlap"):
        np.testing.assert_array_equal(assembled[name], dense.outputs[name])


def test_registration_handles_more_than_eight_views_without_dense_allocation(monkeypatch):
    scene = _scene(980)
    group = _group([scene[:, x : x + 180] for x in range(0, 801, 80)])

    def no_render(*args, **kwargs):
        pytest.fail("Layout registration must not render pixels")

    monkeypatch.setattr(tiled_mosaic, "warp_mosaic_view", no_render)
    layout = _layout(group)
    assert layout["view_count"] == 11 and len(layout["views"]) == 11
    assert len(layout["pairs"]) == 10
    for i, view in enumerate(layout["views"]):
        relative = np.linalg.solve(layout["views"][0]["source_to_mosaic"], view["source_to_mosaic"])
        np.testing.assert_allclose(relative, [[1, 0, i * 80], [0, 1, 0], [0, 0, 1]], atol=0.2)


def test_source_frame_and_wired_index_follow_tile_window():
    scene = _scene(350)
    group = _group([scene[:, x : x + 180] for x in (0, 80, 160)], spatial=True)
    layout = _layout(group)
    result = _render(group, layout, parameters={"tile_size": 64, "tile_index": 0}, index=7)
    tile = result["tile"]
    assert tile["index"] == 7
    payload = metadata(result["image"])
    spatial = payload["spatial"]
    assert spatial["root_id"] == "view-0"
    assert spatial["image_size"] == tile["image_size"]
    anchor = np.asarray(layout["views"][0]["source_to_mosaic"])
    offset = np.eye(3)
    offset[:2, 2] = tile["origin"]
    np.testing.assert_allclose(spatial["to_root"], np.linalg.inv(anchor) @ offset, atol=1e-9)
    for view in tile["views"]:
        np.testing.assert_allclose(
            offset @ view["source_to_tile"], view["source_to_mosaic"], atol=1e-9
        )
    assert metadata(result["coverage"])["spatial"] == spatial
    assert payload["mosaic"]["mosaic_size"] == layout["image_size"]


def test_rendering_large_virtual_canvas_allocates_only_the_requested_window(monkeypatch):
    scene = _scene(350)
    group = _group([scene[:, x : x + 180] for x in (0, 80, 160)])
    layout = _layout(group)
    # An affine layout covers a 34,000 x 12,800 grid from small source views.
    for record in layout["views"]:
        matrix = np.asarray(record["source_to_mosaic"])
        matrix[:2] *= 100
        record["source_to_mosaic"] = matrix.tolist()
    pixels = [entry.value for entry in group.items]
    size, shift = tiled_mosaic.mosaic_geometry(
        pixels, [np.asarray(r["source_to_mosaic"]) for r in layout["views"]]
    )
    for record in layout["views"]:
        record["source_to_mosaic"] = (shift @ record["source_to_mosaic"]).tolist()
    layout["image_size"] = list(size)
    assert min(size) > 12000 and max(size) > 32767
    native = tiled_mosaic.warp_mosaic_view
    calls = []

    def bounded(image, matrix, window, interpolation):
        assert max(window) <= 64
        calls.append(window)
        return native(image, matrix, window, interpolation)

    monkeypatch.setattr(tiled_mosaic, "warp_mosaic_view", bounded)
    result = _render(group, layout, parameters={"tile_size": 64}, index=17)
    assert result["image"].shape[:2] == (64, 64)
    assert result["tile"]["covered_pixels"] > 0
    assert calls
    assert max(size) * min(size) * 15 > 256 * 1024**2


@pytest.mark.parametrize("change", ["model", "grid", "labels", "size", "singular", "nonfinite"])
def test_malformed_or_mismatched_layouts_fail_without_image_emission(change):
    scene = _scene(350)
    group = _group([scene[:, x : x + 180] for x in (0, 80, 160)])
    layout = _layout(group)
    if change == "model":
        layout["model"] = "future_layout"
    elif change == "grid":
        layout["views"][0]["source_size"][0] += 1
    elif change == "labels":
        layout["views"][0]["label"] = "another_camera"
    elif change == "size":
        layout["image_size"][0] += 1
    elif change == "singular":
        layout["views"][0]["source_to_mosaic"] = np.zeros((3, 3)).tolist()
    else:
        layout["views"][0]["source_to_mosaic"][0][0] = float("nan")
    block = builtin_registry().create("cv.render_mosaic_tile")
    assert block.execute({"views": group, "layout": layout}) is None
    assert block.error_kind == "invalid"
    assert block.last_result is None


def test_index_bounds_and_workspace_are_validated_before_warp(monkeypatch):
    scene = _scene(350)
    group = _group([scene[:, x : x + 180] for x in (0, 80, 160)])
    layout = _layout(group)

    def no_warp(*args, **kwargs):
        pytest.fail("Invalid settings must fail before a warp")

    monkeypatch.setattr(tiled_mosaic, "warp_mosaic_view", no_warp)
    block = builtin_registry().create("cv.render_mosaic_tile")
    assert block.execute({"views": group, "layout": layout, "tile_index": -1}) is None
    assert block.error_kind == "invalid"
    monkeypatch.setattr(tiled_mosaic, "_MAX_WORKSPACE", 1)
    assert block.execute({"views": group, "layout": layout}) is None
    assert "workspace limit" in block.error


def test_registration_failure_is_a_ticket_failure_and_layout_is_not_renderable():
    blank = np.zeros((64, 64), np.uint8)
    group = _group([blank, blank])
    block = builtin_registry().create("cv.register_scan_layout")
    result = block.execute({"views": group})
    assert result is not None and not result.passed
    assert result.fail_reasons[0]["code"] == "scan_registration_failed"
    renderer = builtin_registry().create("cv.render_mosaic_tile")
    assert renderer.execute({"views": group, "layout": result.outputs["layout"]}) is None
    assert renderer.error_kind == "invalid"


@pytest.mark.local_examples
def test_saved_tiled_example_runs_and_serializes_layout():
    graph = Graph.load(EXAMPLE / "tiled_scan_mosaic.json", builtin_registry())
    try:
        graph.set_run_profile("lean")
        for _ in range(12):
            graph.run()
        layout = graph.blocks["register"].last_result.outputs["layout"]
        assert layout["view_count"] == 12
        assert layout["image_size"][0] > 16384
        for name in ("first_tile", "middle_tile", "last_tile"):
            block = graph.blocks[name]
            assert block.state == "pass", block.error
            assert block.last_result.outputs["tile"]["covered_pixels"] > 0
        serialized = json.loads(json.dumps(layout, allow_nan=False))
        group = graph.blocks["batch"].last_result.outputs["group"]
        assert _render(group, serialized)["tile"]["covered_pixels"] > 0
    finally:
        graph.close()
