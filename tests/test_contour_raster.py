"""Check raster topology, selected subtrees, source grids and bounded native drawing."""

import copy
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.objects import collection, contour_item
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame, metadata
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.tickets import TicketClock

TYPE_ID = "collection.rasterize_contours"
PROJECT = Path(__file__).parents[1] / "examples/opencv tools/contour_rasterization/project.json"


def scene():
    image = np.zeros((60, 90), np.uint8)
    image[3:53, 3:53] = 255
    image[12:44, 12:44] = 0
    image[20:36, 20:36] = 255
    image[25:30, 25:30] = 0
    image[8:27, 64:84] = 255
    return image


def tree(image=None, payload=None):
    image = scene() if image is None else image
    detector = builtin_registry().create("inspect.contour_hierarchy")
    result = detector.execute({"image": ValuePacket(image, payload or {})})
    assert result is not None, detector.error
    return ValuePacket(result.outputs["collection"], result.payloads["collection"])


def execute(packet=None, *, parameters=None, extra=None):
    block = builtin_registry().create(TYPE_ID, parameters=parameters or {})
    result = block.execute({"collection": tree() if packet is None else packet, **(extra or {})})
    assert result is not None, block.error
    return block, result


def roots(packet):
    return [item for item in packet.value["items"] if item["geometry"]["hierarchy"][3] == -1]


def test_complete_tree_round_trips_holes_islands_and_independent_roots_exactly():
    block, result = execute()
    np.testing.assert_array_equal(result.outputs["mask"], scene())
    assert result.outputs["pixel_count"] == 2087
    assert result.outputs["fraction"] == pytest.approx(2087 / 5400)
    assert result.outputs["info"]["groups"] == 2
    assert result.outputs["info"]["drawn_contours"] == result.outputs["count"] == 5
    assert result.outputs["mask"].dtype == np.uint8
    assert set(np.unique(result.outputs["mask"])) == {0, 255}
    assert result.outputs["image"] is NO_VALUE
    assert result.drawings and result.outputs["overlay"].shape == (60, 90, 3)
    assert "drawContours" in block.documentation


def test_index_and_id_selection_survive_collection_reordering_and_keep_descendants():
    packet = tree()
    part = max(roots(packet), key=lambda item: item["metrics"]["area"])
    expected = scene()
    expected[:, 60:] = 0
    for selection in ("indices", "ids"):
        changed = copy.deepcopy(packet)
        changed.value["items"].reverse()
        key = (
            part["id"]
            if selection == "ids"
            else next(
                index
                for index, item in enumerate(changed.value["items"])
                if item["id"] == part["id"]
            )
        )
        _, result = execute(changed, parameters={"selection": selection}, extra={"items": [key]})
        np.testing.assert_array_equal(result.outputs["mask"], expected)
        assert result.outputs["info"]["selected_ids"] == [part["id"]]
        assert result.outputs["count"] == 1 and result.outputs["info"]["drawn_contours"] == 4


def test_nested_selected_items_are_not_filled_again_but_hole_can_be_a_new_root():
    packet = tree()
    part = max(roots(packet), key=lambda item: item["metrics"]["area"])
    hole = next(
        item for item in packet.value["items"] if item["geometry"]["hierarchy"][3] == part["id"]
    )
    _, parent_only = execute(packet, parameters={"selection": "ids", "items": [part["id"]]})
    _, both = execute(packet, parameters={"selection": "ids", "items": [hole["id"], part["id"]]})
    np.testing.assert_array_equal(both.outputs["mask"], parent_only.outputs["mask"])
    assert both.outputs["info"]["groups"] == 1
    _, hole_only = execute(packet, parameters={"selection": "ids", "items": [hole["id"]]})
    mask = hole_only.outputs["mask"]
    assert mask[15, 15] == 255 and mask[22, 22] == 0 and mask[26, 26] == 255
    assert mask[4, 4] == 0


@pytest.mark.parametrize(("depth", "probe"), [(0, (255, 255)), (1, (0, 0)), (2, (255, 255))])
def test_maximum_relative_depth_limits_holes_and_islands(depth, probe):
    _, result = execute(parameters={"maximum_depth": depth})
    assert (result.outputs["mask"][22, 22], result.outputs["mask"][26, 26]) == probe
    _, filled = execute(parameters={"include_descendants": False})
    assert filled.outputs["mask"][15, 15] == 255


def test_explicit_deeper_selection_outside_a_depth_limit_is_a_separate_group():
    packet = tree()
    part = max(roots(packet), key=lambda item: item["metrics"]["area"])
    hole = next(
        item for item in packet.value["items"] if item["geometry"]["hierarchy"][3] == part["id"]
    )
    island = next(
        item for item in packet.value["items"] if item["geometry"]["hierarchy"][3] == hole["id"]
    )
    _, result = execute(
        packet,
        parameters={
            "selection": "ids",
            "items": [part["id"], island["id"]],
            "maximum_depth": 1,
        },
    )
    assert result.outputs["info"]["groups"] == 2
    assert result.outputs["mask"][22, 22] == 255 and result.outputs["mask"][26, 26] == 0


def test_blob_collections_reconstruct_each_part_without_cancelling_overlaps():
    registry = builtin_registry()
    blobs = registry.create("inspect.blobs", parameters={"min_area": 1, "check_count": False})
    detected = blobs.execute({"image": scene()})
    assert detected is not None, blobs.error
    _, result = execute(
        ValuePacket(detected.outputs["collection"], detected.payloads["collection"])
    )
    np.testing.assert_array_equal(result.outputs["mask"], scene())
    assert result.outputs["info"]["source_kind"] == "blobs"


def test_overlapping_independent_roots_union_while_explicit_parity_cancels_interiors():
    rectangles = [
        np.array([[2, 2], [20, 2], [20, 18], [2, 18]], np.int32),
        np.array([[12, 6], [28, 6], [28, 24], [12, 24]], np.int32),
    ]
    items = [
        contour_item(points, index, [-1, -1, -1, -1]) for index, points in enumerate(rectangles)
    ]
    packet = collection("contours", items, np.zeros((32, 32), np.uint8), {})
    _, result = execute(packet)
    assert result.outputs["mask"][10, 16] == 255
    assert result.payloads["mask"]["spatial"] == frame([32, 32])
    _, parity = execute(packet, parameters={"topology": "even_odd"})
    assert parity.outputs["mask"][10, 16] == 0
    for item in packet["items"]:
        item["geometry"].pop("hierarchy")
    _, solid = execute(packet, parameters={"topology": "solid"})
    np.testing.assert_array_equal(solid.outputs["mask"], result.outputs["mask"])


@pytest.mark.parametrize("damage", ("missing", "cycle", "siblings", "child", "missing_links"))
def test_incomplete_or_inconsistent_topology_fails_visibly(damage):
    packet = tree()
    items = packet.value["items"]
    if damage == "missing":
        child = next(item for item in items if item["geometry"]["hierarchy"][3] >= 0)
        items.remove(child)
        packet.value["count"] -= 1
    elif damage == "cycle":
        items[0]["geometry"]["hierarchy"] = [-1, -1, items[0]["id"], items[0]["id"]]
    elif damage == "siblings":
        items[0]["geometry"]["hierarchy"][0] = items[0]["id"]
    elif damage == "child":
        parent = next(item for item in items if item["geometry"]["hierarchy"][2] >= 0)
        parent["geometry"]["hierarchy"][2] = -1
    else:
        items[0]["geometry"].pop("hierarchy")
    block = builtin_registry().create(TYPE_ID)
    assert block.execute({"collection": packet}) is None
    assert block.error_kind == "invalid" and "Hierarchy" in block.error


@pytest.mark.parametrize("requested", ([True], [-1], [1000], [0, 0], "0"))
def test_invalid_index_selections_are_validation_errors(requested):
    block = builtin_registry().create(TYPE_ID, parameters={"selection": "indices"})
    assert block.execute({"collection": tree(), "items": requested}) is None
    assert block.error_kind == "invalid"


def test_empty_masks_empty_selections_and_one_pixel_contours_are_supported():
    _, empty = execute(tree(np.zeros((8, 9), np.uint8)))
    assert empty.outputs["pixel_count"] == empty.outputs["count"] == 0
    _, selected_none = execute(parameters={"selection": "ids", "items": []})
    assert not selected_none.outputs["mask"].any()
    image = np.zeros((8, 9), np.uint8)
    image[2, 3] = 255
    _, single = execute(tree(image))
    np.testing.assert_array_equal(single.outputs["mask"], image)


@pytest.mark.parametrize("bits", (0, 4, 8))
def test_subpixel_fills_outlines_and_offsets_match_native_reference(bits):
    vertices = np.array([[2.25, 3.75], [20.5, 3.75], [20.5, 18.5], [2.25, 18.5]])
    data = collection(
        "contours", [contour_item(vertices.astype(np.float32), 0)], np.zeros((24, 32), np.uint8), {}
    )
    for mode in ("filled", "outline"):
        params = {
            "topology": "solid",
            "subpixel_bits": bits,
            "mode": mode,
            "offset": [-1.25, 0.5],
            "thickness": 2,
            "line_type": "4_connected",
        }
        _, result = execute(data, parameters=params)
        expected = np.zeros((24, 32), np.uint8)
        points = np.rint((vertices + [-1.25, 0.5]) * (1 << bits)).astype(np.int32)
        if mode == "outline":
            cv2.polylines(expected, [points], True, 255, 2, cv2.LINE_4, bits)
        else:
            cv2.fillPoly(expected, [points], 255, cv2.LINE_4, bits)
        np.testing.assert_array_equal(result.outputs["mask"], expected)


def test_origin_pyramid_ticket_and_original_rgb_depth_survive_rasterization():
    registry = builtin_registry()
    spatial = frame([90, 60], "source")
    spatial["root_size"] = [200, 160]
    spatial["to_root"] = [[2, 0, 5], [0, 2, 7], [0, 0, 1]]
    payload = {
        "spatial": spatial,
        "pyramid": {
            "level": 1,
            "largest_size": [180, 120],
            "to_largest": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        },
        "ticket": TicketClock("raster-ticket").issue("camera"),
        "custom": {"fixture": 133},
    }
    source = ValuePacket(scene(), payload)
    detector = registry.create(
        "inspect.contours",
        parameters={
            "min_area": 0,
            "retrieval": "tree",
            "offset": [100, 200],
        },
    )
    detected = detector.execute({"image": source})
    assert detected is not None, detector.error
    packet = ValuePacket(detected.outputs["collection"], detected.payloads["collection"])
    original = ImageArray(
        np.repeat(scene()[..., None], 3, axis=2).astype(np.uint16) * 200, "rgb", payload
    )
    block = registry.create(TYPE_ID)
    block.exposed_outputs.append("result_graphics")
    result = block.execute({"collection": packet, "image": original})
    assert result is not None, block.error
    np.testing.assert_array_equal(result.outputs["mask"], scene())
    np.testing.assert_array_equal(result.outputs["image"], original)
    assert result.outputs["image"].color_format == "rgb"
    assert result.outputs["image"].dtype == np.uint16
    for name in ("mask", "image", "pixel_count", "info", "overlay"):
        assert result.payloads[name]["spatial"] == spatial
        assert result.payloads[name]["pyramid"] == payload["pyramid"]
        assert result.payloads[name]["ticket"] == payload["ticket"]
        assert result.payloads[name]["custom"] == payload["custom"]
    assert metadata(result.outputs["mask"]) == result.payloads["mask"]
    assert result.outputs["result_graphics"].layers[0].payload["spatial"] == spatial
    assert packet.value["origin"] == [100, 200]


def test_reference_grid_and_known_ticket_checks_require_explicit_alignment():
    packet = tree(payload={"spatial": frame([90, 60], "a")})
    wrong = ImageArray(scene(), "gray", {"spatial": frame([90, 60], "b")})
    block = builtin_registry().create(TYPE_ID)
    assert block.execute({"collection": packet, "image": wrong}) is None
    assert "coordinate frames" in block.error
    block.parameters["alignment"] = "pixel_grid"
    result = block.execute({"collection": packet, "image": wrong})
    assert result is not None and result.payloads["mask"]["spatial"]["root_id"] == "b"
    clock = TicketClock("mismatched")
    packet.payload["ticket"] = clock.issue("a")
    wrong.metadata["ticket"] = clock.issue("b")
    assert block.execute({"collection": packet, "image": wrong}) is None
    assert "different tickets" in block.error
    assert block.execute({"collection": packet, "image": np.zeros((59, 90), np.uint8)}) is None
    assert "dimensions" in block.error


def test_mask_and_region_clip_after_inversion_and_pixel_limits_are_inclusive():
    selection = np.zeros((60, 90), np.uint8)
    selection[10:40, 10:40] = 1
    region = {
        "model": "image_region_v1",
        "shape": "rectangle",
        "image_size": [90, 60],
        "rectangle": [10, 10, 15, 30],
    }
    expected = np.where(selection != 0, 255 - scene(), 0).astype(np.uint8)
    expected[:, 25:] = 0
    count = int(np.count_nonzero(expected))
    block, result = execute(
        parameters={
            "invert": True,
            "check_pixels": True,
            "minimum_pixels": count,
            "maximum_pixels": count,
        },
        extra={"mask": selection, "region": region},
    )
    np.testing.assert_array_equal(result.outputs["mask"], expected)
    assert result.passed and result.outputs["fraction"] == pytest.approx(count / 450)
    block.parameters["maximum_pixels"] = count - 1
    block.parameters["minimum_pixels"] = 0
    failed = block.execute({"collection": tree(), "mask": selection, "region": region})
    assert failed is not None and not failed.passed
    assert failed.fail_reasons[0]["code"] == "raster_pixel_count_out_of_range"


def test_selection_frames_and_empty_eligible_area_are_checked():
    packet = tree(payload={"spatial": frame([90, 60], "a")})
    mask = ImageArray(np.ones((60, 90), np.uint8), "gray", {"spatial": frame([90, 60], "b")})
    block = builtin_registry().create(TYPE_ID)
    assert block.execute({"collection": packet, "mask": mask}) is None
    assert "Selection coordinate frame" in block.error
    _, result = execute(packet, extra={"mask": np.zeros((60, 90), np.uint8)})
    assert result.outputs["pixel_count"] == result.outputs["fraction"] == 0


def test_anonymous_reference_image_keeps_known_collection_coordinates():
    spatial = frame([90, 60], "calibrated camera")
    spatial["to_root"] = [[1, 0, 7], [0, 1, 11], [0, 0, 1]]
    _, result = execute(tree(payload={"spatial": spatial}), extra={"image": scene()})
    assert result.payloads["mask"]["spatial"] == spatial


def test_irregular_sparse_binary_masks_round_trip_exactly():
    rng = np.random.default_rng(133)
    for probability in (0.15, 0.45, 0.85):
        image = (rng.random((32, 48)) < probability).astype(np.uint8) * 255
        _, result = execute(tree(image))
        np.testing.assert_array_equal(result.outputs["mask"], image)


def test_parameters_and_documentation_round_trip_through_xml(tmp_path):
    registry, graph = builtin_registry(), Graph()
    try:
        block = graph.add(
            registry.create(
                TYPE_ID,
                parameters={
                    "selection": "ids",
                    "items": ["part", 2],
                    "topology": "solid",
                    "mode": "outline",
                    "subpixel_bits": 4,
                    "offset": [0.5, -0.5],
                    "invert": True,
                },
            )
        )
        block.documentation = "<p>Inspection mask configuration.</p>"
        block.exposed_outputs.append("result_graphics")
        expected = graph.to_dict()
        path = tmp_path / "raster.xml"
        export_project_xml(expected, path)
        restored = Graph.from_dict(read_project_xml(path), registry)
        try:
            assert restored.to_dict() == expected
        finally:
            restored.close()
    finally:
        graph.close()


@pytest.mark.parametrize("vertices", ([[[1, 2, 3]]], [], [[1, "bad"]], [[1e20, 0]]))
def test_malformed_geometry_is_rejected_before_native_drawing(vertices):
    packet = tree()
    packet.value["items"][0]["geometry"]["points"] = vertices
    block = builtin_registry().create(TYPE_ID)
    assert block.execute({"collection": packet}) is None
    assert block.error_kind == "invalid"


def test_drawing_budget_and_validation_limits_prevent_unbounded_raster_work():
    vertices = np.array([[0, 0], [16383, 0], [16383, 16383], [0, 16383]], np.int32)
    data = collection(
        "contours", [contour_item(vertices, 0, [-1, -1, -1, -1])], np.zeros((1, 1), np.uint8), {}
    )
    data["image_size"] = [8000, 8000]
    item = data["items"][0]
    data["items"] = [{**copy.deepcopy(item), "id": key} for key in range(5)]
    data["count"] = 5
    block = builtin_registry().create(TYPE_ID)
    assert block.execute({"collection": data}) is None
    assert "256M" in block.error
    for updates in (
        {"subpixel_bits": 9},
        {"thickness": 0},
        {"maximum_depth": -2},
        {"offset": [float("inf"), 0]},
        {"minimum_pixels": 5, "maximum_pixels": 4},
    ):
        block = builtin_registry().create(TYPE_ID, parameters=updates)
        assert block.execute({"collection": tree()}) is None
        assert block.error_kind == "invalid"


def test_graphics_off_avoids_preview_work_and_keeps_exact_mask(monkeypatch):
    import opencv_dataflow.contour_raster as implementation

    def unexpected(*args, **kwargs):
        raise AssertionError("Graphics were requested with graphics disabled")

    monkeypatch.setattr(implementation, "overlay_image", unexpected)
    block = builtin_registry().create(TYPE_ID)
    block.set_graphics_mode("off")
    result = block.execute({"collection": tree()})
    assert result is not None, block.error
    np.testing.assert_array_equal(result.outputs["mask"], scene())
    assert not result.graphics and not result.drawings and result.outputs["overlay"] is NO_VALUE


@pytest.mark.local_examples
def test_public_example_and_project_round_trip(tmp_path):
    registry = builtin_registry()
    graph = Graph.load(PROJECT, registry)
    try:
        graph.run()
        assert all(block.state == "pass" for block in graph.processing_blocks())
        for name in ("all", "part", "solid"):
            assert graph.blocks[f"verify_{name}"].last_result.outputs["value"] == 0
        destination = tmp_path / "project.json"
        graph.save(destination)
        loaded = Graph.load(destination, registry)
        try:
            assert loaded.blocks["part"].parameters == graph.blocks["part"].parameters
            assert loaded.blocks["all"].documentation == graph.blocks["all"].documentation
        finally:
            loaded.close()
    finally:
        graph.close()
