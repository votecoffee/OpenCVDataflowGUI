"""Verify native line clipping, collection coordinates, and downstream drawing."""

import copy
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.objects import collection, line_item
from opencv_dataflow.payloads import frame, metadata, moved_payload

TYPE_ID = "collection.clip_lines"
_PROJECT = (
    Path(__file__).parents[1] / "examples/geometry/clip_line_segments/clip_line_segments.json"
)


def _rectangle(size, rectangle, spatial):
    return {
        "model": "image_region_v1",
        "shape": "rectangle",
        "image_size": size,
        "rectangle": rectangle,
        "coordinate_context": spatial,
    }


def test_keeps_crossing_span_and_preserves_an_inside_line_exactly():
    image = np.zeros((80, 120), np.uint8)
    spatial = frame([120, 80], "camera")
    crossing = line_item([[95.5, 32.25], [12.25, 32.25]], 0)
    crossing["id"] = "long edge"
    crossing["metrics"].update(strength=82, coverage=0.9)
    inside = line_item([[32.0001, 24.0001], [46.25, 41.25]], 1)
    outside = line_item([[0, 1], [10, 1]], 2)
    contact = line_item([[20, 10], [30, 20]], 3)
    items = [crossing, inside, outside, contact]
    data = collection("lines", items, image, {"spatial": spatial})
    original = copy.deepcopy(data)
    region = _rectangle([120, 80], [30, 20, 30, 25], spatial)

    block = builtin_registry().create(TYPE_ID, parameters={"minimum_length": 1})
    block.exposed_outputs.append("result_graphics")
    result = block.execute({"collection": data, "region": region})
    assert result and result.passed, block.error
    assert data == original
    outputs = result.outputs
    assert outputs["count"] == 2 and outputs["clipped_count"] == 1
    kept, unmodified = outputs["collection"]["items"]
    assert kept["id"] == "long edge"
    assert kept["geometry"]["points"] == [[59, 32.25], [30, 32.25]]
    assert kept["metrics"]["length"] == 29 and kept["metrics"]["center_x"] == 44.5
    assert kept["metrics"]["strength"] == 82 and kept["metrics"]["coverage"] is None
    assert unmodified == inside
    assert outputs["rejected"]["items"] == [outside, contact]
    assert outputs["rejected_count"] == 2
    (layer,) = outputs["result_graphics"].layers
    assert layer.payload["spatial"] == spatial
    assert len(layer.objects) == 2

    block.parameters["minimum_length"] = 0
    result = block.execute({"collection": data, "region": region})
    assert result and result.outputs["count"] == 3
    assert result.outputs["collection"]["items"][-1]["geometry"]["points"] == [[30, 20], [30, 20]]


def test_clips_window_to_image_and_keeps_collection_origin():
    image = np.zeros((8, 10), np.uint8)
    spatial = frame([10, 8], "camera")
    offset_frame = moved_payload(
        {"spatial": spatial}, np.array([[1, 0, -100], [0, 1, -200], [0, 0, 1]]), [10, 8]
    )
    item = line_item([[98, 203.5], [109, 203.5]], 0)
    data = collection("lines", [item], image, offset_frame, origin=(100, 200))
    region = _rectangle([10, 8], [-2, 2, 8, 10], spatial)
    block = builtin_registry().create(TYPE_ID, parameters={"minimum_length": 5})

    result = block.execute({"collection": data, "region": region})
    assert result and result.passed, block.error
    clipped = result.outputs["collection"]
    assert clipped["origin"] == [100, 200] and clipped["context"] == offset_frame
    (kept,) = clipped["items"]
    assert kept["geometry"]["points"] == [[100, 203.5], [105, 203.5]]
    assert kept["metrics"]["length"] == 5
    assert kept["metrics"]["width"] == 6
    assert kept["metrics"]["center_y"] == 203.5

    result = block.execute({"collection": data})
    assert result and result.outputs["collection"]["items"][0]["metrics"]["length"] == 9
    region["rectangle"] = [20, 20, 1, 1]
    result = block.execute({"collection": data, "region": region})
    assert result and result.passed and result.outputs["count"] == 0
    assert result.outputs["rejected"]["items"] == [item]


def test_rejects_regions_with_incompatible_shape_or_frame():
    image = np.zeros((8, 10), np.uint8)
    spatial = frame([10, 8], "camera")
    item = line_item([[0, 3], [9, 3]], 0)
    data = collection("lines", [item], image, {"spatial": spatial})
    block = builtin_registry().create(TYPE_ID)
    region = _rectangle([10, 8], [0, 0, 10, 8], frame([10, 8], "other"))
    assert block.execute({"collection": data, "region": region}) is None
    assert "coordinate frame" in block.error
    region.update(coordinate_context=spatial, invert=True)
    assert block.execute({"collection": data, "region": region}) is None
    assert "non-inverted rectangle" in block.error


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_example_measures_and_draws_only_the_visible_span(profile):
    graph = Graph.load(_PROJECT, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    source = graph.blocks["source"].last_result.outputs["image"]
    outputs = graph.blocks["clip"].last_result.outputs
    assert outputs["clipped_count"] == outputs["rejected_count"] == 1
    (kept,) = outputs["collection"]["items"]
    assert kept["metrics"]["length"] == 51
    assert outputs["collection"]["context"]["spatial"] == metadata(source)["spatial"]
    mask = graph.blocks["draw"].last_result.outputs["mask"]
    assert cv2.countNonZero(mask) == 52
    assert mask[35, 40] == mask[35, 91] == 255
    assert mask[35, 39] == mask[35, 92] == 0
