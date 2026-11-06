"""Verify exact repetition, independent grid coordinates, and pattern matching."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import frame, metadata
from opencv_dataflow.tickets import TicketClock


@pytest.mark.parametrize(
    ("dtype", "encoding", "shape"),
    (
        (np.uint16, "gray", (3, 5)),
        (np.int16, "gray", (3, 5)),
        (np.float32, "rgb", (3, 5, 3)),
        (np.float64, "rgba", (3, 5, 4)),
    ),
)
def test_tile_repeats_exact_values_and_preserves_encoding(dtype, encoding, shape):
    pixels = np.arange(np.prod(shape), dtype=dtype).reshape(shape)[:, ::-1]
    source = ImageArray(pixels, encoding)
    block = builtin_registry().create("cv.tile_image", parameters={"rows": 2, "columns": 3})
    result = block.execute({"image": source})
    assert result is not None and result.passed, block.error
    output = result.outputs["image"]
    np.testing.assert_array_equal(output, np.tile(pixels, (2, 3, 1) if len(shape) == 3 else (2, 3)))
    assert output.dtype == dtype and output.color_format == encoding
    layout = result.outputs["layout"]
    assert (layout["rows"], layout["columns"]) == (2, 3)
    assert layout["tile_size"] == [5, 3]
    assert layout["image_size"] == [15, 6]


@pytest.mark.parametrize(("rows", "columns"), ((1, 1), (1, 3), (4, 1)))
def test_layout_keeps_source_provenance_and_repetition_uses_new_frame(rows, columns):
    spatial = frame([2, 3], "sensor")
    pyramid = {"level": 1, "largest_size": [4, 6], "to_largest": np.diag([2, 2, 1]).tolist()}
    ticket = TicketClock(session="test").issue("sensor")
    pixels = np.arange(6, dtype=np.uint8).reshape(3, 2)
    source = ImageArray(pixels, "gray", {"spatial": spatial, "pyramid": pyramid, "ticket": ticket})
    block = builtin_registry().create(
        "cv.tile_image", parameters={"rows": rows, "columns": columns}
    )
    result = block.execute({"image": source})
    assert result is not None and result.passed, block.error
    payload = metadata(result.outputs["image"])
    layout = result.outputs["layout"]
    np.testing.assert_array_equal(result.outputs["image"], np.tile(pixels, (rows, columns)))
    assert payload["ticket"] == ticket
    assert layout["source_spatial"] == spatial and layout["source_pyramid"] == pyramid
    if rows == columns == 1:
        assert payload["spatial"] == spatial and payload["pyramid"] == pyramid
    else:
        assert payload["spatial"] == frame([2 * columns, 3 * rows], f"{block.id}:image")
        assert "pyramid" not in payload


def test_tile_rejects_an_oversized_output_before_allocating():
    block = builtin_registry().create("cv.tile_image", parameters={"rows": 4096, "columns": 4096})
    assert block.execute({"image": np.zeros((1, 2), np.uint8)}) is None
    assert "exceeds 16,777,216" in block.error
    assert not block.current_emissions.get("image")


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_example_finds_six_matches_at_the_grid_origins(profile):
    project = (
        Path(__file__).parents[1] / "examples/image_processing" / "tile_image" / "tile_image.json"
    )
    graph = Graph.load(project, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    tile = graph.blocks["tile"].last_result.outputs
    assert tile["image"].shape == (192, 336)
    assert tile["layout"]["source_spatial"]["root_id"] == "reference:shape_reference"
    found = graph.blocks["find"].last_result.outputs
    assert found["count"] == 6
    assert {tuple(match["bounds"]) for match in found["matches"]} == {
        (x, y, 112, 96) for y in (0, 96) for x in (0, 112, 224)
    }
