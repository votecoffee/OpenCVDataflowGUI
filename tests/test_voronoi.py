"""Verify image-clipped Voronoi geometry, selected masks and frame checks."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import ValuePacket, frame


def test_collinear_sites_split_the_image_and_keep_original_indices():
    block = builtin_registry().create("inspect.voronoi_cells", parameters={"cell_index": 1})
    spatial = frame([16, 10], "camera")
    result = block.execute(
        {
            "image": ValuePacket(np.zeros((10, 16), np.uint8), {"spatial": spatial}),
            "points": [[2, 4], [2, 4], [7, 4], [12, 4]],
        }
    )
    assert result is not None, block.error
    cells = result.outputs["cells"]
    assert [cell["source_index"] for cell in cells] == [0, 2, 3]
    assert [cell["area"] for cell in cells] == pytest.approx([50, 50, 60], abs=0.002)
    assert result.outputs["selected_area"] == cells[1]["area"]
    assert result.outputs["cell_count"] == 3
    for cell, expected in zip(cells, ((-0.5, 4.5), (4.5, 9.5), (9.5, 15.5))):
        vertices = np.asarray(cell["vertices"])
        assert (vertices[:, 0].min(), vertices[:, 0].max()) == pytest.approx(expected, abs=0.0002)
        assert (vertices[:, 1].min(), vertices[:, 1].max()) == pytest.approx((-0.5, 9.5))
    mask = result.outputs["mask"]
    assert mask.dtype == np.uint8
    assert np.all(mask[:, 7] == 255) and not mask[:, [2, 12]].any()
    assert result.payloads["cells"]["spatial"] == result.payloads["mask"]["spatial"] == spatial


def test_single_site_covers_even_a_one_row_reference():
    block = builtin_registry().create("inspect.voronoi_cells")
    result = block.execute({"image": np.zeros((1, 13), np.uint8), "points": [[6, 0]]})
    assert result is not None, block.error
    assert result.outputs["selected_area"] == pytest.approx(13)
    assert np.all(result.outputs["mask"] == 255)


def test_cocircular_sites_on_image_corners_have_four_equal_cells():
    block = builtin_registry().create("inspect.voronoi_cells")
    result = block.execute(
        {"image": np.zeros((10, 10), np.uint8), "points": [[0, 0], [9, 0], [9, 9], [0, 9]]}
    )
    assert result is not None, block.error
    assert [cell["area"] for cell in result.outputs["cells"]] == pytest.approx([25] * 4)


@pytest.mark.parametrize(
    ("points", "parameters", "message"),
    [
        ([[-0.6, 0]], {}, "inside the reference image"),
        ([[5, 5]], {"cell_index": 1}, "Cell index"),
        (
            ValuePacket([[5, 5]], {"spatial": frame([10, 10], "other-camera")}),
            {},
            "same coordinate frame",
        ),
    ],
)
def test_invalid_site_coordinates_or_selection_fail_as_a_block_error(points, parameters, message):
    block = builtin_registry().create("inspect.voronoi_cells", parameters=parameters)
    image = ValuePacket(np.zeros((10, 10), np.uint8), {"spatial": frame([10, 10], "camera")})
    assert block.execute({"image": image, "points": points}) is None
    assert message in block.error


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_saved_voronoi_example_measures_selected_cell_and_overlays_geometry(profile):
    project = Path(__file__).parents[1] / "examples/opencv tools/voronoi_cells/voronoi_cells.json"
    graph = Graph.load(project, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    cells = graph.blocks["cells"].last_result
    assert cells.outputs["cell_count"] == 5
    assert sum(cell["area"] for cell in cells.outputs["cells"]) == pytest.approx(12100)
    assert [len(layer.objects) for layer in cells.outputs["result_graphics"].layers] == [5, 1]
    statistics = graph.blocks["statistics"].last_result.outputs
    assert statistics["selected_count"] == np.count_nonzero(cells.outputs["mask"])
    assert 0 < statistics["selected_count"] < 12100
    overlay = graph.blocks["overlay"].last_result.outputs["image"]
    assert overlay.shape == (110, 110, 3) and np.any(overlay[..., 0] != overlay[..., 1])
