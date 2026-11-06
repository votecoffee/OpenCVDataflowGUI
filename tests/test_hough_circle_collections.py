"""Check native circle strength and its use in collection workflows."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import metadata

_EXAMPLE = Path(__file__).parents[1] / "examples/geometry/hough_circle_collections"
_PROJECT = _EXAMPLE / "hough_circle_collections.json"
_PARAMETERS = {"dp": 1.0, "min_distance": 40.0, "min_radius": 15, "max_radius": 36}


def _image():
    return cv2.imread(str(_EXAMPLE / "assets/disks.png"), cv2.IMREAD_GRAYSCALE)


@pytest.mark.parametrize(
    ("method", "native_method", "edge", "threshold"),
    [
        ("gradient", cv2.HOUGH_GRADIENT, 100, 22),
        ("gradient_alt", cv2.HOUGH_GRADIENT_ALT, 300, 0.9),
    ],
)
def test_votes_and_radii_match_native_results(method, native_method, edge, threshold):
    image = _image()
    block = builtin_registry().create(
        "inspect.hough_circles",
        parameters={
            **_PARAMETERS,
            "method": method,
            "edge_threshold": edge,
            "center_threshold": threshold,
        },
    )
    result = block.execute({"image": image})
    assert result and result.passed, block.error
    outputs = result.outputs
    assert outputs["count"] == 2
    native = cv2.HoughCirclesWithAccumulator(
        image,
        native_method,
        1.0,
        40.0,
        param1=edge,
        param2=threshold,
        minRadius=15,
        maxRadius=36,
    ).reshape(-1, 4)
    np.testing.assert_allclose(outputs["circles"], native[:, :3])
    np.testing.assert_allclose(outputs["centers"], [[62, 75], [166, 104]], atol=1)
    assert outputs["votes"] == native[:, 3].astype(int).tolist()
    data = outputs["collection"]
    assert data["kind"] == "circles" and data["count"] == 2
    assert [item["metrics"]["strength"] for item in data["items"]] == outputs["votes"]
    assert [item["metrics"]["radius"] for item in data["items"]] == pytest.approx([30, 20], abs=2)


def test_center_only_retains_legacy_rows_and_has_no_radius_or_vote_strength():
    registry = builtin_registry()
    block = registry.create(
        "inspect.hough_circles",
        parameters={**_PARAMETERS, "center_threshold": 22.0, "max_radius": -1},
    )
    image = _image()
    result = block.execute({"image": image})
    assert result and result.passed, block.error
    assert result.outputs["count"] == 2
    assert result.outputs["votes"] == [None, None]
    assert [row[2] for row in result.outputs["circles"]] == [0, 0]
    data = result.outputs["collection"]
    assert all(item["metrics"]["radius"] is None for item in data["items"])
    assert all(item["metrics"]["strength"] is None for item in data["items"])
    draw = registry.create("collection.draw")
    drawn = draw.execute({"image": image, "collection": data})
    assert drawn and drawn.passed, draw.error
    assert cv2.countNonZero(drawn.outputs["mask"]) == 2

    empty = block.execute({"image": np.zeros_like(image)})
    assert empty and empty.passed
    assert empty.outputs["circles"] == empty.outputs["votes"] == []
    assert empty.outputs["collection"]["count"] == 0


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_saved_project_ranks_circles_and_draws_selected_disk(profile):
    graph = Graph.load(_PROJECT, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    source = graph.blocks["source"].last_result.outputs["image"]
    ranked = graph.blocks["sort"].last_result.outputs["collection"]
    assert ranked["context"]["spatial"] == metadata(source)["spatial"]
    strongest, second = ranked["items"]
    assert strongest["metrics"]["strength"] > second["metrics"]["strength"] > 0
    filtered = graph.blocks["filter"].last_result.outputs
    assert filtered["count"] == filtered["rejected_count"] == 1
    assert filtered["collection"]["items"] == [strongest]
    drawn = graph.blocks["selected"].last_result.outputs
    assert drawn["count"] == 1
    assert drawn["mask"][75, 62] == 255 and drawn["mask"][104, 166] == 0
    assert cv2.countNonZero(drawn["mask"]) == pytest.approx(np.pi * 30**2, rel=0.05)
