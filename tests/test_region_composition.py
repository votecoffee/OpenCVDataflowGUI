import copy
import json

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.examples import combined_regions_graph
from opencv_dataflow.selection import rasterize
from opencv_dataflow.toolblocks import encapsulate


def execute(kind, inputs, **parameters):
    block = builtin_registry().create(kind, parameters=parameters)
    block.execute(inputs)
    assert block.state in ("pass", "fail"), block.error
    return block


def shapes():
    image = np.zeros((80, 120, 3), np.uint8)
    a = execute("region.shape", {"image": image}, rectangle=[10, 10, 60, 50]).last_result.outputs
    b = execute(
        "region.shape", {"image": image}, shape="circle", center=[65, 40], radius=25
    ).last_result.outputs
    return image, a, b


@pytest.mark.parametrize("operation", ["union", "intersection", "difference", "xor", "invert"])
def test_operations_match_boolean_masks_and_preserve_structured_regions(operation):
    image, a, b = shapes()
    original = copy.deepcopy(a["region"])
    block = execute(
        "region.combine",
        {"image": image, "region_a": a["region"], "region_b": b["region"]},
        operation=operation,
    )
    output = block.last_result.outputs
    expected = {
        "union": a["mask"] | b["mask"],
        "intersection": a["mask"] & b["mask"],
        "difference": a["mask"] & ~b["mask"],
        "xor": a["mask"] ^ b["mask"],
        "invert": ~a["mask"],
    }[operation]
    np.testing.assert_array_equal(output["mask"], expected)
    restored = json.loads(json.dumps(output["region"]))
    np.testing.assert_array_equal(rasterize(restored, image.shape), expected)
    assert a["region"] == original
    histogram = execute(
        "inspect.histogram", {"image": image, "region": restored}
    ).last_result.outputs
    assert all(
        c["finite_samples"] == np.count_nonzero(expected)
        for c in histogram["histogram"]["channels"]
    )


def test_nested_difference_keeps_holes_and_measurement_uses_pixel_area():
    image = np.zeros((100, 100), np.uint8)
    outer = execute("region.shape", {"image": image}, shape="circle", center=[50, 50], radius=30)
    inner = execute("region.shape", {"image": image}, shape="circle", center=[50, 50], radius=10)
    ring = execute(
        "region.combine",
        {
            "image": image,
            "region_a": outer.last_result.outputs["region"],
            "region_b": inner.last_result.outputs["region"],
        },
        operation="difference",
    )
    mask = ring.last_result.outputs["mask"]
    assert mask[50, 50] == 0 and mask[50, 75] == 255
    measured = execute(
        "region.measure",
        {"image": image, "region": ring.last_result.outputs["region"]},
        min_area=2000,
        max_area=3000,
        min_coverage=0.2,
        max_coverage=0.3,
    )
    output = measured.last_result.outputs
    assert measured.state == "pass"
    assert output["area"] == np.count_nonzero(mask)
    assert output["centroid"] == [[50.0, 50.0]]  # Centroid can lie in a hole.
    assert output["bounds"] == [20, 20, 61, 61]
    assert output["coverage"] == pytest.approx(output["area"] / 10000)
    assert output["data"]["area_pixels"] == output["area"]
    measured.parameters["min_area"] = 2800
    measured.execute({"image": image, "region": ring.last_result.outputs["region"]})
    assert measured.state == "fail" and measured.last_result is not None


def test_whole_empty_and_intersected_region_measurements():
    image, a, b = shapes()
    whole = execute("region.measure", {"image": image}).last_result.outputs
    assert whole["area"] == 80 * 120 and whole["coverage"] == 1
    assert whole["bounds"] == [0, 0, 120, 80]
    empty = execute("region.measure", {"image": image, "mask": np.zeros((80, 120), np.uint8)})
    assert empty.state == "pass"
    assert empty.last_result.outputs["bounds"] == empty.last_result.outputs["centroid"] == []
    result = execute("region.measure", {"image": image, "mask": a["mask"], "region": b["region"]})
    assert result.last_result.outputs["area"] == cv2.countNonZero(a["mask"] & b["mask"])


@pytest.mark.parametrize(
    "inputs,operation",
    [({}, "union"), ({}, "difference"), ({"region_b": {"model": "bad"}}, "intersection")],
)
def test_missing_or_malformed_operands_report_errors(inputs, operation):
    image, a, _ = shapes()
    block = builtin_registry().create("region.combine", parameters={"operation": operation})
    block.execute({"image": image, "region_a": a["region"], **inputs})
    assert block.state == "error" and block.last_result is None


def test_expression_limits_and_mismatched_child_dimensions():
    image, a, b = shapes()
    composite = {
        "model": "image_region_v1",
        "image_size": [120, 80],
        "shape": "composite",
        "operation": "union",
        "children": [a["region"], b["region"]],
    }
    bad = copy.deepcopy(composite)
    bad["children"][1]["image_size"] = [10, 10]
    with pytest.raises(ValueError, match="dimensions"):
        rasterize(bad, image.shape)
    deep = a["region"]
    for _ in range(18):
        deep = {**composite, "operation": "invert", "children": [deep]}
    with pytest.raises(ValueError, match="16 levels"):
        rasterize(deep, image.shape)
    wide = a["region"]
    for _ in range(7):
        wide = {**composite, "children": [wide, wide]}
    with pytest.raises(ValueError, match="128"):
        rasterize(wide, image.shape)


@pytest.mark.parametrize(
    "parameters",
    [
        {"min_area": -1},
        {"max_area": 2, "min_area": 3},
        {"min_coverage": 0.6, "max_coverage": 0.4},
        {"max_coverage": 1.1},
    ],
)
def test_measurement_limit_validation(parameters):
    block = builtin_registry().create("region.measure", parameters=parameters)
    block.execute({"image": np.zeros((10, 10), np.uint8)})
    assert block.state == "error" and block.last_result is None


def test_saved_composition_project_and_nested_reuse(tmp_path):
    registry = builtin_registry()
    graph = Graph.load("examples/custom tools/combined_regions/combined_regions.json", registry)
    assert graph.to_dict() == combined_regions_graph(registry).to_dict()
    previous = None
    for _ in range(3):
        graph.run()
        assert all(b.state == "pass" for b in graph.blocks.values())
        result = graph.blocks["combined_components"].last_result.outputs
        assert result["count"] == 2
        mask = graph.blocks["combined_union"].last_output_images["mask"]
        if previous is not None:
            np.testing.assert_array_equal(mask, previous)
        previous = mask.copy()
    selected = [
        "combined_outer",
        "combined_inner",
        "combined_rectangle",
        "combined_ring",
        "combined_union",
    ]
    block = encapsulate(graph, selected, "Reusable combined region", registry)
    path = tmp_path / "combined.json"
    graph.save(path)
    restored = Graph.load(path, registry)
    restored.run()
    assert restored.blocks[block.id].state == "pass"
    assert restored.blocks["combined_components"].last_result.outputs["count"] == 2
