"""Check fixed-scale matching, peak suppression, and pose outputs."""

import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.payloads import NO_VALUE


def test_template_instances_detect_all_parts_and_limit_search():
    registry = builtin_registry()
    source = registry.create("input.parts_demo").execute({}).outputs
    block = registry.create(
        "inspect.template_instances",
        parameters={"min_score": 0.85, "draw_overlay": False, "pose_offset_x": 2},
    )
    inputs = {"image": source["image"], "template": source["reference"]}
    result = block.execute(inputs)
    assert result is not None, block.error
    assert result.passed
    assert result.outputs["count"] == 25
    assert result.outputs["overlay"] is NO_VALUE
    assert {tuple(match["bounds"][:2]) for match in result.outputs["matches"]} == {
        (20 + 56 * x, 20 + 56 * y) for x in range(5) for y in range(5)
    }
    for match, pose in zip(result.outputs["matches"], result.outputs["poses"], strict=True):
        assert pose["origin"] == [match["center"][0] + 2, match["center"][1]]
    search = np.zeros_like(source["image"])
    search[:60, :60] = 255
    limited = block.execute({**inputs, "search_mask": search})
    assert limited.outputs["count"] == 1
    assert limited.outputs["matches"][0]["bounds"] == [20, 20, 32, 32]


def test_template_instances_accept_one_pixel_response():
    image = np.arange(16, dtype=np.uint8).reshape(4, 4)
    block = builtin_registry().create("inspect.template_instances")
    result = block.execute({"image": image, "template": image})
    assert result is not None, block.error
    assert result.outputs["count"] == 1
    assert result.outputs["response"].shape == (1, 1)
