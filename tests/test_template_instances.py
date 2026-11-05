"""Check fixed-scale matching, peak suppression, and pose outputs."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.result_graphics import ACCEPTED, REJECTED


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


def test_template_instances_review_includes_weaker_peaks_without_emitting_poses():
    registry = builtin_registry()
    template = np.random.default_rng(9).integers(0, 255, (24, 24), dtype=np.uint8)
    search = np.zeros((100, 160), np.uint8)
    search[45:69, 20:44] = template
    search[45:69, 100:124] = template
    search[45:57, 100:124] = 0
    block = registry.create("inspect.template_instances", parameters={"min_score": 0.9})
    result = block.execute({"image": search, "template": template})
    assert result is not None, block.error
    assert result.outputs["count"] == 1
    graphic = result.graphics["template instances"][40:]
    assert np.any(np.all(graphic[:, :65] == ACCEPTED, axis=-1))
    assert np.any(np.all(graphic[:, 85:] == REJECTED, axis=-1))
    assert not np.any(np.all(result.outputs["overlay"] == REJECTED, axis=-1))


def test_template_instances_accept_one_pixel_response():
    image = np.arange(16, dtype=np.uint8).reshape(4, 4)
    block = builtin_registry().create("inspect.template_instances")
    result = block.execute({"image": image, "template": image})
    assert result is not None, block.error
    assert result.outputs["count"] == 1
    assert result.outputs["response"].shape == (1, 1)


@pytest.mark.parametrize("method", ["ccoeff_normed", "ccorr_normed", "sqdiff_normed"])
def test_subpixel_template_peaks_refine_centers_without_changing_scores(method):
    y, x = np.mgrid[:31, :37]
    template = (
        180 * np.exp(-((x - 11) ** 2 / 24 + (y - 12) ** 2 / 16))
        + 100 * np.exp(-((x - 23) ** 2 / 13 + (y - 18) ** 2 / 21))
    ).astype(np.float32)
    image = np.zeros((92, 140), np.float32)
    expected = [(23.34375, 18.625), (84.25, 49.375)]
    for x, y in expected:
        image += cv2.warpAffine(template, np.float32([[1, 0, x], [0, 1, y]]), (140, 92))
    registry = builtin_registry()
    inputs = {"image": image, "template": template}
    multiple = registry.create(
        "inspect.template_instances",
        parameters={"method": method, "maximum_matches": 2, "pose_offset_x": 2},
    )
    before = multiple.execute(inputs).outputs["matches"]
    multiple.parameters["subpixel"] = True
    after = multiple.execute(inputs).outputs["matches"]
    assert [(m["bounds"], m["score"]) for m in before] == [(m["bounds"], m["score"]) for m in after]
    ranked = sorted(after, key=lambda match: match["bounds"][0])
    np.testing.assert_allclose(
        [match["subpixel_top_left"] for match in ranked], expected, atol=0.03
    )
    for match in ranked:
        x, y = match["subpixel_top_left"]
        np.testing.assert_allclose(match["center"], [x + 18, y + 15])
        np.testing.assert_allclose(match["pose"]["origin"], [x + 20, y + 15])

    single = registry.create("inspect.match_template", parameters={"method": method})
    initial = single.execute(inputs).outputs
    single.parameters["subpixel"] = True
    refined = single.execute(inputs).outputs
    assert refined["location"]["top_left"] == initial["location"]["top_left"]
    assert refined["raw_score"] == initial["raw_score"]
    np.testing.assert_allclose(refined["center"][0], after[0]["center"])


@pytest.mark.parametrize("kind", ["inspect.match_template", "inspect.template_instances"])
def test_subpixel_template_falls_back_at_search_mask_boundary(kind):
    template = np.random.default_rng(21).integers(0, 255, (7, 9), dtype=np.uint8)
    image = np.zeros((24, 28), np.uint8)
    image[5:12, 8:17] = template
    mask = np.zeros_like(image)
    mask[8, 12] = 255
    block = builtin_registry().create(kind, parameters={"subpixel": True})
    result = block.execute({"image": image, "template": template, "search_mask": mask})
    assert result is not None and result.passed, block.error
    if kind == "inspect.match_template":
        assert "subpixel_top_left" not in result.outputs["location"]
        assert result.outputs["center"] == [[12, 8]]
    else:
        match = result.outputs["matches"][0]
        assert "subpixel_top_left" not in match
        assert match["center"] == [12, 8]
