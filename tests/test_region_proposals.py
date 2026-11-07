"""Verify native segmentation and original-grid proposal collections."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import ValuePacket, frame, packet_for


def test_graph_segments_preserve_native_regions_and_clip_to_selection():
    image = np.full((96, 128), 45, np.uint8)
    image[:, 63:] = 200
    mask = np.zeros_like(image)
    mask[10:87, 17:120] = 255
    registry = builtin_registry()
    block = registry.create("cv.graph_segments")
    result = block.execute({"image": image, "mask": mask})
    assert result is not None, block.error
    native = cv2.ximgproc.segmentation.createGraphSegmentation(0.5, 300, 100).processImage(image)
    labels = result.outputs["labels"]
    assert labels.dtype == np.int32
    np.testing.assert_array_equal(labels[mask != 0], native[mask != 0] + 1)
    assert not labels[mask == 0].any()
    assert result.outputs["count"] == np.unique(labels[mask != 0]).size
    boundaries = result.outputs["boundaries"]
    assert boundaries[10, 17] == 255
    assert not boundaries[mask == 0].any()
    assert boundaries[25:70, 58:68].any()


@pytest.mark.parametrize("mode", ("single", "fast", "quality"))
def test_native_modes_propose_a_visible_part(mode):
    image = np.full((120, 170, 3), 45, np.uint8)
    image[24:78, 33:89] = (30, 175, 225)
    block = builtin_registry().create("inspect.selective_search", parameters={"mode": mode})
    result = block.execute({"image": image})
    assert result is not None, block.error
    expected = 54 * 56
    items = result.outputs["collection"]["items"]
    assert items
    for item in items:
        x, y, w, h = item["geometry"]["bounds"]
        shared = max(0, min(x + w, 89) - max(x, 33)) * max(0, min(y + h, 78) - max(y, 24))
        if shared / (expected + w * h - shared) > 0.75:
            break
    else:
        pytest.fail(f"{mode} did not generate a proposal covering the known part")


def test_scaled_proposals_use_input_grid_and_work_with_collection_blocks(monkeypatch):
    class NativeSearch:
        def setBaseImage(self, image):
            assert image.shape == (38, 64, 3)
            assert image[5, 5].tolist() == [80, 40, 20]  # Native input is BGR.

        def switchToSelectiveSearchFast(self, k, increment, sigma):
            assert (k, increment, sigma) == (160, 140, 0.7)

        def process(self):
            return np.array([[0, 0, 10, 10], [10, 5, 20, 20], [16, 8, 12, 14]], np.int32)

    monkeypatch.setattr(
        cv2.ximgproc.segmentation, "createSelectiveSearchSegmentation", NativeSearch
    )
    image = ImageArray(np.full((120, 200, 3), (20, 40, 80), np.uint8), "rgb")
    image = ValuePacket(image, {"spatial": frame([200, 120], "fixture")})
    mask = np.zeros((120, 200), np.uint8)
    mask[32:85, 45:95] = 255
    registry = builtin_registry()
    block = registry.create("inspect.selective_search", parameters={
        "working_side": 64, "max_proposals": 1, "base_k": 160, "inc_k": 140, "sigma": 0.7,
    })
    result = block.execute({"image": image, "mask": mask})
    assert result is not None, block.error
    assert result.outputs["info"] == {
        "mode": "fast", "working_size": [64, 38], "native_count": 3,
        "selected_count": 2, "truncated": True,
    }
    collection = result.outputs["collection"]
    assert collection["kind"] == "boxes" and collection["count"] == 1
    assert collection["context"]["spatial"]["root_id"] == "fixture"
    proposal = collection["items"][0]
    assert proposal["id"] == 2 and proposal["geometry"]["bounds"] == [31, 15, 63, 64]
    assert len(result.drawings["proposals"].objects) == 1
    filtered = registry.create("collection.filter", parameters={"minimum": 3500})
    accepted = filtered.execute({"collection": packet_for(result, "collection")})
    assert accepted is not None, filtered.error
    assert accepted.outputs["count"] == 1
    draw = registry.create("collection.draw")
    drawn = draw.execute({"image": image, "collection": packet_for(accepted, "collection")})
    assert drawn is not None, draw.error
    assert np.count_nonzero(drawn.outputs["mask"]) == 63 * 64
