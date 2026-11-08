"""Verify measurements for sparse label IDs and selected, disconnected regions."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.objects import validate_collection
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock


def test_measure_label_regions_preserves_ids_and_uses_selected_pixel_centers():
    labels = np.full((8, 10), -1, np.int32)
    large_id = 2**31 - 1
    labels[1, 1] = large_id  # Excluded by the region.
    labels[2, 3:6] = large_id
    labels[4, 7] = large_id  # Same ID in a disconnected island.
    labels[3, 3:5] = -5
    labels[3, 6] = 0
    mask = np.ones(labels.shape, np.uint8)
    mask[2, 4] = 0
    intensity = np.arange(labels.size, dtype=np.uint16).reshape(labels.shape) * 100
    spatial = frame([10, 8], "part")
    ticket = TicketClock().issue("camera")
    source = ValuePacket(labels, {"spatial": spatial, "ticket": ticket})
    region = {
        "model": "image_region_v1", "image_size": [10, 8],
        "shape": "rectangle", "rectangle": [2, 2, 7, 5],
    }
    block = builtin_registry().create("inspect.measure_label_regions")
    result = block.execute({
        "labels": source, "intensity_image": ValuePacket(intensity, {"spatial": spatial}),
        "mask": mask, "region": region,
    })
    assert result is not None, block.error
    output = validate_collection(result.outputs["collection"])
    assert output["context"]["spatial"] == spatial
    assert result.payloads["collection"]["ticket"] == ticket
    assert [item["id"] for item in output["items"]] == [-5, large_id]
    item = output["items"][1]
    assert item["geometry"]["bounds"] == [3, 2, 5, 3]
    assert item["metrics"]["area"] == 3
    assert item["metrics"]["extent"] == pytest.approx(3 / 15)
    assert item["metrics"]["intensity"] == pytest.approx((2300 + 2500 + 4700) / 3)
    assert result.outputs["points"][1] == pytest.approx([5, 8 / 3])
    assert result.drawings["measured labels"].source == "labels"


def test_measure_label_regions_can_include_zero_and_handle_empty_selections():
    labels = np.array([[-1, 0], [0, 7]], np.int16)
    block = builtin_registry().create(
        "inspect.measure_label_regions", parameters={"ignored_labels": [-1]}
    )
    result = block.execute({"labels": labels})
    assert result is not None, block.error
    assert result.outputs["count"] == 2
    item = result.outputs["collection"]["items"][0]
    assert item["id"] == 0 and item["metrics"]["area"] == 2
    assert item["metrics"]["intensity"] is None

    empty = block.execute({"labels": labels, "mask": np.zeros((2, 2), np.uint8)})
    assert empty is not None, block.error
    assert empty.outputs["count"] == 0 and empty.outputs["points"] == []


def test_measure_label_regions_rejects_wrong_grid_and_noninteger_labels():
    block = builtin_registry().create("inspect.measure_label_regions")
    labels = np.ones((4, 6), np.int32)
    image = np.zeros_like(labels, dtype=np.uint8)
    source = ValuePacket(labels, {"spatial": frame([6, 4], "reference")})
    other = ValuePacket(image, {"spatial": frame([6, 4], "sample")})
    assert block.execute({"labels": source, "intensity_image": other}) is None
    assert "must align" in block.error
    assert block.execute({"labels": labels.astype(np.float32)}) is None
    assert "label plane" in block.error
    block.parameters["ignored_labels"] = [False]
    assert block.execute({"labels": labels}) is None
    assert "int32 IDs" in block.error
