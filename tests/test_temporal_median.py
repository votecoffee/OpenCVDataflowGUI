"""Validate batch temporal medians, selection, and capture provenance."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import FlowGroup, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock


def observations(images):
    return FlowGroup(
        tuple(
            ValuePacket(image, {"spatial": frame(image.shape[1::-1], f"capture-{index}")})
            for index, image in enumerate(images)
        ),
        ("camera",) * len(images),
    )


@pytest.mark.parametrize("dtype", (np.uint8, np.uint16, np.float32))
@pytest.mark.parametrize("count", (3, 4))
def test_temporal_median_matches_reference_and_preserves_batch_ticket(dtype, count):
    rng = np.random.default_rng(7)
    images = [rng.integers(0, 200, (24, 31), dtype=np.uint8).astype(dtype) for _ in range(count)]
    original = np.stack(images)
    group = observations(images)
    ticket = TicketClock("test").issue("batch")
    block = builtin_registry().create("sequence.temporal_median")
    result = block.execute({"frames": ValuePacket(group, {"ticket": ticket})})
    assert result is not None, block.error
    expected = np.median(original, axis=0)
    if dtype != np.float32:
        expected = np.rint(expected)
    output = result.outputs["image"]
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(images, original)
    assert output.dtype == dtype
    assert result.outputs["observation_count"] == count
    assert result.payloads["image"]["ticket"] == ticket
    assert result.payloads["image"]["spatial"] == group.items[(count - 1) // 2].payload["spatial"]


def test_temporal_median_preserves_anchor_alpha_and_samples_outside_selection():
    images = [
        ImageArray(np.full((8, 12, 4), [value, 20, 30, alpha], np.uint8), "rgba")
        for value, alpha in ((17, 100), (240, 180), (19, 210))
    ]
    mask = np.zeros((8, 12), np.uint8)
    mask[:, 3:10] = 255
    block = builtin_registry().create("sequence.temporal_median")
    result = block.execute({"frames": observations(images), "mask": mask})
    assert result is not None, block.error
    output = result.outputs["image"]
    assert image_format(output) == "rgba"
    np.testing.assert_array_equal(output[mask == 0], images[1][mask == 0])
    assert (output[..., 0][mask != 0] == 19).all()
    assert (output[..., 3] == 180).all()


def test_temporal_median_rejects_mixed_grids_and_incomplete_batches():
    source = np.zeros((10, 12), np.uint8)
    block = builtin_registry().create("sequence.temporal_median")
    assert block.execute({"frames": observations([source])}) is None
    assert "two to 64" in block.error
    assert block.execute({"frames": observations([source, source[:, :-1]])}) is None
    assert "matching size" in block.error

    nonimage = FlowGroup(
        (ValuePacket(np.zeros(3), {}), ValuePacket(source, {})), ("vector", "image")
    )
    assert block.execute({"frames": nonimage}) is None
    assert "nonempty image" in block.error

    group = observations([source, source.copy()])
    context = group.items[1].payload["spatial"]
    context["root_id"] = "capture-0"
    context["to_root"][0][2] = 1
    assert block.execute({"frames": group}) is None
    assert "same pixel grid" in block.error
