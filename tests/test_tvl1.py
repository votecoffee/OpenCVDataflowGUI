"""Exercise TV-L1 through both graph inputs and preserve observation metadata."""

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.payloads import FlowGroup, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock


def test_tvl1_single_image_matches_native_and_preserves_selection():
    rng = np.random.default_rng(10)
    source = np.asfortranarray(rng.integers(55, 88, (13, 18), dtype=np.uint8))
    mask = np.zeros(source.shape, np.uint8)
    mask[:, 4:14] = 255
    block = builtin_registry().create(
        "cv.denoise_tvl1", parameters={"lambda_weight": 0.9, "iterations": 15}
    )
    spatial = frame([18, 13], "image")
    result = block.execute({"image": ValuePacket(source, {"spatial": spatial}), "mask": mask})
    assert result is not None, block.error
    expected = np.empty(source.shape, np.uint8)
    cv2.denoise_TVL1([source], expected, lambda_=0.9, niters=15)
    expected[mask == 0] = source[mask == 0]
    np.testing.assert_array_equal(result.outputs["image"], expected)
    assert result.outputs["observation_count"] == 1
    assert result.payloads["image"]["spatial"] == spatial


def test_tvl1_fuses_ordered_observations_and_preserves_reference_ticket():
    rng = np.random.default_rng(16)
    frames = [rng.integers(50, 100, (18, 19), dtype=np.uint8) for _ in range(4)]
    clock = TicketClock("tvl1")
    entries = tuple(
        ValuePacket(image, {"ticket": clock.issue("camera"), "spatial": frame([19, 18], f"{i}")})
        for i, image in enumerate(frames)
    )
    group = FlowGroup(entries, ("camera",) * len(entries))
    block = builtin_registry().create("cv.denoise_tvl1")
    result = block.execute({"observations": group})
    assert result is not None, block.error
    expected = np.empty_like(frames[0])
    cv2.denoise_TVL1(frames, expected, lambda_=1.0, niters=30)
    np.testing.assert_array_equal(result.outputs["image"], expected)
    assert result.outputs["observation_count"] == 4
    assert result.payloads["image"]["ticket"] == entries[1].payload["ticket"]
    assert result.payloads["image"]["spatial"] == entries[1].payload["spatial"]

    assert block.execute({"image": frames[0], "observations": group}) is None
    assert "exactly one" in block.error
    assert block.execute({"image": frames[0].astype(np.uint16)}) is None
    assert "8-bit monochrome" in block.error
    group.items[0].payload["spatial"]["root_id"] = "1"
    group.items[0].payload["spatial"]["to_root"][0][2] = 2
    assert block.execute({"observations": group}) is None
    assert "same pixel grid" in block.error
