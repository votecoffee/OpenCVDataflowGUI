"""Check temporal denoising against OpenCV and verify center-frame provenance."""

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import FlowGroup, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock


def sequence(images):
    clock = TicketClock("test")
    return FlowGroup(
        tuple(
            ValuePacket(
                image,
                {
                    "ticket": clock.issue("camera"),
                    "spatial": frame([image.shape[1], image.shape[0]], f"frame-{i}"),
                },
            )
            for i, image in enumerate(images)
        ),
        ("camera",) * len(images),
    )


def test_temporal_nlm_uses_selected_window_and_preserves_center_ticket():
    rng = np.random.default_rng(8)
    images = [rng.integers(900, 940, (19, 20), dtype=np.uint16) for _ in range(5)]
    group = sequence(images)
    block = builtin_registry().create(
        "cv.temporal_nlm",
        parameters={
            "target_index": 3,
            "temporal_window": 3,
            "template_window": 3,
            "search_window": 5,
            "strength": 18,
            "norm": "l1",
        },
    )
    mask = np.full(images[0].shape, 255, np.uint8)
    mask[:, :3] = 0
    result = block.execute({"frames": group, "mask": mask})
    assert result is not None, block.error
    expected = cv2.fastNlMeansDenoisingMulti(images[2:], 1, 3, [18.0], None, 3, 5, cv2.NORM_L1)
    expected[mask == 0] = images[3][mask == 0]
    np.testing.assert_array_equal(result.outputs["image"], expected)
    assert result.payloads["image"]["ticket"] == group.items[3].payload["ticket"]
    assert result.payloads["image"]["spatial"] == group.items[3].payload["spatial"]
    assert result.outputs["info"] == {"target_index": 3, "first_index": 2, "window_size": 3}


def test_temporal_luminance_chroma_preserves_rgb_order_and_center_alpha():
    rng = np.random.default_rng(11)
    frames = [
        ImageArray(rng.integers(80, 100, (13, 15, 4), dtype=np.uint8), "rgba") for _ in range(3)
    ]
    block = builtin_registry().create(
        "cv.temporal_nlm",
        parameters={
            "mode": "luminance_chroma",
            "template_window": 3,
            "search_window": 5,
            "strength": 10,
            "color_strength": 8,
        },
    )
    result = block.execute({"frames": sequence(frames)})
    assert result is not None, block.error
    bgr = [cv2.cvtColor(np.asarray(frame_), cv2.COLOR_RGBA2BGR) for frame_ in frames]
    expected = cv2.fastNlMeansDenoisingColoredMulti(bgr, 1, 3, None, 10, 8, 3, 5)
    output = result.outputs["image"]
    assert image_format(output) == "rgba"
    np.testing.assert_array_equal(output[..., :3], expected[..., ::-1])
    np.testing.assert_array_equal(output[..., 3], frames[1][..., 3])


def test_temporal_nlm_rejects_incomplete_window_and_mismatched_frames():
    source = np.zeros((8, 9), np.uint8)
    block = builtin_registry().create("cv.temporal_nlm")
    assert block.execute({"frames": sequence([source] * 2)}) is None
    assert "window does not fit" in block.error
    assert block.execute({"frames": sequence([source, source, source[:, :-1]])}) is None
    assert "matching size" in block.error
    group = sequence([source] * 3)
    for entry in group.items:
        entry.payload["spatial"]["root_id"] = "shared"
    group.items[-1].payload["spatial"]["to_root"][0][2] = 1
    assert block.execute({"frames": group}) is None
    assert "same pixel grid" in block.error
