"""Exercise exposure stack alignment, coordinate provenance and fusion."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import FlowGroup, ValuePacket, frame, packet_for
from opencv_dataflow.tickets import TicketClock


@pytest.mark.parametrize("border", ["crop", "zero_pad"])
def test_mtb_recovers_translations_and_uses_reference_grid(border):
    rng = np.random.default_rng(14)
    raw = cv2.resize(rng.integers(20, 190, (16, 24), np.uint8), (192, 128))
    shifts = [(-3, 2), (0, 0), (4, -3)]
    gains = [0.7, 1.0, 1.4]
    entries = []
    clock = TicketClock()
    for index, ((x, y), gain) in enumerate(zip(shifts, gains, strict=True)):
        scaled = np.rint(np.clip(raw * gain, 0, 255)).astype(np.uint8)
        matrix = np.float32([[1, 0, x], [0, 1, y]])
        pixels = cv2.warpAffine(scaled, matrix, (192, 128), borderMode=cv2.BORDER_REPLICATE)
        label = f"frame_{index}"
        entries.append(ValuePacket(pixels, {
            "spatial": frame([192, 128], label), "ticket": clock.issue(label),
        }))
    group = FlowGroup(tuple(entries), ("short", "medium", "long"))
    batch = clock.issue("batch")
    registry = builtin_registry()
    block = registry.create("cv.align_exposures", parameters={
        "reference_index": 1, "border_mode": border,
    })
    result = block.execute({"exposures": ValuePacket(group, {"ticket": batch})})
    assert result is not None, block.error
    info = result.outputs["info"]
    assert [(v["dx"], v["dy"]) for v in info["shifts"]] == [(3, -2), (0, 0), (-4, 3)]
    assert info["overlap_xywh"] == [3, 3, 185, 123]
    x, y, w, h = info["output_xywh"]
    assert [x, y, w, h] == ([3, 3, 185, 123] if border == "crop" else [0, 0, 192, 128])
    aligned = result.outputs["aligned"]
    assert aligned.labels == group.labels
    for packet, entry, gain in zip(aligned.items, entries, gains, strict=True):
        assert packet.payload["ticket"] == entry.payload["ticket"]
        spatial = packet.payload["spatial"]
        assert spatial["root_id"] == "frame_1"
        assert spatial["image_size"] == [w, h]
        np.testing.assert_array_equal(spatial["to_root"], [[1, 0, x], [0, 1, y], [0, 0, 1]])
        valid = packet.value[3 - y : 126 - y, 3 - x : 188 - x]
        np.testing.assert_array_equal(valid, np.rint(np.clip(raw[3:126, 3:188] * gain, 0, 255)))
    if border == "zero_pad":
        assert not np.any(aligned.items[0].value[:, :3])
    fusion = registry.create("cv.fuse_exposures")
    fused = fusion.execute({"exposures": packet_for(result, "aligned")})
    assert fused is not None, fusion.error
    assert fused.outputs["image"].shape == (h, w)
    assert fused.payloads["image"]["ticket"] == batch
    assert fused.payloads["image"]["spatial"]["root_id"] == "frame_1"


def test_mertens_native_float_and_mask_preserve_bgr_and_reference():
    rgb = np.random.default_rng(24).integers(10, 140, (24, 32, 3), dtype=np.uint8)
    brighter = np.minimum(rgb.astype(np.uint16) * 2, 255).astype(np.uint8)
    images = [ImageArray(image[..., ::-1].copy(), "bgr") for image in (rgb, brighter)]
    group = FlowGroup(tuple(ValuePacket(i, {}) for i in images), ("short", "long"))
    mask = np.zeros(rgb.shape[:2], np.uint8)
    mask[5:20, 7:22] = 255
    block = builtin_registry().create("cv.fuse_exposures", parameters={
        "reference_index": 1, "exposure_weight": 1.0, "output_depth": "float32",
    })
    result = block.execute({"exposures": group, "mask": mask})
    assert result is not None, block.error
    output = result.outputs["image"]
    assert output.color_format == "bgr" and output.dtype == np.float32
    native = cv2.createMergeMertens(1.0, 1.0, 1.0).process([rgb, brighter])[..., ::-1]
    np.testing.assert_allclose(output[mask != 0], native[mask != 0], atol=1e-6)
    np.testing.assert_allclose(output[mask == 0], images[1][mask == 0] / 255.0, atol=1e-7)


def test_invalid_exposure_batches_and_featureless_alignment_fail_safely(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Called native fusion on an invalid batch")

    registry = builtin_registry()
    fusion = registry.create("cv.fuse_exposures")
    monkeypatch.setattr(cv2, "createMergeMertens", unexpected)
    flat = np.zeros((32, 32), np.uint8)
    malformed = FlowGroup((ValuePacket(flat, {}), ValuePacket(flat[:-1], {})), ("a", "b"))
    assert fusion.execute({"exposures": malformed}) is None
    assert "matching size" in fusion.error
    group = FlowGroup((ValuePacket(flat, {}), ValuePacket(flat.copy(), {})), ("a", "b"))
    align = registry.create("cv.align_exposures")
    assert align.execute({"exposures": group}) is None
    assert "no usable luminance variation" in align.error
