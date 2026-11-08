"""Check masked pair counts and directional co-occurrence properties."""

import json

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock


@pytest.mark.parametrize("symmetric", [False, True])
def test_glcm_counts_both_selected_endpoints_and_uses_signed_offsets(symmetric):
    feature = pytest.importorskip("skimage.feature")
    image = np.array([
        [10, 20, 30, 40],
        [20, 40, 30, 10],
        [30, 20, 10, 30],
        [40, 10, 40, 20],
    ], np.uint16)
    levels = ((image.astype(int) - 10) * 4 // 30).clip(0, 3)
    mask = np.ones(image.shape, np.uint8)
    mask[1, 2] = 0
    region = {
        "model": "image_region_v1", "image_size": [4, 4],
        "shape": "rectangle", "rectangle": [1, 0, 3, 4],
    }
    valid = mask.astype(bool)
    valid[:, 0] = False
    offsets = [[1, 0], [-1, 1]]
    spatial = frame([4, 4], "surface")
    ticket = TicketClock().issue("camera")
    block = builtin_registry().create("inspect.gray_level_cooccurrence", parameters={
        "levels": 4, "range_mode": "manual", "range_min": 10, "range_max": 40,
        "symmetric": symmetric, "offsets": offsets, "offset_index": 1, "metric": "entropy",
    })
    result = block.execute({
        "image": ValuePacket(image, {"ticket": ticket, "spatial": spatial}),
        "mask": mask, "region": region,
    })
    assert result is not None, block.error
    measurements = result.outputs["measurements"]
    json.dumps(measurements, allow_nan=False)
    assert measurements["range"] == [10, 40]
    for row, (dx, dy) in zip(measurements["offsets"], offsets, strict=True):
        counts = np.zeros((4, 4), np.int32)
        for y, x in np.argwhere(valid):
            if 0 <= x + dx < 4 and 0 <= y + dy < 4 and valid[y + dy, x + dx]:
                counts[levels[y, x], levels[y + dy, x + dx]] += 1
        assert row["offset"] == [dx, dy]
        assert row["pair_count"] == counts.sum()
        if symmetric:
            counts += counts.T.copy()
        assert row["contrast"] == pytest.approx(feature.graycoprops(counts[..., None, None])[0, 0])
    assert result.outputs["value"] == pytest.approx(row["entropy"])
    np.testing.assert_allclose(result.outputs["matrix"], counts / counts.sum(), atol=1e-7)
    assert result.outputs["matrix"].dtype == np.float32
    assert result.payloads["matrix"]["ticket"] == ticket
    assert "spatial" not in result.payloads["matrix"]
    assert "Co-occurrence matrix" in result.graphics


def test_glcm_matches_unmasked_native_and_checks_inclusive_limit():
    feature = pytest.importorskip("skimage.feature")
    image = np.tile(np.array([0, 80, 160, 255], dtype=np.uint8), (6, 1))
    native = feature.graycomatrix(image, [1], [0, np.pi / 2], levels=256, symmetric=True)
    expected = feature.graycoprops(native, "contrast")[0]
    block = builtin_registry().create("inspect.gray_level_cooccurrence", parameters={
        "levels": 256, "check_threshold": True, "threshold": float(expected[0]),
    })
    result = block.execute({"image": image})
    assert result is not None and result.passed, block.error
    np.testing.assert_allclose([r["contrast"] for r in result.outputs["measurements"]["offsets"]],
                               expected)
    block.parameters["threshold"] = float(expected[0]) - 1
    result = block.execute({"image": image})
    assert result is not None and not result.passed, block.error
    assert result.fail_reasons[0]["code"] == "texture_cooccurrence_out_of_tolerance"


def test_glcm_rejects_no_selected_pairs_and_invalid_offsets():
    image = np.zeros((8, 9), np.uint8)
    mask = image.copy()
    mask[2, 3] = 255
    block = builtin_registry().create("inspect.gray_level_cooccurrence")
    assert block.execute({"image": image, "mask": mask}) is None
    assert "no selected pixel pairs" in block.error
    block.parameters["offsets"] = [[1, 0], [1, 0]]
    assert block.execute({"image": image}) is None
    assert "distinct" in block.error
