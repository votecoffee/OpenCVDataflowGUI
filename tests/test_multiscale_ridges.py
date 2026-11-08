"""Verify native multiscale filtering, winning scales, and region selection."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock


@pytest.mark.parametrize("polarity", ["dark", "bright", "both"])
def test_multiscale_ridges_match_sato_and_report_selected_scale(polarity):
    sato = pytest.importorskip("skimage.filters").sato
    rng = np.random.default_rng(91)
    bgr = rng.integers(2000, 4000, (24, 30, 3), dtype=np.uint16)
    rgb = ImageArray(bgr[:, :, ::-1].copy(), "rgb")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ticket = TicketClock().issue("camera")
    spatial = frame([30, 24], "fixture")
    mask = np.zeros(gray.shape, dtype=np.uint8)
    mask[3:20, 4:27] = 255
    region = {
        "model": "image_region_v1", "image_size": [30, 24],
        "shape": "rectangle", "rectangle": [6, 2, 24, 15],
    }
    selected = np.zeros(gray.shape, dtype=bool)
    selected[3:17, 6:27] = True
    sigmas = (1, 2.5, 4)
    dark = [sato(gray, sigmas=(sigma,), mode="constant", cval=1200) for sigma in sigmas]
    bright = [sato(-gray, sigmas=(sigma,), mode="constant", cval=-1200) for sigma in sigmas]
    scores = {"dark": dark, "bright": bright, "both": np.maximum(dark, bright)}[polarity]
    expected = np.max(scores, axis=0)
    scales = np.take(sigmas, np.argmax(scores, axis=0))
    block = builtin_registry().create("inspect.multiscale_ridges", parameters={
        "polarity": polarity, "sigmas": list(reversed(sigmas)),
        "border": "constant", "border_value": 1200,
    })
    result = block.execute({
        "image": ValuePacket(rgb, {"ticket": ticket, "spatial": spatial}),
        "mask": mask, "region": region,
    })
    assert result is not None, block.error
    np.testing.assert_array_equal(result.outputs["image"], np.where(selected, expected, 0))
    np.testing.assert_array_equal(
        result.outputs["scale"], np.where(selected & (expected > 0), scales, 0)
    )
    for name in ("image", "scale"):
        assert result.outputs[name].dtype == np.float32
        assert result.payloads[name]["ticket"] == ticket
        assert result.payloads[name]["spatial"] == spatial


def test_multiscale_ridges_separates_scale_at_dark_and_bright_ridge_centers():
    pytest.importorskip("skimage.filters")
    x = np.arange(120, dtype=np.float32)
    line = 120 - 65 * np.exp(-0.5 * ((x - 30) / 0.85) ** 2)
    line += 65 * np.exp(-0.5 * ((x - 90) / 3) ** 2)
    image = np.tile(line, (40, 1))
    block = builtin_registry().create("inspect.multiscale_ridges", parameters={
        "polarity": "both", "sigmas": [1, 2, 4],
    })
    result = block.execute({"image": image})
    assert result is not None, block.error
    np.testing.assert_array_equal(result.outputs["scale"][20, [30, 90]], [1, 4])
    assert np.all(result.outputs["image"][20, [30, 90]] > 10)
    flat = block.execute({"image": np.zeros_like(image)})
    assert not flat.outputs["image"].any() and not flat.outputs["scale"].any()


@pytest.mark.parametrize("sigmas", [[], [1, 1], [True], ["1"], [0.5], [13]])
def test_multiscale_ridges_validates_scale_list(sigmas):
    block = builtin_registry().create("inspect.multiscale_ridges", parameters={"sigmas": sigmas})
    assert block.execute({"image": np.zeros((8, 10), np.uint8)}) is None
    assert "distinct numeric sigmas" in block.error
