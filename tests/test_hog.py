"""Verify native HOG order, geometry and optional orientation graphics."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.hog import _cell_votes
from opencv_dataflow.images import ImageArray


@pytest.mark.parametrize("signed", [False, True])
def test_hog_matches_native_vector_for_color_window(signed):
    pixels = np.random.default_rng(78).integers(0, 256, (32, 48, 3), dtype=np.uint8)
    block = builtin_registry().create(
        "inspect.hog", parameters={"signed_gradient": signed, "win_sigma": 5.0}
    )
    result = block.execute({"image": ImageArray(pixels, "rgb")})
    assert result is not None, block.error
    native = cv2.HOGDescriptor(
        (48, 32), (16, 16), (8, 8), (8, 8), 9,
        _winSigma=5.0, _gammaCorrection=True, _signedGradient=signed,
    ).compute(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
    np.testing.assert_array_equal(result.outputs["descriptor"], native.reshape(-1))
    info = result.outputs["info"]
    assert info["block_grid"] == [5, 3]
    assert info["descriptor_length"] == 540
    assert info["settings"]["signed_gradient"] is signed
    objects = result.drawings["Dominant cell gradients"].objects
    assert len(objects) == 24
    assert {obj.kind for obj in objects} == {"arrow" if signed else "polyline"}
    block.graphics_mode = "off"
    quiet = block.execute({"image": ImageArray(pixels, "rgb")})
    assert quiet.outputs == result.outputs
    assert not quiet.drawings


def test_hog_cell_votes_keep_native_column_order_and_average_overlaps():
    # Two adjacent blocks, two cells in each, one row and two bins.
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8], np.float32)
    votes = _cell_votes(values, (2, 1), (2, 1), (1, 1), (3, 1), 2)
    np.testing.assert_array_equal(votes[:, 0], [[1, 2], [4, 5], [7, 8]])


def test_hog_rejects_invalid_geometry_before_calling_opencv(monkeypatch):
    monkeypatch.setattr(cv2, "HOGDescriptor", lambda *a, **k: pytest.fail("Called OpenCV"))
    registry = builtin_registry()
    block = registry.create("inspect.hog")
    assert block.execute({"image": np.zeros((19, 32), np.uint8)}) is None
    assert "whole HOG blocks" in block.error
    block.parameters["block_stride"] = [5, 8]
    assert block.execute({"image": np.zeros((32, 32), np.uint8)}) is None
    assert "whole cells" in block.error


def test_hog_flat_window_has_zero_vector_and_no_glyphs():
    block = builtin_registry().create("inspect.hog")
    result = block.execute({"image": np.full((128, 64), 200, np.uint8)})
    assert result is not None, block.error
    assert len(result.outputs["descriptor"]) == 3780
    assert not any(result.outputs["descriptor"])
    assert not result.drawings
