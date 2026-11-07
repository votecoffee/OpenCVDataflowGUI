"""Check native photo filter parity and selected output semantics."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray, image_format


@pytest.mark.parametrize(
    "type_id, binding, parameters, native_options",
    [
        ("cv.edge_preserving_photo", cv2.edgePreservingFilter, {}, {"flags": cv2.RECURS_FILTER}),
        (
            "cv.edge_preserving_photo", cv2.edgePreservingFilter,
            {"method": "normalized_convolution"}, {"flags": cv2.NORMCONV_FILTER},
        ),
        ("cv.detail_enhance", cv2.detailEnhance, {}, {}),
        ("cv.pencil_sketch", cv2.pencilSketch, {"shade_factor": 0.05}, {"shade_factor": 0.05}),
        ("cv.stylize_photo", cv2.stylization, {}, {}),
    ],
)
def test_photo_filter_matches_native_inside_mask(type_id, binding, parameters, native_options):
    color = np.random.default_rng(71).integers(0, 256, size=(29, 35, 3), dtype=np.uint8)
    source = ImageArray(color, "rgb")
    mask = np.zeros(color.shape[:2], dtype=np.uint8)
    mask[6:22, 7:25] = 255
    selected = mask != 0
    block = builtin_registry().create(
        type_id, parameters={**parameters, "sigma_s": 18, "sigma_r": 0.2}
    )
    result = block.execute({"image": source, "mask": mask})
    assert result is not None, block.error
    bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    expected = binding(bgr, sigma_s=18, sigma_r=0.2, **native_options)
    if type_id == "cv.pencil_sketch":
        gray, expected = expected
        np.testing.assert_array_equal(result.outputs["gray"][selected], gray[selected])
        assert not result.outputs["gray"][~selected].any()
    output = result.outputs["image"]
    assert image_format(output) == "rgb"
    np.testing.assert_array_equal(output[selected], expected[..., ::-1][selected])
    np.testing.assert_array_equal(output[~selected], color[~selected])


def test_photo_filter_refuses_invalid_input_before_native_call(monkeypatch):
    monkeypatch.setattr(cv2, "stylization", lambda *a, **k: pytest.fail("Called on invalid input"))
    block = builtin_registry().create("cv.stylize_photo")
    for image in (np.zeros((12, 14), np.uint8), np.zeros((1, 14, 3), np.uint8)):
        assert block.execute({"image": image}) is None
        assert "uint8 BGR or RGB" in block.error
    block.parameters["sigma_r"] = 0
    assert block.execute({"image": np.zeros((12, 14, 3), np.uint8)}) is None
    assert block.error_kind == "invalid"
