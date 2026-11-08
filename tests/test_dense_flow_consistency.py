"""Verify dense optical-flow consistency and the native sparse-to-dense wrapper."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.sequence_tools import _flow_consistency
from opencv_dataflow.toolbox import block_entry


def test_dense_flow_consistency_samples_the_backward_field_at_forward_endpoints():
    forward = np.zeros((5, 8, 2), np.float32)
    forward[..., 0] = 1.5
    backward = np.zeros_like(forward)
    backward[..., 0] = -1.5
    backward[2, 3:5, 0] = 0.5
    selected = np.ones((5, 8), np.uint8)
    selected[0, 0] = 0
    mask, error, report = _flow_consistency(forward, backward, selected, 0.75)
    assert mask.dtype == np.uint8 and error.dtype == np.float32
    assert error[2, 1] == 1.0 and mask[2, 1] == 0
    assert error[2, 2] == 2.0 and mask[2, 2] == 0
    assert error[2, 4] == 0 and mask[2, 4] == 255
    assert np.isnan(error[0, 0]) and np.isnan(error[:, -2:]).all()
    assert np.count_nonzero(mask[:, -2:]) == 0
    assert report == {"selected": 39, "evaluable": 29, "consistent": 26, "tolerance_pixels": 0.75}


@pytest.mark.parametrize("kind", ["farneback", "dis", "lk_grid_flow"])
def test_dense_check_is_opt_in_and_preserves_the_forward_field(kind):
    registry = builtin_registry()
    params = {"max_gap_ms": 0}
    plain = registry.create(f"sequence.{kind}", parameters=params)
    checked = registry.create(f"sequence.{kind}", parameters={**params, "consistency_tolerance": 1})
    a = np.random.default_rng(18).integers(0, 256, (96, 128), dtype=np.uint8)
    a = cv2.GaussianBlur(a, (3, 3), 0.55)
    b = cv2.warpAffine(
        a, np.float32([[1, 0, 3], [0, 1, 2]]), (128, 96), borderMode=cv2.BORDER_REFLECT
    )
    for block in (plain, checked):
        first = block.execute({"image": a})
        assert first is not None and not first.outputs["ready"], block.error
        assert first.outputs["consistency_mask"] is NO_VALUE
    unchecked = plain.execute({"image": b})
    result = checked.execute({"image": b})
    assert unchecked is not None and result is not None, (plain.error, checked.error)
    assert unchecked.outputs["consistency_mask"] is NO_VALUE
    assert unchecked.outputs["consistency_error"] is NO_VALUE
    assert result.outputs["ready"]
    vectors = result.outputs["flow"].vectors
    np.testing.assert_array_equal(vectors, unchecked.outputs["flow"].vectors)
    inner = np.s_[24:-24, 24:-24]
    np.testing.assert_allclose(np.median(vectors[inner], axis=(0, 1)), [3, 2], atol=0.35)
    assert (result.outputs["consistency_mask"][inner] == 255).mean() > 0.95
    assert result.outputs["status"]["consistency"]["selected"] == a.size
    assert result.outputs["status"]["consistency"]["evaluable"] < a.size
    assert checked.execute({"image": b, "reset": True}).outputs["consistency_error"] is NO_VALUE


def test_dense_grid_excludes_untrackable_regions_and_resets_after_a_blank_frame():
    registry = builtin_registry()
    block = registry.create(
        "sequence.lk_grid_flow",
        parameters={
            "grid_step": 6,
            "window": 11,
            "levels": 1,
            "max_gap_ms": 0,
            "consistency_tolerance": 0.8,
        },
    )
    a = np.zeros((96, 128), np.uint8)
    a[:, :60] = np.random.default_rng(19).integers(0, 256, (96, 60), dtype=np.uint8)
    b = np.roll(a, (1, 2), axis=(0, 1))
    block.execute({"image": a})
    result = block.execute({"image": b})
    assert result is not None, block.error
    assert block_entry(block).path[:2] == ("OpenCV", "Video Analysis (video)")
    assert (result.outputs["support_mask"][20:-20, 16:42] == 255).all()
    assert not result.outputs["support_mask"][:, 100:].any()
    assert not result.outputs["consistency_mask"][:, 100:].any()
    assert np.isnan(result.outputs["consistency_error"][:, 100:]).all()
    assert not result.outputs["flow"].vectors[:, 100:].any()

    blank = np.zeros_like(a)
    block.execute({"image": blank, "reset": True})
    assert block.execute({"image": blank}) is None
    assert "no trackable points" in block.error
    assert not block.execute({"image": a}).outputs["ready"]


@pytest.mark.parametrize(
    ("field", "value"),
    [("grid_step", 1), ("window", 4), ("levels", -1), ("minimum_support", 0)],
)
def test_dense_grid_rejects_invalid_editor_values(field, value):
    block = builtin_registry().create("sequence.lk_grid_flow")
    with pytest.raises(ValidationError) as exc:
        block.validate_parameters({**block.parameters, field: value})
    assert exc.value.fields == (field,)
