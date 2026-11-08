"""Verify width estimation, clipped-edge classification, and strip geometry."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock


def test_width_uses_only_the_run_touching_center_and_distinguishes_failure_types():
    image = np.zeros((11, 6), np.uint8)
    image[3:8, 0] = 255
    image[1, 0] = 255
    image[4:7, 1] = 255
    image[1:3, 2] = 255
    image[:8, 3] = 255
    image[3:8, 4:] = 255
    validity = np.full_like(image, 255)
    validity[8, 4] = 0
    validity[5, 5] = 0
    tool = builtin_registry().create("inspect.strip_width", parameters={
        "minimum_width": 4, "maximum_width": 6,
    })
    result = tool.execute({"image": image, "mask": validity})
    assert result is not None, tool.error
    profile = result.outputs["profile"]
    assert profile["widths"] == [5, 3, None, None, None, None]
    assert profile["status"] == [
        "pass", "out_of_range", "missing", "unmeasurable", "unmeasurable", "unmeasurable",
    ]
    assert result.outputs["measurements"]["mean"] == 4
    assert result.outputs["pass_fraction"] == pytest.approx(1 / 6)
    assert [reason["code"] for reason in result.fail_reasons] == [
        "strip_width_unmeasurable", "strip_foreground_missing", "strip_width_out_of_range",
    ]
    assert set(result.drawings) == {
        "Width out of range", "Missing foreground", "Width unmeasurable",
    }


def test_width_uses_sampling_scale_and_keeps_the_strip_frame_and_ticket():
    registry = builtin_registry()
    image = np.zeros((14, 24), np.uint8)
    image[4:9] = 255
    ticket = TicketClock().issue("camera")
    source = ValuePacket(image, {"ticket": ticket, "spatial": frame([24, 14], "camera")})
    unwrap = registry.create("cv.unwrap_path_strip", parameters={
        "vertices": [[2, 6], [21, 6]], "half_width": 4, "cross_samples": 21,
        "interpolation": "nearest",
    })
    strip = unwrap.execute({"image": source})
    assert strip is not None, unwrap.error
    tool = registry.create(
        "inspect.strip_width", parameters={"minimum_width": 4, "maximum_width": 6}
    )
    mask = strip.outputs["valid_mask"]
    sampling = strip.outputs["sampling"]
    result = tool.execute({"image": strip.outputs["image"], "mask": mask, "sampling": sampling})
    assert result is not None and result.passed, tool.error
    assert result.outputs["profile"]["units"] == "source px"
    assert result.outputs["profile"]["widths"] == pytest.approx([5.2] * 20)
    assert result.payloads["profile"]["spatial"] == sampling["strip_spatial"]
    assert result.payloads["pass_fraction"]["ticket"] == ticket
    assert tool.execute({"image": strip.outputs["image"], "sampling": sampling}) is None
    assert "validity mask" in tool.error
    wrong = {**sampling, "strip_spatial": frame([20, 21], "other")}
    assert tool.execute({"image": strip.outputs["image"], "mask": mask, "sampling": wrong}) is None
    assert "pixel grid" in tool.error


def test_width_classification_thresholds_have_independent_denominators():
    image = np.zeros((7, 5), np.uint8)
    image[2:5, :2] = 255
    image[3, 2] = 255
    image[:, 4] = 255
    tool = builtin_registry().create("inspect.strip_width", parameters={
        "minimum_width": 2, "minimum_valid_fraction": 0.8,
        "minimum_coverage": 0.75, "minimum_in_range_fraction": 0.66,
    })
    result = tool.execute({"image": image})
    assert result.passed
    assert result.outputs["pass_fraction"] == 0.4
    assert result.outputs["measurements"]["coverage"] == 0.75
    assert result.outputs["measurements"]["in_range_fraction"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("mask_value, reason", [
    (255, "strip_foreground_missing"),
    (0, "strip_width_unmeasurable"),
])
def test_width_requires_some_measurable_foreground_even_when_fraction_checks_are_disabled(
    mask_value, reason
):
    image = np.zeros((5, 8), np.uint8)
    tool = builtin_registry().create("inspect.strip_width", parameters={
        "minimum_coverage": 0, "minimum_valid_fraction": 0, "minimum_in_range_fraction": 0,
    })
    result = tool.execute({"image": image, "mask": np.full_like(image, mask_value)})
    assert not result.passed
    assert result.outputs["measurements"]["minimum"] is None
    assert [r["code"] for r in result.fail_reasons] == [reason]


def test_width_uses_inclusive_limits_for_fractional_steps():
    image = np.zeros((5, 4), np.uint8)
    image[1:4] = 255
    spatial = frame([4, 5], "strip")
    sampling = {
        "model": "path_strip_v1", "strip_spatial": spatial, "center_row": 2, "cross_step": 0.1,
    }
    tool = builtin_registry().create("inspect.strip_width", parameters={
        "minimum_width": 0.3, "maximum_width": 0.3,
    })
    result = tool.execute({
        "image": ValuePacket(image, {"spatial": spatial}), "sampling": sampling,
        "mask": np.full_like(image, 255),
    })
    assert result is not None and result.passed, tool.error
