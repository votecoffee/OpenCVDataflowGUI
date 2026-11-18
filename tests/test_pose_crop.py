"""Pose sampling, measured support, frame/ticket provenance and contained failures."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.poses import local_offset, make_pose
from opencv_dataflow.tickets import TicketClock


@pytest.fixture
def make():
    blocks = []

    def create(**p):
        block = builtin_registry().create(
            "cv.crop_pose",
            parameters={"width": 4, "height": 3, "local_width": 4.0, "local_height": 3.0, **p},
        )
        blocks.append(block)
        return block

    yield create
    for block in blocks:
        block.close()


def pose_at(image, x, y, angle=0, payload=None):
    return make_pose(local_offset(x, y, angle), list(image.shape[1::-1]), payload or {})


def test_fractional_ramp_pixel_centers_and_resized_local_span(make):
    y, x = np.indices((20, 20), dtype=np.float32)
    image = 2 * x + 11 * y
    pose = pose_at(image, 8.125, 7.25)
    block = make(local_width=8.0)
    result = block.execute({"image": image, "pose": pose})
    assert result is not None, block.error
    yy, xx = np.indices((3, 4))
    sx = 8.125 + (xx - 1.5) * 2
    sy = 7.25 + yy - 1
    np.testing.assert_allclose(result.outputs["image"], 2 * sx + 11 * sy, atol=1e-5)
    np.testing.assert_allclose(
        result.outputs["geometry"]["output_to_input"], [[2, 0, 5.125], [0, 1, 6.25], [0, 0, 1]]
    )
    assert result.outputs["coverage"] == 1 and np.all(result.outputs["valid_mask"] == 255)


@pytest.mark.parametrize("method", ["nearest", "linear", "cubic", "lanczos4"])
def test_native_affine_sampling_preserves_shear_and_integer_channel_units(make, method):
    rng = np.random.default_rng(18)
    pixels = rng.integers(0, 50000, (48, 48, 3), dtype=np.uint16)
    image = ImageArray(pixels, "rgb")
    matrix = np.array([[1.2, 0.3, 24], [-0.4, 0.8, 24], [0, 0, 1]])
    pose = make_pose(matrix, [48, 48], {})
    block = make(interpolation=method)
    result = block.execute({"image": image, "pose": pose})
    assert result is not None, block.error
    expected = matrix @ np.array([[1, 0, -1.5], [0, 1, -1], [0, 0, 1]])
    flag = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos4": cv2.INTER_LANCZOS4,
    }[method]
    reference = cv2.warpAffine(pixels, expected[:2], (4, 3), flags=flag | cv2.WARP_INVERSE_MAP)
    np.testing.assert_array_equal(result.outputs["image"], reference)
    assert (
        result.outputs["image"].dtype == np.uint16 and result.outputs["image"].color_format == "rgb"
    )
    np.testing.assert_allclose(result.outputs["geometry"]["output_to_input"], expected)
    np.testing.assert_array_equal(image, pixels)


def test_shared_root_pose_reprojects_and_moves_spatial_and_pyramid_payload(make):
    image = np.full((40, 40), 100, np.uint8)
    root = frame([80, 80], "camera")
    grid = {**root, "image_size": [40, 40], "to_root": [[2, 0, 0.5], [0, 2, 0.5], [0, 0, 1]]}
    ticket = TicketClock("camera").issue("camera")
    context = {
        "spatial": grid,
        "ticket": ticket,
        "pyramid": {"level": 1, "largest_size": [80, 80], "to_largest": grid["to_root"]},
        "custom": {"kept": True},
    }
    pose = make_pose(local_offset(40, 40, 30), [80, 80], {"spatial": root, "ticket": ticket})
    block = make()
    result = block.execute({"image": ValuePacket(image, context), "pose": pose})
    assert result is not None, block.error
    mapping = np.asarray(result.outputs["geometry"]["output_to_input"])
    expected = np.asarray(grid["to_root"]) @ mapping
    for name in result.outputs:
        payload = result.payloads[name]
        assert payload["ticket"] == ticket and payload["custom"] == {"kept": True}
        np.testing.assert_allclose(payload["spatial"]["to_root"], expected)
        np.testing.assert_allclose(payload["pyramid"]["to_largest"], expected)
    assert result.payloads["image"]["spatial"]["image_size"] == [4, 3]


@pytest.mark.parametrize(
    "border", ["constant", "replicate", "reflect", "reflect_101", "reflect_edge", "wrap"]
)
def test_extrapolation_does_not_invent_measured_support(make, border):
    image = np.full((10, 10), 200, np.uint8)
    pose = pose_at(image, 0, 0)
    block = make(
        width=4,
        height=4,
        local_height=4.0,
        bounds_policy="pad",
        border=border,
        minimum_coverage=0.5,
    )
    result = block.execute({"image": image, "pose": pose})
    assert result is not None, block.error
    assert result.outputs["coverage"] == 0.25 and not result.passed
    assert result.fail_reasons[0]["code"] == "crop.coverage"
    assert np.count_nonzero(result.outputs["valid_mask"]) == 4
    strict = make(width=4, height=4, local_height=4.0)
    assert strict.execute({"image": image, "pose": pose}) is None and not strict.current_emissions


def test_selection_mask_and_alpha_are_filled_without_changing_source(make):
    image = np.full((12, 12, 4), 200, np.uint8)
    image[..., 3] = 255
    image[5, 5, 3] = 0
    mask = np.full((12, 12), 255, np.uint8)
    mask[6, 6] = 0
    original = image.copy()
    block = make(bounds_policy="pad", interpolation="nearest", width=4, height=4, local_height=4.0)
    result = block.execute({"image": image, "pose": pose_at(image, 5.5, 5.5), "mask": mask})
    assert result is not None and result.outputs["coverage"] == 14 / 16, block.error
    assert np.count_nonzero(result.outputs["valid_mask"] == 0) == 2
    np.testing.assert_array_equal(result.outputs["image"][1, 1], [0, 0, 0, 0])
    np.testing.assert_array_equal(image, original)
    assert result.drawings
    block.set_graphics_mode("off")
    quiet = block.execute({"image": image, "pose": pose_at(image, 5.5, 5.5), "mask": mask})
    np.testing.assert_array_equal(quiet.outputs["image"], result.outputs["image"])
    assert not quiet.drawings


def test_wrong_root_ticket_and_projective_pose_are_rejected(make):
    image = np.full((20, 20), 100, np.uint8)
    ticket = TicketClock("a").issue("a")
    other = TicketClock("b").issue("b")
    source = ValuePacket(image, {"ticket": ticket, "spatial": frame([20, 20], "a")})
    block = make()
    for pose in [
        make_pose(local_offset(10, 10, 0), [20, 20], {"spatial": frame([20, 20], "b")}),
        pose_at(image, 10, 10, payload={"ticket": other}),
        make_pose([[1, 0, 10], [0, 1, 10], [0.01, 0, 1]], [20, 20], {}),
    ]:
        assert (
            block.execute({"image": source, "pose": pose}) is None and not block.current_emissions
        )


def test_inspection_child_ticket_accepts_the_known_parent_pose(make):
    image = np.full((20, 20), 100, np.uint8)
    clock = TicketClock("camera")
    parent = clock.issue("camera")
    child = clock.issue("part")
    context = {"ticket": child, "inspection": {"parent_ticket": parent, "feature_id": 7}}
    pose = pose_at(image, 10, 10, payload={"ticket": parent})
    result = make().execute(
        {"image": ValuePacket(image, context), "pose": ValuePacket(pose, context)}
    )
    assert result is not None and result.payloads["image"]["ticket"] == child
    assert result.payloads["image"]["inspection"]["parent_ticket"] == parent


@pytest.mark.parametrize(
    "settings",
    [
        {"width": 0},
        {"height": 1.0},
        {"local_width": 0},
        {"local_height": "3"},
        {"interpolation": "area"},
        {"bounds_policy": "clip"},
        {"minimum_coverage": 2},
        {"border_value": True},
        {"maximum_crop_pixels": 1},
    ],
)
def test_parameter_validation_identifies_invalid_fields(make, settings):
    with pytest.raises(ValueError):
        make(**settings).validate_parameters()


def test_budget_rejection_precedes_sampling_and_native_failure_recovers(make, monkeypatch):
    image = np.ones((20, 20), np.uint8)
    pose = pose_at(image, 10, 10)
    native = cv2.warpAffine
    monkeypatch.setattr(cv2, "warpAffine", lambda *a, **k: pytest.fail("Budget reached sampling"))
    limited = make(maximum_image_pixels=10)
    assert limited.execute({"image": image, "pose": pose}) is None
    block = make()
    monkeypatch.setattr(cv2, "warpAffine", native)
    assert block.execute({"image": image, "pose": pose}) is not None

    def broken(*a, **k):
        raise cv2.error("controlled sampling failure")

    monkeypatch.setattr(cv2, "warpAffine", broken)
    assert block.execute({"image": image, "pose": pose}) is None and not block.current_emissions
    monkeypatch.setattr(cv2, "warpAffine", native)
    assert block.execute({"image": image, "pose": pose}) is not None


def test_disabled_annotation_emits_a_completed_empty_layer_instead_of_waiting():
    from opencv_dataflow.graphics_layers import GraphicsBundle
    from opencv_dataflow.run_profiles import PROFILES

    block = builtin_registry().create("cv.annotation_layer")
    block.run_profile = PROFILES["lean"]
    block.set_graphics_mode("off")
    try:
        result = block.execute({"image": np.ones((20, 20), np.uint8)})
        assert result is not None and block.state == "pass", block.error
        assert result.outputs["result_graphics"] == GraphicsBundle(())
        assert block.current_emissions["result_graphics"]
    finally:
        block.close()
