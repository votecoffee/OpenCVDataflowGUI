"""Native paired arithmetic, quantity/geometry references and guarded result transport."""

import copy
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.images import DepthArray, ImageArray
from opencv_dataflow.learned_stereo import LearnedStereo
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.run_profiles import PROFILES
from opencv_dataflow.stereo3d import ReprojectDisparity
from opencv_dataflow.tickets import TicketClock

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/learned_stereo"


def tool(**parameters):
    return LearnedStereo(
        parameters={
            "model_path": str(BASE / "models/plane/model.onnx"),
            "width": 31,
            "height": 23,
            "engine": "classic",
            **parameters,
        }
    )


def sources(encoding="bgr", dtype="uint8", right="right"):
    result = []
    for name in ("left", right):
        image = cv2.imread(str(BASE / "assets" / (name + ".png")))
        if encoding in ("rgb", "rgba"):
            image = image[..., ::-1]
        if encoding in ("rgba", "bgra"):
            image = np.dstack((image, np.full(image.shape[:2], 123, np.uint8)))
        if encoding == "gray":
            image = image[..., 2]
        if dtype == "uint16":
            image = image.astype(np.uint16) * 257
        elif dtype == "float32":
            image = image.astype(np.float32) / 255
        result.append(ImageArray(image, encoding))
    return dict(zip(("left", "right"), result))


@pytest.mark.parametrize("encoding", ["bgr", "rgb", "bgra", "rgba", "gray"])
@pytest.mark.parametrize("dtype", ["uint8", "uint16", "float32"])
def test_native_both_views_known_plane_preserve_pixels_depth_encoding(encoding, dtype):
    inputs = sources(encoding, dtype)
    before = copy.deepcopy(inputs)
    block = tool()
    try:
        result = block.execute(inputs)
        assert result is not None and result.passed, block.error
        assert result.outputs["image"].color_format == encoding
        for name in inputs:
            np.testing.assert_array_equal(inputs[name], before[name])
        np.testing.assert_array_equal(result.outputs["image"], inputs["left"])
        disparity = result.outputs["disparity"]
        valid = result.outputs["valid_mask"] != 0
        assert valid[:, 4:].all() and not valid[:, :3].any()
        np.testing.assert_allclose(disparity[valid], 3, atol=8e-6)
        assert np.isnan(disparity[~valid]).all()
        assert result.outputs["info"]["units"] == "source_pixels"
        changed = sources(encoding, dtype, right="subpixel")
        result = block.execute(changed)
        np.testing.assert_allclose(result.outputs["disparity"][:, 2:], 1.5, atol=8e-6)
        assert result.outputs["info"]["native_calls"] == 1
    finally:
        block.close()


@pytest.mark.parametrize("layout", ["NHW", "NCHW", "NHWC", "HW", "CHW", "HWC"])
@pytest.mark.parametrize(
    "units,factor", [("network_pixels", 1), ("head_pixels", 31 / 10), ("source_pixels", 1)]
)
def test_layout_scale_sign_and_source_units_are_explicit(monkeypatch, layout, units, factor):
    head = np.full((8, 10), -1.5, np.float32)
    encoded = {
        "NHW": head[None],
        "NCHW": head[None, None],
        "NHWC": head[None, :, :, None],
        "HW": head,
        "CHW": head[None],
        "HWC": head[:, :, None],
    }[layout]
    block = tool(
        layout=layout,
        head_units=units,
        head_scale=2.0,
        head_offset=-1.0,
        head_sign="right_minus_left",
    )
    monkeypatch.setattr(block, "infer_tensors", lambda *a, **k: ({"d": encoded}, "fixture"))
    try:
        result = block.execute(sources())
        assert result is not None, block.error
        actual = result.outputs["disparity"]
        valid = result.outputs["valid_mask"] != 0
        np.testing.assert_allclose(actual[valid], 4 * factor, atol=2e-6)
        assert result.outputs["info"]["geometry"]["head_to_source_disparity_factor"] == -factor
    finally:
        block.close()


def test_fractional_right_correspondence_requires_both_mask_neighbors_and_independent_frames(
    monkeypatch,
):
    clock = TicketClock("stereo")
    ticket = clock.issue("left", "fixed", "pair")
    images = sources()
    left = frame([31, 23], "left-camera")
    right = frame([31, 23], "right-camera")
    payload = {
        "ticket": ticket,
        "spatial": left,
        "recipe_ref": {"id": "part", "revision": 2, "digest": "a" * 64},
        "custom": {"x": [1]},
        "pyramid": {
            "level": 1,
            "largest_size": [62, 46],
            "to_largest": [[2, 0, 0.5], [0, 2, 0.5], [0, 0, 1]],
        },
    }
    inputs = {
        "left": ValuePacket(images["left"], payload),
        "right": ValuePacket(
            images["right"], {"ticket": {**ticket, "source": "right"}, "spatial": right}
        ),
    }
    mask = np.full((23, 31), 255, np.uint8)
    mask[:, 8] = 0
    inputs["right_mask"] = ValuePacket(mask, {"spatial": right})
    inputs["left_mask"] = ValuePacket(np.full((23, 31), 255, np.uint8), {"spatial": left})
    inputs["region"] = {
        "model": "image_region_v1",
        "shape": "rectangle",
        "image_size": [31, 23],
        "rectangle": [4, 0, 24, 23],
        "coordinate_context": left,
    }
    block = tool()
    monkeypatch.setattr(
        block,
        "infer_tensors",
        lambda *a, **k: ({"d": np.full((1, 23, 31), 1.5, np.float32)}, "fixture"),
    )
    try:
        result = block.execute(inputs)
        assert result is not None, block.error
        expected = np.zeros((23, 31), bool)
        expected[:, 4:28] = True
        expected[:, 9:11] = False
        np.testing.assert_array_equal(result.outputs["valid_mask"] != 0, expected)
        p = result.payloads["disparity"]
        for key in payload:
            assert p[key] == payload[key]
        assert p["stereo_pair"]["views"]["right"]["spatial"] == right
        assert p["stereo_pair"]["tickets_verified"]
        inputs["right_mask"] = ValuePacket(mask, {"spatial": left})
        assert block.execute(inputs) is None and "coordinate frame" in block.error
    finally:
        block.close()


@pytest.mark.parametrize(
    "policy,other_session,other_value,missing,accepted",
    [
        ("same_ticket", False, False, False, True),
        ("same_ticket", True, False, False, False),
        ("same_ticket", False, True, False, False),
        ("same_value", True, False, False, True),
        ("same_value", True, True, False, False),
        ("same_ticket", False, False, True, False),
        ("unchecked", True, True, False, True),
    ],
)
def test_ticket_pair_policy_does_not_conflate_capture_identity(
    policy, other_session, other_value, missing, accepted
):
    a = TicketClock("a").issue("left")
    b = TicketClock("b" if other_session else "a").issue("right")
    if other_value:
        b["value"] = 2
    inputs = sources()
    inputs["left"] = ValuePacket(inputs["left"], {"ticket": a})
    if not missing:
        inputs["right"] = ValuePacket(inputs["right"], {"ticket": b})
    block = tool(ticket_policy=policy)
    try:
        result = block.execute(inputs)
        assert (result is not None) == accepted, block.error
        if accepted:
            assert result.outputs["info"]["pair"]["tickets_verified"] == (policy != "unchecked")
    finally:
        block.close()


def test_raw_invalid_sentinel_spreads_only_through_required_interpolation_support(monkeypatch):
    block = tool(invalid_value_enabled=True, invalid_head_value=-99.0, check_correspondence=False)
    head = np.full((1, 3, 4), 3, np.float32)
    head[0, 1, 1] = -99
    monkeypatch.setattr(block, "infer_tensors", lambda *a, **k: ({"d": head}, "fixture"))
    try:
        result = block.execute(sources())
        assert result is not None, block.error
        mask = result.outputs["valid_mask"] != 0
        assert mask[0, 0] and mask[-1, -1] and not mask[11, 11]
        assert np.isnan(result.outputs["disparity"][~mask]).all()
    finally:
        block.close()


def test_physical_reprojection_uses_original_disparity_and_known_camera_rays():
    block = tool()
    cloud = ReprojectDisparity(parameters={"sample_step": 1})
    rec = {
        "model": "stereo_rectification_v1",
        "image_size": [31, 23],
        "units": "m",
        "frame": "left_rectified",
        "Q": [[1, 0, 0, -15], [0, 1, 0, -11], [0, 0, 0, 80], [0, 0, 10, 0]],
    }
    try:
        result = block.execute({**sources(), "rectification": rec})
        assert result is not None, block.error
        assert result.outputs["rectification"] == rec
        output = cloud.execute({"disparity": result.outputs["disparity"], "rectification": rec})
        assert output is not None, cloud.error
        valid = result.outputs["valid_mask"] != 0
        y, x = np.nonzero(valid)
        expected = np.column_stack(((x - 15) * 0.1 / 3, (y - 11) * 0.1 / 3, np.full(len(x), 8 / 3)))
        np.testing.assert_allclose(output.outputs["cloud"].points, expected, atol=8e-6)
        assert output.outputs["cloud"].units == "m"
    finally:
        block.close()
        cloud.close()


@pytest.mark.parametrize("mode", ["named", "concatenated"])
@pytest.mark.parametrize("layout", ["NCHW", "NHWC"])
def test_input_bindings_packing_and_per_view_normalization(mode, layout, monkeypatch):
    block = tool(input_mode=mode, input_layout=layout, mean=[0.1, 0.2, 0.3], std=[2.0, 3.0, 4.0])
    images = sources()
    expected = [
        (
            np.asarray(images[k])[..., ::-1].astype(np.float32) / 255
            - np.array([0.1, 0.2, 0.3], np.float32)
        )
        / np.array([2.0, 3.0, 4.0], np.float32)
        for k in ("left", "right")
    ]

    def infer(arrays, **kwargs):
        prepared = [np.concatenate(expected, axis=2)] if mode == "concatenated" else expected
        assert [name for name, _ in arrays] == (
            ["left"] if mode == "concatenated" else ["left", "right"]
        )
        for (_, array), image in zip(arrays, prepared):
            target = image.transpose(2, 0, 1)[None] if layout == "NCHW" else image[None]
            np.testing.assert_allclose(array, target, atol=1e-7)
            assert array.dtype == np.float32 and array.flags.c_contiguous
        return {"d": np.full((1, 23, 31), 3.0, np.float32)}, "fixture"

    monkeypatch.setattr(block, "infer_tensors", infer)
    try:
        assert block.execute(images) is not None, block.error
    finally:
        block.close()


def test_empty_selection_no_model_io_and_lean_or_display_never_changes_numbers(monkeypatch):
    block = tool(model_path="missing.onnx")
    try:
        result = block.execute({**sources(), "left_mask": np.zeros((23, 31), np.uint8)})
        assert (
            result is not None and not result.passed and result.outputs["info"]["native_calls"] == 0
        )
        assert np.isnan(result.outputs["disparity"]).all()
        block.parameters["model_path"] = str(BASE / "models/plane/model.onnx")
        normal = block.execute(sources()).outputs["disparity"].copy()
        block.run_profile = PROFILES["lean"]
        result = block.execute(sources())
        np.testing.assert_array_equal(result.outputs["disparity"], normal)
        assert result.outputs["preview"] is NO_VALUE and not result.graphics
        block.run_profile = PROFILES["debug"]
        block.parameters["preview_mode"] = "fixed"
        block.parameters["display_max"] = 1000.0
        np.testing.assert_array_equal(block.execute(sources()).outputs["disparity"], normal)
    finally:
        block.close()


@pytest.mark.parametrize(
    "parameters",
    [
        {"head_scale": 0},
        {"left_name": "right"},
        {"output_names": ["a", "b"]},
        {"mean": [0]},
        {"std": [1, 0, 1]},
        {"width": True},
        {"minimum_disparity": 10, "maximum_disparity": 2},
        {"head_sign": "guessed"},
        {"display_min": 4, "display_max": 3},
        {"fusion": 1},
        {"maximum_input_elements": 1},
        {"maximum_source_pixels": 1},
        {"max_output_elements": 1},
        {"engine": "new", "fusion": False},
    ],
)
def test_parameter_and_resource_failures_emit_no_result_and_allow_recovery(parameters):
    block = tool(**parameters)
    try:
        assert block.execute(sources()) is None and block.error
        assert not block.current_emissions
        good = tool()
        block.parameters = copy.deepcopy(good.parameters)
        good.close()
        assert block.execute(sources()) is not None, block.error
    finally:
        block.close()


@pytest.mark.parametrize("bad", ["size", "float", "nan", "depth", "rectification"])
def test_invalid_images_or_rectification_are_contained(bad):
    inputs = sources()
    if bad == "size":
        inputs["right"] = np.zeros((22, 31, 3), np.uint8)
    if bad == "float":
        inputs["left"] = np.full((23, 31, 3), 2.0, np.float32)
    if bad == "nan":
        inputs["right"] = np.full((23, 31, 3), np.nan, np.float32)
    if bad == "depth":
        inputs["left"] = DepthArray(np.ones((23, 31)), "m")
    if bad == "rectification":
        inputs["rectification"] = {
            "model": "stereo_rectification_v1",
            "image_size": [31, 23],
            "Q": np.zeros((4, 4)).tolist(),
        }
    block = tool()
    try:
        assert block.execute(inputs) is None and block.error
    finally:
        block.close()
