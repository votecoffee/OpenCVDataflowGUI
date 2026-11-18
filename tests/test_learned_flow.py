"""Independent vector quantities, pairing, consumer contracts and observation budgets."""

import copy
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.flow_files import LoadFlowFile, SaveFlowFile
from opencv_dataflow.flow_warp import WarpDenseFlow
from opencv_dataflow.images import DepthArray, ImageArray
from opencv_dataflow.learned_flow import LearnedFlow, endpoint_support
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.run_profiles import PROFILES
from opencv_dataflow.tickets import TicketClock

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/learned_flow"


def tool(**parameters):
    return LearnedFlow(
        parameters={
            "model_path": str(BASE / "models/plane/model.onnx"),
            "width": 31,
            "height": 23,
            **parameters,
        }
    )


def sources(encoding="bgr", dtype="uint8", other="second"):
    images = []
    for name in ("first", other):
        image = cv2.imread(str(BASE / "assets" / (name + ".png")))
        if encoding == "gray":
            image = image[..., 2]
        elif encoding in ("rgb", "rgba"):
            image = image[..., ::-1]
        if encoding in ("rgba", "bgra"):
            image = np.dstack((image, np.full(image.shape[:2], 123, np.uint8)))
        if dtype == "uint16":
            image = image.astype(np.uint16) * 257
        elif dtype == "float32":
            image = image.astype(np.float32) / 255
        images.append(ImageArray(image, encoding))
    return dict(zip(("first", "second"), images))


@pytest.mark.parametrize("encoding", ["bgr", "rgb", "bgra", "rgba", "gray"])
@pytest.mark.parametrize("dtype", ["uint8", "uint16", "float32"])
def test_native_affine_plane_quantities_preserve_original_images(encoding, dtype):
    inputs = sources(encoding, dtype)
    before = copy.deepcopy(inputs)
    block = tool()
    try:
        result = block.execute(inputs)
        assert result is not None and result.passed, block.error
        valid = result.outputs["valid_mask"] != 0
        vectors = result.outputs["flow"].vectors
        expected = [0.9, 1.2] if encoding.startswith("gray") else [2.0, -1.0]
        np.testing.assert_allclose(
            vectors[valid], np.broadcast_to(expected, (int(valid.sum()), 2)), atol=1e-5
        )
        assert np.isnan(vectors[~valid]).all() and valid[2:-2, 2:-3].all()
        np.testing.assert_array_equal(result.outputs["image"], inputs["first"])
        assert result.outputs["image"].color_format == inputs["first"].color_format
        for name in inputs:
            np.testing.assert_array_equal(inputs[name], before[name])
        assert result.outputs["info"]["direction"] == "first_to_second"
        assert "first-to-second" in result.outputs["flow"].describe()["coordinates"]
        result = block.execute(sources(encoding, dtype, other="subpixel"))
        assert result is not None, block.error
        valid = result.outputs["valid_mask"] != 0
        expected = [1.2, 1.6] if encoding.startswith("gray") else [1.5, 1.0]
        np.testing.assert_allclose(
            result.outputs["flow"].vectors[valid],
            np.broadcast_to(expected, (int(valid.sum()), 2)),
            atol=1e-5,
        )
    finally:
        block.close()


@pytest.mark.parametrize("layout", ["NCHW", "NHWC", "CHW", "HWC"])
@pytest.mark.parametrize("units", ["network_pixels", "head_pixels", "source_pixels"])
@pytest.mark.parametrize("order", ["xy", "yx"])
def test_vector_layout_order_affine_and_anisotropic_source_scaling(
    monkeypatch, layout, units, order
):
    raw = np.broadcast_to([1.0, -2.0], (6, 8, 2)).astype(np.float32).copy()
    encoded = {
        "NCHW": raw.transpose(2, 0, 1)[None],
        "NHWC": raw[None],
        "CHW": raw.transpose(2, 0, 1),
        "HWC": raw,
    }[layout]
    block = tool(
        width=16,
        height=12,
        head_units=units,
        layout=layout,
        component_order=order,
        head_scale=[2.0, 0.5],
        head_offset=[0.5, 1.0],
        check_correspondence=False,
    )
    monkeypatch.setattr(block, "infer_tensors", lambda *a, **k: ({"flow": encoded}, "fixture"))
    factor = np.array(
        {
            "network_pixels": [31 / 16, 23 / 12],
            "head_pixels": [31 / 8, 23 / 6],
            "source_pixels": [1, 1],
        }[units]
    )
    decoded = np.array([1.0, -2.0] if order == "xy" else [-2.0, 1.0])
    expected = (decoded * [2, 0.5] + [0.5, 1]) * factor
    try:
        result = block.execute(sources())
        assert result is not None, block.error
        assert result.outputs["valid_mask"].all()
        np.testing.assert_allclose(
            result.outputs["flow"].vectors, np.broadcast_to(expected, (23, 31, 2)), atol=2e-6
        )
    finally:
        block.close()


def test_fractional_endpoint_mask_and_distinct_tickets_frames_pyramids(monkeypatch):
    images = sources()
    a, b = frame([31, 23], "first"), frame([31, 23], "second")
    clock = TicketClock("video")
    first_ticket, second_ticket = clock.issue("camera"), clock.issue("camera")
    context = {
        "spatial": a,
        "ticket": first_ticket,
        "aux": {"lot": ["A"]},
        "recipe_ref": {"id": "part", "revision": 2, "digest": "a" * 64},
        "pyramid": {
            "level": 1,
            "largest_size": [62, 46],
            "to_largest": [[2, 0, 0.5], [0, 2, 0.5], [0, 0, 1]],
        },
    }
    mask = np.full((23, 31), 255, np.uint8)
    mask[8, 8] = 0
    inputs = {
        "first": ValuePacket(images["first"], context),
        "second": ValuePacket(images["second"], {"spatial": b, "ticket": second_ticket}),
        "second_mask": ValuePacket(mask, {"spatial": b}),
    }
    block = tool()
    head = np.broadcast_to([1.5, 0.5], (23, 31, 2)).astype(np.float32)[None]
    block.parameters["layout"] = "NHWC"
    monkeypatch.setattr(block, "infer_tensors", lambda *a, **k: ({"flow": head}, "fixture"))
    try:
        result = block.execute(inputs)
        assert result is not None, block.error
        expected = np.ones((23, 31), bool)
        expected[:, -2:] = False
        expected[-1, :] = False
        expected[7:9, 6:8] = False
        np.testing.assert_array_equal(result.outputs["valid_mask"] != 0, expected)
        payload = result.payloads["flow"]
        for key, value in context.items():
            assert payload[key] == value
        assert payload["flow_pair"]["frames"]["second"]["ticket"] == second_ticket
        assert payload["flow_pair"]["frames"]["second"]["spatial"] == b
        assert (
            payload["flow_association"]["source"]["image"]["sha256"]
            != payload["flow_association"]["target"]["image"]["sha256"]
        )
        inputs["second_mask"] = ValuePacket(mask, {"spatial": a})
        assert block.execute(inputs) is None and "coordinate frame" in block.error
        inputs.pop("second_mask")
        block.parameters["alignment"] = "same_frame"
        assert block.execute(inputs) is None and "coordinate frame" in block.error
    finally:
        block.close()


def test_tiny_endpoint_contributors_are_never_accepted_via_mask_tolerance():
    mask = np.ones((3, 3), np.uint8)
    mask[1, 2] = 0
    assert not endpoint_support(mask, np.array([[1 + 1e-12]]), np.array([[1.0]])).item()
    assert endpoint_support(mask, np.array([[1.0]]), np.array([[1.0]])).item()


@pytest.mark.parametrize("mode", ["named", "concatenated"])
@pytest.mark.parametrize("layout", ["NCHW", "NHWC"])
def test_input_binding_layout_and_normalization_are_per_frame(mode, layout, monkeypatch):
    images = sources()
    block = tool(input_mode=mode, input_layout=layout, mean=[0.1, 0.2, 0.3], std=[2.0, 3.0, 4.0])
    expected = [
        (
            np.asarray(images[k])[..., ::-1].astype(np.float32) / 255
            - np.array([0.1, 0.2, 0.3], np.float32)
        )
        / np.array([2.0, 3.0, 4.0], np.float32)
        for k in ("first", "second")
    ]

    def infer(arrays, **kwargs):
        prepared = [np.concatenate(expected, axis=2)] if mode == "concatenated" else expected
        assert [k for k, _ in arrays] == (
            ["first"] if mode == "concatenated" else ["first", "second"]
        )
        for (_, array), image in zip(arrays, prepared):
            target = image.transpose(2, 0, 1)[None] if layout == "NCHW" else image[None]
            np.testing.assert_allclose(array, target, atol=1e-7)
            assert array.dtype == np.float32 and array.flags.c_contiguous
        return {"flow": np.zeros((1, 2, 23, 31), np.float32)}, "fixture"

    monkeypatch.setattr(block, "infer_tensors", infer)
    try:
        assert block.execute(images) is not None, block.error
    finally:
        block.close()


def test_reverse_means_swapping_frames_and_unbound_field_still_describes_direction():
    images = sources()
    block = tool(bind_image_content=False)
    try:
        result = block.execute({"first": images["second"], "second": images["first"]})
        assert result is not None, block.error
        valid = result.outputs["valid_mask"] != 0
        np.testing.assert_allclose(
            result.outputs["flow"].vectors[valid],
            np.broadcast_to([-2, 1], (int(valid.sum()), 2)),
            atol=1e-5,
        )
        assert "flow_association" not in result.payloads["flow"]
        assert "first-to-second" in result.outputs["flow"].describe()["coordinates"]
        assert result.drawings["learned motion"].source == "first"
    finally:
        block.close()


def test_sentinel_and_motion_limits_withhold_support_without_clipping(monkeypatch):
    block = tool(invalid_value_enabled=True, invalid_head_value=0.0)
    head = np.zeros((1, 2, 23, 31), np.float32)
    head[:, 0] = 5
    monkeypatch.setattr(block, "infer_tensors", lambda *a, **k: ({"flow": head}, "fixture"))
    try:
        result = block.execute(sources())
        assert result is not None and not result.passed
        assert np.isnan(result.outputs["flow"].vectors).all()
        block.parameters.update(invalid_value_enabled=False, maximum_motion=1.0)
        result = block.execute(sources())
        assert result is not None and not result.passed
        assert np.isnan(result.outputs["flow"].vectors).all()
    finally:
        block.close()


def test_existing_forward_warp_and_content_bound_file_roundtrip(tmp_path):
    block = tool()
    warp = WarpDenseFlow(parameters={"direction": "source_to_target", "consistency": "off"})
    save = SaveFlowFile(
        parameters={
            "path": str(tmp_path / "motion.flo"),
            "path_base": "working_directory",
            "metadata_mode": "require_images",
            "invalid_vectors": "preserve",
        }
    )
    load = LoadFlowFile(
        parameters={
            "path": str(tmp_path / "motion.flo"),
            "path_base": "working_directory",
            "metadata_mode": "verify_images",
            "expected_direction": "source_to_target",
        }
    )
    images = sources()
    try:
        result = block.execute(images)
        assert result is not None, block.error
        packet = ValuePacket(result.outputs["flow"], result.payloads["flow"])
        inputs = {
            "image": images["first"],
            "target": images["second"],
            "flow": packet,
            "flow_support": result.outputs["valid_mask"],
        }
        output = warp.execute(inputs)
        assert output is not None and output.passed, warp.error
        valid = output.outputs["coverage"] != 0
        np.testing.assert_array_equal(output.outputs["image"][valid], images["second"][valid])
        warp.parameters["direction"] = "target_to_source"
        assert warp.execute(inputs) is None and "direction" in warp.error
        warp.parameters["direction"] = "source_to_target"
        wrong = images["second"].copy()
        wrong[10, 10] = 0
        assert (
            warp.execute({**inputs, "target": wrong}) is None
            and "target image content" in warp.error
        )
        output = save.execute(
            {"flow": packet, "source": images["first"], "target": images["second"]}
        )
        assert output is not None, save.error
        output = load.execute({"source": images["first"], "target": images["second"]})
        assert output is not None, load.error
        np.testing.assert_array_equal(
            output.outputs["flow"].vectors, result.outputs["flow"].vectors
        )
    finally:
        for item in (block, warp, save, load):
            item.close()


def test_empty_selection_zero_vectors_lean_and_explicit_numeric_ports(monkeypatch):
    block = tool(model_path="missing.onnx")
    try:
        result = block.execute({**sources(), "first_mask": np.zeros((23, 31), np.uint8)})
        assert (
            result is not None and not result.passed and result.outputs["info"]["native_calls"] == 0
        )
        assert np.isnan(result.outputs["flow"].vectors).all()
        block.parameters["model_path"] = str(BASE / "models/plane/model.onnx")
        result = block.execute(sources())
        numeric = result.outputs["flow"].vectors.copy()
        assert result.drawings and result.outputs["preview"] is not NO_VALUE
        block.run_profile = PROFILES["lean"]
        result = block.execute(sources())
        np.testing.assert_array_equal(result.outputs["flow"].vectors, numeric)
        assert result.outputs["preview"] is NO_VALUE and result.outputs["magnitude"] is NO_VALUE
        assert not result.drawings and not result.graphics
        block.exposed_outputs.extend(["magnitude", "angle"])
        result = block.execute(sources())
        assert result.outputs["magnitude"] is not NO_VALUE
        valid = result.outputs["valid_mask"] != 0
        np.testing.assert_allclose(result.outputs["magnitude"][valid], np.sqrt(5), atol=1e-5)
        np.testing.assert_allclose(
            result.outputs["angle"][valid], np.degrees(np.arctan2(-1, 2)) % 360, atol=0.02
        )
        monkeypatch.setattr(
            block,
            "infer_tensors",
            lambda *a, **k: ({"flow": np.zeros((1, 2, 23, 31), np.float32)}, "fixture"),
        )
        result = block.execute(sources())
        assert result is not None and result.outputs["valid_mask"].all()
        assert not result.outputs["flow"].vectors.any()
    finally:
        block.close()


@pytest.mark.parametrize(
    "parameters",
    [
        {"head_scale": [0, 1]},
        {"head_offset": [1]},
        {"std": [1, 0, 1]},
        {"mean": [0]},
        {"first_name": "second"},
        {"output_names": ["a", "b"]},
        {"layout": "guessed"},
        {"width": True},
        {"bind_image_content": 1},
        {"visual_max_motion": 0},
        {"maximum_source_pixels": 1},
        {"maximum_input_elements": 1},
        {"max_output_elements": 1},
        {"engine": "new", "fusion": False},
    ],
)
def test_invalid_configuration_and_resource_failures_are_contained_and_recoverable(parameters):
    block = tool(**parameters)
    try:
        assert block.execute(sources()) is None and block.error and not block.current_emissions
        good = tool()
        block.parameters = copy.deepcopy(good.parameters)
        good.close()
        assert block.execute(sources()) is not None, block.error
    finally:
        block.close()


@pytest.mark.parametrize(
    "head",
    [
        np.ones((2, 2, 23, 31)),
        np.ones((1, 3, 23, 31)),
        np.ones((2, 23, 31)),
        np.full((1, 2, 23, 31), np.nan),
    ],
)
def test_wrong_head_rank_batch_channel_or_nonfinite_values_clear_native_state(head, monkeypatch):
    block = tool()
    monkeypatch.setattr(block, "infer_tensors", lambda *a, **k: ({"flow": head}, "fixture"))
    try:
        assert block.execute(sources()) is None and block.error
        assert block._net is None and block._snapshot is None
    finally:
        block.close()


@pytest.mark.parametrize("bad", ["size", "range", "nan", "depth"])
def test_invalid_image_pair_is_contained(bad):
    inputs = sources()
    if bad == "size":
        inputs["second"] = np.zeros((22, 31, 3), np.uint8)
    if bad == "range":
        inputs["first"] = np.full((23, 31, 3), 2.0, np.float32)
    if bad == "nan":
        inputs["second"] = np.full((23, 31, 3), np.nan, np.float32)
    if bad == "depth":
        inputs["first"] = DepthArray(np.ones((23, 31)), "m")
    block = tool()
    try:
        assert block.execute(inputs) is None and block.error
    finally:
        block.close()
