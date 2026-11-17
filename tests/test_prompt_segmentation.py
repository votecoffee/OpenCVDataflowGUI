"""Independent prompt tensors, mask mapping, context binding and bounded failures."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame, packet_for
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.prompt_mask_decode import DecodePromptMasks
from opencv_dataflow.prompt_segmentation import (
    EfficientSAMInference,
    SegmentationPrompts,
    native_prompts,
)
from opencv_dataflow.raster_regions import raster_mask
from opencv_dataflow.run_profiles import PROFILES
from opencv_dataflow.tickets import TicketClock


def source():
    pixels = np.zeros((20, 40, 3), np.uint8)
    pixels[..., 2] = np.arange(40)
    context = {
        "spatial": frame([40, 20], "camera"),
        "ticket": TicketClock().issue("camera"),
        "custom": {"lot": "test"},
    }
    return ValuePacket(pixels, context)


def planes():
    values = np.full((1, 1, 3, 1024, 1024), -2, np.float32)
    values[0, 0, 0, :, :512] = 2
    values[0, 0, 1, :, 512:] = 2
    values[0, 0, 2] = 2
    return values


def prepared(monkeypatch, *, native=None):
    original = source()
    builder = SegmentationPrompts(
        parameters={"points": [[30, 10], [3, 10]], "point_labels": [1, 0]}
    )
    definition = builder.execute({"image": original})
    assert definition is not None, builder.error
    inference = EfficientSAMInference()
    values = planes() if native is None else native

    def forward(arrays, *args, **kwargs):
        assert kwargs["array_outputs"] is True
        names = [name for name, _ in arrays]
        assert names == ["batched_images", "batched_point_coords", "batched_point_labels"]
        inference._model_identity = {"bytes": 1, "sha256": "0" * 64}
        return {
            "output_masks": values,
            "iou_predictions": np.float32([[[0.9, 0.8, 0.7]]]),
        }, "fixture.onnx"

    monkeypatch.setattr(inference, "infer_tensors", forward)
    result = inference.execute({"image": original, "prompts": packet_for(definition, "prompts")})
    assert result is not None, inference.error
    args = {
        "image": original,
        **{n: packet_for(result, n) for n in ("candidates", "qualities", "prompts", "info")},
    }
    return args, result


def decode(args, **parameters):
    tool = DecodePromptMasks(parameters=parameters)
    result = tool.execute(args)
    assert result is not None, tool.error
    return tool, result


def test_point_and_box_coordinates_labels_padding_are_explicit_and_native_shaped():
    builder = SegmentationPrompts(
        parameters={"points": [[10, 5], [30, 10]], "point_labels": [1, 0], "box": [5, 2, 20, 15]}
    )
    result = builder.execute({"image": source()})
    assert result is not None, builder.error
    coords, labels = native_prompts(result.outputs["prompts"])
    assert coords.dtype == labels.dtype == np.float32
    np.testing.assert_array_equal(labels, [[[1, 0, 2, 3, -1, -1]]])
    np.testing.assert_allclose(
        coords[0, 0, :4], np.array([[10, 5], [30, 10], [5, 2], [25, 17]]) * [1024 / 40, 1024 / 20]
    )
    np.testing.assert_array_equal(coords[0, 0, 4:], 0)
    assert result.drawings


@pytest.mark.parametrize(
    "parameters",
    [
        {"points": [[1, 2]], "point_labels": []},
        {"points": [[1, 2]], "point_labels": [True]},
        {"points": [[1, 2], [1, 2]], "point_labels": [1, 0]},
        {"points": [[1, float("nan")]], "point_labels": [1]},
        {"points": [[1, 1]] * 5, "point_labels": [1] * 5, "box": [0, 0, 5, 5]},
        {"box": [0, 0, -1, 2]},
    ],
)
def test_invalid_prompt_fields_reject_without_native_work(parameters):
    with pytest.raises(ValidationError):
        SegmentationPrompts(parameters=parameters).validate_parameters()


@pytest.mark.parametrize(
    "parameters",
    [
        {},
        {"points": [[40, 10]], "point_labels": [1]},
        {"points": [[1, 2]], "point_labels": [0]},
        {"box": [30, 10, 20, 20]},
    ],
)
def test_missing_positive_or_outside_source_prompts_emit_no_event(parameters):
    tool = SegmentationPrompts(parameters=parameters)
    assert tool.execute({"image": source()}) is None and not tool.current_emissions


def test_required_membership_prefers_consistent_mask_and_preserves_source_and_exact_region(
    monkeypatch,
):
    args, inference = prepared(monkeypatch)
    _, result = decode(args, emit_logits=True)
    assert result.passed and result.outputs["quality"] == pytest.approx(0.8)
    expected = np.zeros((20, 40), np.uint8)
    expected[:, 20:] = 255
    np.testing.assert_array_equal(result.outputs["mask"], expected)
    np.testing.assert_array_equal(
        raster_mask(result.outputs["region"], (20, 40), [100000]), expected
    )
    np.testing.assert_array_equal(result.outputs["image"], args["image"].value)
    assert result.outputs["logits"].dtype == np.float32 and result.drawings
    assert result.payloads["mask"]["ticket"] == args["image"].payload["ticket"]
    assert result.payloads["mask"]["custom"] == {"lot": "test"}
    assert inference.outputs["candidates"].items[0].payload["spatial"]["image_size"] == [1024, 1024]


@pytest.mark.parametrize(
    "policy,passed,index",
    [("report_only", True, 0), ("prefer_consistent", True, 1), ("require_consistent", True, 1)],
)
def test_membership_policy_and_native_quality_order_are_separate(
    policy, passed, index, monkeypatch
):
    args, _ = prepared(monkeypatch)
    _, result = decode(args, prompt_policy=policy)
    assert result.passed is passed and result.outputs["report"]["selected_index"] == index


def test_quality_area_and_membership_failures_retain_current_geometry(monkeypatch):
    args, _ = prepared(monkeypatch)
    _, result = decode(
        args, selection="index", candidate_index=0, minimum_quality=1, minimum_pixels=600
    )
    assert not result.passed and result.outputs["pixel_count"] == 400
    assert {r["code"] for r in result.fail_reasons} == {
        "prompt_segmentation.quality",
        "prompt_segmentation.prompts",
        "prompt_segmentation.area",
    }


def test_output_selection_does_not_crop_inference_or_change_raw_prompt_membership(monkeypatch):
    args, _ = prepared(monkeypatch)
    args["mask"] = ValuePacket(np.zeros((20, 40), np.uint8), args["image"].payload)
    _, result = decode(args)
    assert result.outputs["pixel_count"] == 0 and not result.passed
    assert result.outputs["report"]["candidates"][1]["prompts_consistent"]
    assert result.outputs["report"]["box_is_output_clip"] is False


@pytest.mark.parametrize(
    "damage",
    [
        "image",
        "prompt",
        "quality_token",
        "child_token",
        "ticket",
        "frame",
        "quality_nan",
        "logit_nan",
        "labels",
    ],
)
def test_mixed_or_changed_inputs_reject_instead_of_reusing_a_mask(damage, monkeypatch):
    args, _ = prepared(monkeypatch)
    if damage == "image":
        args["image"].value[0, 0] = 42
    if damage == "prompt":
        args["prompts"].value["points"][0][0] = 31
    if damage == "quality_token":
        args["qualities"].value["inference_revision"] = "other"
    if damage == "child_token":
        args["candidates"].value.items[0].payload["prompt_inference"] = "other"
    if damage == "ticket":
        args["qualities"].payload["ticket"] = TicketClock().issue("other")
    if damage == "frame":
        args["candidates"].value.items[0].payload["spatial"] = frame([1024, 1024], "other")
    if damage == "quality_nan":
        args["qualities"].value["values"][0] = float("nan")
    if damage == "logit_nan":
        args["candidates"].value.items[0].value[0, 0] = float("nan")
    if damage == "labels":
        args["candidates"].value.labels = ("2", "1", "0")
    tool = DecodePromptMasks()
    assert tool.execute(args) is None and not tool.current_emissions


def test_direct_rgb_uint8_preprocessing_matches_native_order_and_alpha_depth_reject(monkeypatch):
    original = source()
    original.value = ImageArray(original.value[..., ::-1], "rgb")
    builder = SegmentationPrompts(parameters={"box": [1, 1, 30, 15]})
    prompt = builder.execute({"image": original})
    tool = EfficientSAMInference()
    seen = []

    def native(arrays, *a, **k):
        seen.append(arrays[0][1].copy())
        tool._model_identity = {"bytes": 1, "sha256": "0" * 64}
        return {
            "output_masks": planes(),
            "iou_predictions": np.float32([[[0.9, 0.8, 0.7]]]),
        }, "fixture"

    monkeypatch.setattr(tool, "infer_tensors", native)
    assert tool.execute({"image": original, "prompts": packet_for(prompt, "prompts")}) is not None
    rgb = cv2.resize(np.asarray(original.value), (1024, 1024)).astype(np.float32) / 255
    np.testing.assert_array_equal(seen[0], rgb.transpose(2, 0, 1)[None])
    original.value = original.value.astype(np.uint16)
    assert tool.execute({"image": original, "prompts": packet_for(prompt, "prompts")}) is None


def test_optional_outputs_lean_budgets_native_failure_recovery_and_input_immutability(monkeypatch):
    args, _ = prepared(monkeypatch)
    before = args["candidates"].value.items[1].value.copy()
    tool = DecodePromptMasks(parameters={"emit_region": False})
    tool.run_profile = PROFILES["lean"]
    result = tool.execute(args)
    assert result.outputs["region"] is NO_VALUE and result.outputs["logits"] is NO_VALUE
    assert not result.drawings and not tool.current_emissions.get("region")
    assert DecodePromptMasks(parameters={"maximum_image_pixels": 1}).execute(args) is None
    assert DecodePromptMasks(parameters={"maximum_region_runs": 1}).execute(args) is None
    resize = cv2.resize
    monkeypatch.setattr(cv2, "resize", lambda *a, **k: (_ for _ in ()).throw(cv2.error("native")))
    assert tool.execute(args) is None and not tool.current_emissions
    monkeypatch.setattr(cv2, "resize", resize)
    assert tool.execute(args).passed
    np.testing.assert_array_equal(args["candidates"].value.items[1].value, before)


def test_registry_xml_and_budget_validation(tmp_path):
    registry = builtin_registry()
    graph = Graph()
    for kind in ("ml.segmentation_prompts", "ml.efficientsam", "ml.decode_prompt_masks"):
        graph.add(registry.create(kind))
    export_project_xml(graph.to_dict(), tmp_path / "program.xml")
    assert read_project_xml(tmp_path / "program.xml") == graph.to_dict()
    graph.close()
    with pytest.raises(ValidationError):
        EfficientSAMInference(parameters={"max_output_elements": 1}).validate_parameters()
    with pytest.raises(ValidationError):
        DecodePromptMasks(parameters={"emit_region": 1}).validate_parameters()


def test_offscreen_parameter_choices_and_documentation_do_not_run_a_model(app, monkeypatch):
    from PySide6.QtWidgets import QComboBox

    from opencv_dataflow.app import MainWindow

    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda *a, **k: pytest.fail("native import"))
    window = MainWindow()
    graph = Graph()
    native = graph.add(EfficientSAMInference())
    decoder = graph.add(DecodePromptMasks())
    try:
        window.replace_graph(graph)
        for block, names in (
            (native, ("engine", "target", "receipt_mode")),
            (decoder, ("selection", "prompt_policy", "resampling")),
        ):
            window.inspector.select(block)
            assert block.description and block.documentation
            for name in names:
                assert isinstance(window.inspector.fields[name], QComboBox)
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
        graph.close()
