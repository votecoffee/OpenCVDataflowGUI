"""Named binding, native numerical references, ticket roles and resource lifecycle."""

import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.dnn_blocks import DNNInference
from opencv_dataflow.ml_common import selected_tensor, tensor_record
from opencv_dataflow.multi_inference import MakeTensor, MultiInputInference, PrepareImageTensor
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.tickets import TicketClock, ticket_key
from opencv_dataflow.toolbox import block_entry

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/multi_input_inference"
TRUTH = json.loads((BASE / "truth.json").read_bytes())


def model(**parameters):
    return MultiInputInference(
        parameters={
            "model_path": str(BASE / "assets/pair.onnx"),
            "input_1_shape": [1, 3, 8, 8],
            "input_2_shape": [1, 1, 8, 8],
            **parameters,
        }
    )


def tensors(left=0.8, right=0.2):
    return {
        "input_1": tensor_record(np.full((1, 3, 8, 8), left, np.float32), 1000),
        "input_2": tensor_record(np.full((1, 1, 8, 8), right, np.float32), 1000),
    }


@pytest.mark.parametrize("engine", ["auto", "classic", "new"])
def test_native_complete_rebinding_named_outputs_cache_and_close(engine):
    block = model(engine=engine, output_names=["scores"])
    try:
        first = block.execute(tensors())
        assert first is not None, block.error
        np.testing.assert_allclose(
            selected_tensor(first.outputs["tensors"]), [[0.6] * 3], atol=2e-6
        )
        native = block._net
        second = block.execute(tensors(0.4, 0.3))
        assert block._net is native
        np.testing.assert_allclose(
            selected_tensor(second.outputs["tensors"]), [[0.1] * 3], atol=2e-6
        )
        block.parameters["output_names"] = ["scores", "left_mean"]
        third = block.execute(tensors(0.7, 0.1))
        assert list(third.outputs["tensors"]) == ["scores", "left_mean"]
        np.testing.assert_allclose(
            selected_tensor(third.outputs["tensors"], "left_mean"), [[0.7] * 3], atol=2e-6
        )
        assert not third.outputs["input_info"]["tickets_verified"]
        assert third.outputs["preprocessing"] is NO_VALUE
    finally:
        block.close()
    assert block._net is None


@pytest.mark.parametrize("case", list(TRUTH["cases"]))
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_native_programs_against_independent_arithmetic_and_metadata(case, profile):
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    graph.set_run_profile(profile)
    try:
        graph.run()
        block = graph.blocks["multi"]
        expected = TRUTH["cases"][case]
        if "error" in expected:
            assert block.last_result is None and expected["error"] in block.error
            assert block._net is None and not any(block.current_emissions.values())
            return
        result = block.last_result
        assert result is not None and result.passed, block.error
        np.testing.assert_allclose(
            selected_tensor(result.outputs["tensors"], "scores"), [expected["scores"]], atol=2e-6
        )
        assert graph.blocks["classify"].last_result.outputs["label"] == expected["label"]
        assert graph.blocks["classify"].last_result.passed
        info = result.outputs["input_info"]
        assert info["tickets_verified"] and len(info["inputs"]) == (
            4 if case == "four_inputs" else 2
        )
        owner = graph.blocks["left"].last_result.payloads["tensor"]
        assert ticket_key(result.payloads["tensors"]["ticket"]) == ticket_key(owner["ticket"])
        assert result.payloads["tensors"]["spatial"] == owner["spatial"]
        assert "tensor_preprocessing" not in result.payloads["tensors"]
        assert result.outputs["preprocessing"]["color_format"] == "rgb"
        if case == "two_sources":
            assert (
                info["inputs"][0]["spatial"]["root_id"] != info["inputs"][1]["spatial"]["root_id"]
            )
            assert ticket_key(info["inputs"][0]["ticket"]) != ticket_key(
                info["inputs"][1]["ticket"]
            )
        json.dumps(info, allow_nan=False)
    finally:
        graph.close()


def test_missing_or_bad_input_clears_a_warm_network_and_next_complete_input_recovers():
    block = model()
    try:
        assert block.execute(tensors()).passed
        assert block.execute({"input_1": tensors()["input_1"]}) is None
        assert "missing" in block.error and block._net is None
        assert not any(block.current_emissions.values())
        assert block.execute(tensors()).passed
        bad = tensors()
        bad["input_2"]["shape"] = [1, 64]
        assert block.execute(bad) is None and block._net is None
        assert block.execute(tensors()).passed
    finally:
        block.close()


def test_named_tensor_map_requires_explicit_selector_when_ambiguous():
    block = model()
    inputs = tensors()
    inputs["input_2"] = {"right": inputs["input_2"], "unused": inputs["input_1"]}
    assert block.execute(inputs) is None and "select an output name" in block.error
    block.parameters["input_2_tensor"] = "right"
    try:
        assert block.execute(inputs).passed
    finally:
        block.close()


def test_unknown_native_input_name_is_contained_and_valid_binding_recovers():
    block = model(input_2_name="missing_model_input")
    try:
        assert block.execute(tensors()) is None and block._net is None
        assert not any(block.current_emissions.values())
        block.parameters["input_2_name"] = "right"
        assert block.execute(tensors()).passed
    finally:
        block.close()


def test_declared_subset_of_required_model_inputs_is_not_allowed_to_reuse_old_values():
    block = model()
    try:
        assert block.execute(tensors()).passed
        block.parameters["active_inputs"] = 1
        assert block.execute({"input_1": tensors()["input_1"]}) is None
        assert block._net is None and not any(block.current_emissions.values())
        block.parameters["active_inputs"] = 2
        assert block.execute(tensors()).passed
    finally:
        block.close()


@pytest.mark.parametrize("policy", ["same_ticket", "same_value", "unchecked"])
def test_ticket_policy_is_explicit_and_reference_roles_remain_distinct(policy):
    left = TicketClock(session="left-session").issue("left")
    right = TicketClock(session="right-session").issue("right")
    inputs = {
        name: ValuePacket(value, {"ticket": ticket})
        for (name, value), ticket in zip(tensors().items(), (left, right), strict=True)
    }
    block = model(ticket_policy=policy)
    try:
        result = block.execute(inputs)
        if policy == "same_ticket":
            assert result is None and "different tickets" in block.error
        else:
            assert result.passed
            assert ticket_key(result.payloads["tensors"]["ticket"]) == ticket_key(left)
        block.parameters["input_2_role"] = "reference"
        assert block.execute(inputs).passed
    finally:
        block.close()


def test_partly_missing_current_ticket_and_recipe_conflicts_are_invalid():
    ticket = TicketClock().issue("left")
    raw = tensors()
    inputs = {"input_1": ValuePacket(raw["input_1"], {"ticket": ticket}), "input_2": raw["input_2"]}
    block = model()
    assert block.execute(inputs) is None and "Every Current" in block.error
    refs = [{"id": "part", "revision": n, "digest": str(n) * 64} for n in (1, 2)]
    inputs = {
        name: ValuePacket(value, {"ticket": ticket, "recipe_ref": ref})
        for (name, value), ref in zip(raw.items(), refs, strict=True)
    }
    assert block.execute(inputs) is None and "recipe revisions" in block.error
    inputs["input_2"].payload["recipe_ref"] = dict(refs[0])
    try:
        assert block.execute(inputs).passed
    finally:
        block.close()


@pytest.mark.parametrize(
    "parameters",
    [
        {"input_2_name": "left"},
        {"input_2_shape": [0, 3]},
        {"input_2_role": "other"},
        {"active_inputs": 0},
        {"active_inputs": 1.0},
        {"primary_input": "input_3"},
        {"input_1_role": "reference"},
        {"maximum_total_elements": 0},
        {"fusion": 1},
    ],
)
def test_invalid_bindings_validate_without_opening_a_model(parameters, monkeypatch):
    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda *a: pytest.fail("Opened invalid model"))
    block = model(**parameters)
    with pytest.raises(ValidationError):
        block.validate_parameters()
    assert block.execute(tensors()) is None


@pytest.mark.parametrize(
    "field,value",
    [("maximum_input_elements", 100), ("maximum_total_elements", 200), ("max_output_elements", 1)],
)
def test_input_and_output_limits_reject_without_partial_results(field, value):
    block = model(**{field: value})
    assert block.execute(tensors()) is None
    assert block._net is None and not any(block.current_emissions.values())


def test_tensor_builder_data_override_boolean_widening_and_float32_overflow():
    block = MakeTensor(parameters={"shape": [1, 2], "values": [2, 3]})
    assert block.execute({}).outputs["tensor"]["values"] == [2.0, 3.0]
    assert block.execute({"data": [True, False]}).outputs["tensor"]["values"] == [1.0, 0.0]
    assert block.execute({"data": [1e300, 2]}) is None
    assert block.execute({"data": [1]}) is None


@pytest.mark.parametrize("layout", ["nchw", "nhwc"])
def test_prepared_pixels_geometry_pyramid_recipe_and_source_are_preserved(layout):
    image = np.full((10, 20, 3), [20, 80, 200], np.uint16)
    before = image.copy()
    ticket = TicketClock().issue("camera")
    context = {
        "ticket": ticket,
        "spatial": frame([20, 10], "camera"),
        "pyramid": {
            "level": 1,
            "largest_size": [40, 20],
            "to_largest": np.diag([2, 2, 1]).tolist(),
        },
        "recipe_ref": {"id": "part", "revision": 1, "digest": "0" * 64},
    }
    block = PrepareImageTensor(
        parameters={
            "width": 8,
            "height": 8,
            "layout": layout,
            "resize_mode": "letterbox",
            "scale": 1 / 200,
        }
    )
    result = block.execute({"image": ValuePacket(image, context)})
    assert result is not None, block.error
    expected_shape = [1, 3, 8, 8] if layout == "nchw" else [1, 8, 8, 3]
    assert result.outputs["tensor"]["shape"] == expected_shape
    values = np.array(result.outputs["tensor"]["values"]).reshape(expected_shape)
    if layout == "nhwc":
        values = values.transpose(0, 3, 1, 2)
    np.testing.assert_allclose(values[0, :, 3, 3], [1, 0.4, 0.1], atol=1e-6)
    np.testing.assert_array_equal(values[:, :, :2], 0)
    np.testing.assert_array_equal(image, before)
    assert result.payloads["tensor"]["pyramid"] == context["pyramid"]
    assert result.payloads["tensor"]["recipe_ref"] == context["recipe_ref"]
    assert result.payloads["tensor"]["tensor_preprocessing"]["input_size"] == [20, 10]


def test_prepare_selection_and_preallocation_parameter_guards(monkeypatch):
    image = np.full((8, 8, 3), 255, np.uint8)
    block = PrepareImageTensor(parameters={"width": 8, "height": 8})
    out = block.execute({"image": image, "mask": np.zeros((8, 8), np.uint8)})
    assert out.outputs["tensor"]["values"] == [0.0] * (3 * 8 * 8)
    block.parameters.update(width=4096, height=4096)
    monkeypatch.setattr(cv2, "resize", lambda *a, **k: pytest.fail("Allocated before budget check"))
    assert block.execute({"image": image}) is None and "budget" in block.error


def test_inconsistent_prepared_tensor_provenance_is_rejected():
    raw = tensors()
    context = {
        "spatial": frame([40, 24], "camera"),
        "tensor_preprocessing": {
            "tensor_shape": [1, 1, 8, 8],
            "dtype": "float32",
            "layout": "nchw",
            "input_size": [40, 24],
        },
    }
    raw["input_1"] = ValuePacket(raw["input_1"], context)
    block = model()
    assert block.execute(raw) is None and "provenance disagrees" in block.error


def test_same_value_timestamp_policy_normalizes_equivalent_timezones():
    left = TicketClock().issue("left", "external", {"timestamp": "2026-11-17T01:00:00-05:00"})
    right = dict(left, session="another-camera", value="2026-11-17T06:00:00Z")
    raw = tensors()
    inputs = {
        name: ValuePacket(value, {"ticket": ticket})
        for (name, value), ticket in zip(raw.items(), (left, right), strict=True)
    }
    block = model(ticket_policy="same_value")
    try:
        assert block.execute(inputs).passed
    finally:
        block.close()


def test_shared_single_input_selects_multi_output_subset_and_unknown_names_fail():
    path = BASE.parent / "dnn_detection/assets/proposal_heads.onnx"
    block = DNNInference(
        parameters={"model_path": str(path), "width": 64, "height": 64, "output_names": ["rows"]}
    )
    try:
        result = block.execute({"image": np.full((16, 16, 3), 200, np.uint8)})
        assert result is not None and list(result.outputs["tensors"]) == ["rows"], block.error
        block.parameters["output_names"] = ["missing"]
        assert block.execute({"image": np.full((16, 16, 3), 200, np.uint8)}) is None
        assert block._net is None
    finally:
        block.close()


def test_metadata_dropdowns_native_handles_omitted_and_xml_save_as(tmp_path):
    registry = builtin_registry()
    for kind in ("ml.image_tensor", "ml.make_tensor", "ml.dnn_multi"):
        block = registry.create(kind)
        block.validate_parameters()
        assert block.parameter_help.keys() == block.defaults.keys()
        assert "Deep Neural Networks (dnn)" in block_entry(block).path
    graph = Graph.load(BASE / "paired_images.json", registry)
    try:
        graph.run()
        graph.save(tmp_path / "project.json")
        xml = tmp_path / "project.xml"
        export_project_xml(graph.to_dict(), xml)
        restored = Graph.from_dict(read_project_xml(xml), registry, project_dir=tmp_path)
        try:
            assert restored.blocks["multi"]._net is None
            restored.run()
            assert restored.blocks["classify"].last_result.passed
        finally:
            restored.close()
    finally:
        graph.close()


def test_full_regeneration_and_hashes_preserve_original_collection(tmp_path):
    before = {
        p.relative_to(BASE): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in BASE.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    spec = importlib.util.spec_from_file_location("multi_example", BASE / "build_example.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = tmp_path
    module.build()
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / "assets" / name).read_bytes()).hexdigest() == digest
        assert hashlib.sha256((tmp_path / "assets" / name).read_bytes()).hexdigest() == digest
    for path in BASE.glob("*.json"):
        assert json.loads((tmp_path / path.name).read_bytes()) == json.loads(path.read_bytes())
    assert all(
        hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest
        for name, digest in before.items()
    )
