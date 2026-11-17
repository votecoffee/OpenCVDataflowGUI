"""Native batch arithmetic, image ownership, atomic failures and saved examples."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.batch_inference import BatchImageInference, UnpackImageInference
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.control_flow import Unbatch
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.ml_common import selected_tensor
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import FlowGroup, ValuePacket, frame
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.tickets import TicketClock, ticket_key
from opencv_dataflow.timing import advance_timing, mark
from opencv_dataflow.toolbox import block_entry

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/batch_inference"
TRUTH = json.loads((BASE / "truth.json").read_bytes())


def model(**parameters):
    return BatchImageInference(
        parameters={
            "model_path": str(BASE / "assets/dynamic.onnx"),
            "width": 8,
            "height": 8,
            "batch_size": 2,
            "output_names": ["scores"],
            **parameters,
        }
    )


def group(n=3, h=12, w=16, depth=np.uint8):
    clock = TicketClock(session="batch-controls")
    entries = []
    for i in range(n):
        context = {
            "ticket": clock.issue(f"camera-{i}"),
            "spatial": frame([w, h], f"grid-{i}"),
            "companion": {"row": i},
            "processing_time": advance_timing(None, mark(), mark(), source=True),
        }
        image = np.full((h, w, 3), [10 + i * 10, 20 + i * 10, 100 + i * 10], depth)
        entries.append(ValuePacket(image, context))
    return FlowGroup(tuple(entries), tuple(f"row-{i}" for i in range(n)))


def scores(row):
    return selected_tensor(row.value["tensors"], "scores")[0]


@pytest.mark.parametrize("engine", ["auto", "classic", "new"])
def test_native_batch_rows_arithmetic_ticket_frame_and_full_rebinding(engine):
    block, images = model(engine=engine), group()
    originals = copy.deepcopy(images)
    try:
        result = block.execute({"images": images})
        assert result is not None, block.error
        rows = result.outputs["results"]
        assert rows.labels == images.labels and len(rows.items) == 3
        assert [c["submitted_shape"][0] for c in result.outputs["info"]["native_calls"]] == [2, 1]
        for index, (row, original) in enumerate(zip(rows.items, images.items, strict=True)):
            np.testing.assert_allclose(scores(row), original.value[0, 0, ::-1] / 255, atol=2e-6)
            np.testing.assert_array_equal(row.value["image"], original.value)
            assert row.value["index"] == index
            assert ticket_key(row.payload["ticket"]) == ticket_key(original.payload["ticket"])
            assert row.payload["spatial"] == original.payload["spatial"]
            assert row.payload["companion"] == {"row": index}
            assert (
                row.payload["processing_time"]["started_ns"]
                == original.payload["processing_time"]["started_ns"]
            )
            unpack = UnpackImageInference().execute({"result": row})
            assert unpack.passed and unpack.outputs["index"] == index
            assert unpack.payloads["tensors"]["spatial"] == original.payload["spatial"]
            assert unpack.payloads["image"]["ticket"] == original.payload["ticket"]
        # Equal signatures reuse one native handle; every activation binds fresh pixels.
        block.execute({"images": group(2)})
        native = block._net
        fresh = group(2)
        for entry in fresh.items:
            entry.value[:] = [200, 80, 20]
        changed = block.execute({"images": fresh})
        assert block._net is native
        np.testing.assert_allclose(
            scores(changed.outputs["results"].items[0]), [20 / 255, 80 / 255, 200 / 255], atol=2e-6
        )
        for original, after in zip(originals.items, images.items, strict=True):
            np.testing.assert_array_equal(original.value, after.value)
            assert original.payload == after.payload
    finally:
        block.close()
    assert block._net is None


def test_dynamic_batch_and_spatial_axes_change_across_activations():
    block = model(batch_size=4, spatial_mode="source")
    try:
        for n, h, w in ((1, 8, 8), (3, 13, 21), (2, 10, 6), (1, 7, 9)):
            inputs = group(n, h, w)
            result = block.execute({"images": inputs})
            assert result is not None, block.error
            assert result.outputs["info"]["native_calls"][0]["submitted_shape"] == [n, 3, h, w]
            for row, entry in zip(result.outputs["results"].items, inputs.items, strict=True):
                np.testing.assert_allclose(scores(row), entry.value[0, 0, ::-1] / 255, atol=2e-6)
                assert row.value["preprocessing"]["input_to_network"] == np.eye(3).tolist()
    finally:
        block.close()


def test_rgb_letterbox_and_pyramid_provenance_remain_on_original_grid():
    images = group(2, 8, 16)
    for entry in images.items:
        entry.value = ImageArray(entry.value, "rgb")
        entry.payload["pyramid"] = {
            "level": 1,
            "largest_size": [32, 16],
            "to_largest": np.diag([2, 2, 1]).tolist(),
        }
    block = model(resize_mode="letterbox")
    try:
        result = block.execute({"images": images})
        assert result is not None, block.error
        for row, original in zip(result.outputs["results"].items, images.items, strict=True):
            np.testing.assert_allclose(scores(row), original.value[0, 0] / 510, atol=2e-6)
            assert row.value["image"].color_format == "rgb"
            assert row.payload["pyramid"] == original.payload["pyramid"]
            assert row.value["preprocessing"]["input_to_network"] == [
                [0.5, 0, 0],
                [0, 0.5, 2],
                [0, 0, 1],
            ]
    finally:
        block.close()


@pytest.mark.parametrize("policy, expected", [("variable", [2, 1]), ("pad_repeat", [2, 2])])
def test_tail_policy_and_selected_output_order(policy, expected):
    block = model(tail_policy=policy, output_names=["squares", "scores"])
    try:
        result = block.execute({"images": group()})
        assert result is not None, block.error
        assert [c["submitted_shape"][0] for c in result.outputs["info"]["native_calls"]] == expected
        assert result.outputs["info"]["padding_rows"] == (policy == "pad_repeat")
        assert len(result.outputs["results"].items) == 3
        for row in result.outputs["results"].items:
            assert list(row.value["tensors"]) == ["squares", "scores"]
            np.testing.assert_allclose(
                selected_tensor(row.value["tensors"], "squares")[0], scores(row) ** 2, atol=2e-6
            )
    finally:
        block.close()


@pytest.mark.parametrize("case", list(TRUTH["cases"]))
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_projects_numerical_order_ownership_and_intentional_rejections(case, profile):
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    graph.set_run_profile(profile)
    try:
        graph.run()
        block, expected = graph.blocks["infer"], TRUTH["cases"][case]
        if "error" in expected:
            assert block.last_result is None and expected["error"] in block.error
            assert not any(block.current_emissions.values()) and block._net is None
            assert graph.blocks["classify"].last_result is None
            return
        assert block.last_result is not None, block.error
        rows = block.last_result.outputs["results"].items
        np.testing.assert_allclose(
            [scores(row) for row in rows], expected["scores"], atol=expected["tolerance"]
        )
        assert block.last_result.outputs["info"]["padding_rows"] == expected["padding_rows"]
        for i, row in enumerate(rows):
            original = graph.blocks[f"source_{i}"].last_result
            assert row.payload["ticket"] == original.outputs["ticket"]
            assert row.payload["spatial"] == original.payloads["image"]["spatial"]
        assert graph.blocks["classify"].last_result.outputs["label"] == expected["selected_label"]
        assert graph.blocks["classify"].last_result.passed
        assert graph.blocks["classify"].timing.count == (3 if case == "serial_results" else 1)
        assert graph.blocks["unpack"].last_output_images or profile == "lean"
    finally:
        graph.close()


def test_fixed_native_shape_wrong_native_output_and_warm_failure_recovery():
    block = model(
        model_path=str(BASE / "assets/fixed.onnx"),
        input_shape=[2, 3, 8, 8],
        tail_policy="pad_repeat",
    )
    try:
        assert block.execute({"images": group()}).passed
        # Deliberately false wildcard declaration cannot make a fixed Reshape accept N=1.
        block.parameters.update(input_shape=[-1, 3, 8, 8], tail_policy="variable")
        assert block.execute({"images": group(1)}) is None
        assert not any(block.current_emissions.values()) and block._net is None
        block.parameters["tail_policy"] = "pad_repeat"
        assert block.execute({"images": group(1)}).passed
        block.parameters["input_name"] = "missing"
        assert block.execute({"images": group(2)}) is None and block._net is None
        block.parameters["input_name"] = "image"
        assert block.execute({"images": group(2)}).passed
    finally:
        block.close()


def test_late_chunk_failure_emits_no_partial_results_and_recovers(monkeypatch):
    block = model()
    original = block.infer_tensors
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise cv2.error("forced late native error")
        return original(*args, **kwargs)

    monkeypatch.setattr(block, "infer_tensors", fail_second)
    assert block.execute({"images": group()}) is None
    assert "forced late" in block.error and block._net is None
    assert not any(block.current_emissions.values())
    monkeypatch.setattr(block, "infer_tensors", original)
    try:
        assert block.execute({"images": group()}).passed
    finally:
        block.close()


def test_nonfinite_image_rejects_after_warm_model_and_later_input_recovers():
    block = model()
    try:
        assert block.execute({"images": group(2)}).passed
        bad = group(2, depth=np.float32)
        bad.items[1].value[0, 0, 0] = np.nan
        assert block.execute({"images": bad}) is None and block._net is None
        assert not any(block.current_emissions.values())
        assert block.execute({"images": group(2)}).passed
    finally:
        block.close()


def test_scalar_per_image_vector_output_is_sliced_to_single_image_rows(monkeypatch):
    block = model()

    def native_rows(inputs, token, *, array_outputs):
        assert array_outputs
        blob = inputs[0][1]
        return {"scores": blob.mean(axis=(1, 2, 3))}, "scalar.onnx"

    monkeypatch.setattr(block, "infer_tensors", native_rows)
    result = block.execute({"images": group(2)})
    assert result is not None, block.error
    for row in result.outputs["results"].items:
        unpack = UnpackImageInference().execute({"result": row})
        assert unpack is not None
        assert unpack.outputs["tensors"]["scores"]["shape"] == [1]


def test_unpack_rejects_wrong_batch_or_source_geometry():
    block = model()
    try:
        result = block.execute({"images": group(2)})
        row = result.outputs["results"].items[0]
        row.value["tensors"]["scores"]["shape"] = [3, 1]
        unpack = UnpackImageInference()
        assert unpack.execute({"result": row}) is None and "one image row" in unpack.error
        row.value["tensors"]["scores"]["shape"] = [1, 3]
        row.value["preprocessing"]["input_size"] = [99, 99]
        assert unpack.execute({"result": row}) is None and not any(
            unpack.current_emissions.values()
        )
    finally:
        block.close()


def test_preflight_and_aggregate_output_budgets(monkeypatch):
    import opencv_dataflow.batch_inference as module

    original = module.preprocess
    monkeypatch.setattr(
        module, "preprocess", lambda *a: pytest.fail("Preprocessed before preflight")
    )
    for parameters in (
        {"maximum_images": 2},
        {"maximum_image_pixels": 100},
        {"maximum_source_pixels": 200},
        {"maximum_tensor_elements": 383},
        {"input_shape": [2, 3, 8, 8]},
        {"tail_policy": "reject"},
    ):
        block = model(**parameters)
        assert block.execute({"images": group()}) is None and block._net is None
    monkeypatch.setattr(module, "preprocess", original)
    block = model(max_output_elements=6)  # first chunk fits; second takes activation total to 9.
    assert block.execute({"images": group()}) is None and "activation element budget" in block.error
    assert not any(block.current_emissions.values()) and block._net is None


def test_masks_regions_raw_depth_normalization_and_independent_geometry():
    images = group(2, 8, 8, np.uint16)
    for entry in images.items:
        entry.value[:] = [1000, 2000, 4000]
    masks = FlowGroup(
        tuple(
            ValuePacket(
                np.pad(np.full((8, 4), 255, np.uint8), ((0, 0), (0, 4))), copy.deepcopy(e.payload)
            )
            for e in images.items
        ),
        images.labels,
    )
    regions = FlowGroup(
        tuple(
            ValuePacket(
                {
                    "model": "image_region_v1",
                    "shape": "rectangle",
                    "image_size": [8, 8],
                    "rectangle": [0, 0, 8, 4],
                },
                copy.deepcopy(e.payload),
            )
            for e in images.items
        ),
        images.labels,
    )
    block = model(color_format="rgb", mean=[100, 200, 300], std=[2, 4, 5], scale=0.01)
    try:
        result = block.execute({"images": images, "masks": masks, "regions": regions})
        assert result is not None, block.error
        expected = ((np.array([4000, 2000, 1000]) * 0.25) - [100, 200, 300]) * 0.01 / [2, 4, 5]
        for row in result.outputs["results"].items:
            np.testing.assert_allclose(scores(row), expected, atol=2e-6)
            np.testing.assert_array_equal(row.value["image"], images.items[0].value)
        masks.items[0].payload["ticket"] = TicketClock().issue("wrong")
        assert (
            block.execute({"images": images, "masks": masks}) is None
            and "different ticket" in block.error
        )
        masks.items[0].payload = copy.deepcopy(images.items[0].payload)
        masks.items[0].payload["spatial"]["root_id"] = "wrong-grid"
        assert (
            block.execute({"images": images, "masks": masks}) is None
            and "coordinate frames" in block.error
        )
        masks.labels = tuple(reversed(masks.labels))
        assert block.execute({"images": images, "masks": masks}) is None and "labels" in block.error
    finally:
        block.close()


def test_per_member_outcomes_and_reject_failed_input_without_history_change():
    images = group(2)
    images.items[1].payload["outcome"] = {
        "model": "tool_outcome_v1",
        "passed": False,
        "chain_passed": False,
        "source": {"id": "prior"},
        "fail_reasons": [{"code": "prior", "message": "Prior fail"}],
    }
    block = model()
    try:
        result = block.execute({"images": images})
        assert result is not None, block.error
        rows = result.outputs["results"].items
        assert rows[0].payload["outcome"]["chain_passed"]
        assert not rows[1].payload["outcome"]["chain_passed"]
        assert rows[1].payload["outcome"]["fail_reasons"][0]["code"] == "prior"
        previous, inputs, native = block.last_result, block.last_inputs, block._net
        block.input_failure_policies["images"] = "reject_failed"
        block.execute({"images": images})
        assert (
            block.last_result is previous and block.last_inputs is inputs and block._net is native
        )
        assert block.input_rejected and not any(block.current_emissions.values())
    finally:
        block.close()


def test_mixed_recipes_reject_even_with_independent_outcome_policy():
    images = group(2)
    for n, entry in enumerate(images.items):
        entry.payload["recipe_ref"] = {"id": "part", "revision": n + 1, "digest": str(n + 1) * 64}
    block = model()
    assert block.execute({"images": images}) is None and "recipe revisions" in block.error
    images.items[1].payload["recipe_ref"] = copy.deepcopy(images.items[0].payload["recipe_ref"])
    try:
        result = block.execute({"images": images})
        assert result.passed
        assert all(
            row.payload["recipe_ref"] == images.items[0].payload["recipe_ref"]
            for row in result.outputs["results"].items
        )
    finally:
        block.close()


@pytest.mark.parametrize(
    "parameters",
    [
        {"batch_size": True},
        {"batch_size": 0},
        {"maximum_images": 65},
        {"input_shape": [-1, 3]},
        {"input_shape": [-2, 3, 8, 8]},
        {"fusion": 1},
        {"tail_policy": "drop"},
        {"input_name": " "},
        {"std": [1, 0, 1]},
        {"mean": [1, 2]},
        {"width": 8.5},
        {"scale": float("inf")},
        {"padding_value": True},
    ],
)
def test_invalid_configuration_is_pure(parameters):
    block = model(**parameters)
    with pytest.raises((ValidationError, ValueError, TypeError)):
        block.validate_parameters()
    assert block._net is None and not block.last_inputs


def test_native_row_unbatch_and_json_xml_save_as(tmp_path):
    registry = builtin_registry()
    for kind in ("ml.dnn_batch", "ml.unpack_inference"):
        block = registry.create(kind)
        block.validate_parameters()
        assert "Deep Neural Networks (dnn)" in block_entry(block).path
    block = model()
    try:
        result = block.execute({"images": group()})
        serial = Unbatch().execute({"group": result.outputs["results"]})
        assert [
            UnpackImageInference().execute({"result": row}).outputs["index"]
            for row in serial.emissions["value"]
        ] == [0, 1, 2]
    finally:
        block.close()
    graph = Graph.load(BASE / "padded_fixed.json", registry)
    try:
        graph.run()
        graph.save(tmp_path / "saved.json")
        export_project_xml(graph.to_dict(), tmp_path / "saved.xml")
        restored = Graph.from_dict(
            read_project_xml(tmp_path / "saved.xml"), registry, project_dir=tmp_path
        )
        try:
            assert restored.blocks["infer"]._net is None
            restored.run()
            assert restored.blocks["classify"].last_result.passed
        finally:
            restored.close()
    finally:
        graph.close()


def test_regenerate_exact_assets_and_projects_without_modifying_originals(tmp_path):
    before = {
        p.relative_to(BASE): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in BASE.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    spec = importlib.util.spec_from_file_location("batch_example", BASE / "build_example.py")
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
