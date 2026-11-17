"""Pinned model/settings, explicit fallback and portable per-activation receipts."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.assistant_cli import main
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.dnn_blocks import DNNInference
from opencv_dataflow.ml_common import selected_tensor
from opencv_dataflow.model_contracts import export_contract, read_contract, revision
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.project_xml import export_project_xml, read_project_xml

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/model_contracts"
TRUTH = json.loads((BASE / "truth.json").read_bytes())
MODEL = BASE.parent / "batch_inference/assets/dynamic.onnx"
IMAGE = np.full((8, 8, 3), [20, 40, 200], np.uint8)


def inference(**parameters):
    return DNNInference(
        parameters={
            "model_path": str(MODEL),
            "input_name": "image",
            "width": 8,
            "height": 8,
            "output_names": ["scores"],
            **parameters,
        }
    )


@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize("case", list(TRUTH["cases"]))
def test_projects_against_independent_arithmetic_and_receipt_ownership(case, profile):
    if case == "strict_target" and cv2.dnn.DNN_TARGET_CUDA in cv2.dnn.getAvailableTargets(
        cv2.dnn.DNN_BACKEND_CUDA
    ):
        pytest.skip("This negative example requires an unavailable CUDA target")
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    graph.set_run_profile(profile)
    try:
        graph.run()
        expected = TRUTH["cases"][case]
        block = graph.blocks[expected["block"]]
        if "error" in expected:
            assert block.last_result is None and expected["error"] in block.error
            assert block._net is None and not any(block.current_emissions.values())
            return
        result = block.last_result
        assert result is not None, block.error
        assert graph.blocks["classify"].last_result.outputs["label"] == expected["label"]
        if case == "receipts_off":
            assert result.outputs["receipt"] is NO_VALUE
            return
        receipt = result.outputs["receipt"]
        assert receipt["model"] == "onnx_inference_receipt_v1"
        assert len(receipt["calls"]) == (2 if case == "pinned_batch" else 1)
        for call in receipt["calls"]:
            assert call["contract_revision"] and call["artifact"]["sha256"]
            assert call["configuration_verified"]
            assert call["external_preprocessing_verified"] == (case == "pinned_multi")
            assert call["environment"]["opencv"] == cv2.__version__
            assert not call["native_placement_verified"]
            assert all(type(t) is int for t in call["requested_available_target_ids"])
            assert cv2.dnn.DNN_TARGET_CPU in call["selected_available_target_ids"]
        if case == "pinned_batch":
            for i, packet in enumerate(result.outputs["results"].items):
                assert packet.value["receipt"]["group_index"] == i
                assert packet.value["receipt"]["native_row_index"] == i % 2
                assert packet.value["receipt"]["native_call_index"] == i // 2
            row = graph.blocks["unpack"].last_result
            assert row.outputs["receipt"]["group_index"] == 1
            assert row.payloads["receipt"]["ticket"] == row.payloads["image"]["ticket"]
        else:
            np.testing.assert_allclose(
                selected_tensor(result.outputs["tensors"], "scores"),
                [expected["scores"]],
                atol=2e-6,
            )
            assert result.payloads["receipt"]["ticket"] == result.payloads["tensors"]["ticket"]
        if case == "cpu_fallback":
            call = receipt["calls"][0]
            if cv2.dnn.DNN_TARGET_CUDA not in cv2.dnn.getAvailableTargets(cv2.dnn.DNN_BACKEND_CUDA):
                assert call["fallback"] == "requested_target_unavailable"
                assert call["selected_execution"]["target"] == "cpu"
                assert call["requested_execution"]["target"] == "cuda"
        json.dumps(receipt, allow_nan=False)
    finally:
        graph.close()


def test_detailed_hashes_native_loaded_identity_and_cache_lifecycle():
    block = inference(receipt_mode="detailed")
    try:
        result = block.execute({"image": IMAGE})
        assert result is not None, block.error
        call = result.outputs["receipt"]["calls"][0]
        expected = np.broadcast_to(
            (np.array([200, 40, 20], np.float32) * np.float32(1 / 255))[:, None, None], (3, 8, 8)
        )[None]
        assert (
            call["inputs"][0]["sha256"]
            == hashlib.sha256(np.ascontiguousarray(expected).tobytes()).hexdigest()
        )
        native = selected_tensor(result.outputs["tensors"], "scores")
        assert call["outputs"][0]["sha256"] == hashlib.sha256(native.tobytes()).hexdigest()
        assert call["artifact"]["sha256"] == hashlib.sha256(MODEL.read_bytes()).hexdigest()
        second = block.execute({"image": IMAGE})
        assert second.outputs["receipt"]["calls"][0]["cache_reused"]
        assert len(second.outputs["receipt"]["calls"]) == 1
        block.parameters["receipt_mode"] = "summary"
        assert (
            "sha256"
            not in block.execute({"image": IMAGE}).outputs["receipt"]["calls"][0]["inputs"][0]
        )
        block.parameters["receipt_mode"] = "off"
        assert block.execute({"image": IMAGE}).outputs["receipt"] is NO_VALUE
        assert not block.current_emissions.get("receipt") and block._snapshot is None
    finally:
        block.close()
    assert block._net is None and not block._receipt_calls


def test_bound_model_same_size_and_restored_mtime_tampering_is_rechecked_when_off(tmp_path):
    block = inference(receipt_mode="off")
    export_contract(block, tmp_path / "bundle")
    block.project_dir = tmp_path
    block.parameters.update(contract_path="bundle/contract.json", model_path="")
    path = tmp_path / "bundle/model.onnx"
    original, stamp = path.read_bytes(), path.stat()
    assert block.execute({"image": IMAGE}).passed
    path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert block.execute({"image": IMAGE}) is None and "checksum" in block.error
    assert block._net is None and not any(block.current_emissions.values())
    path.write_bytes(original)
    try:
        assert block.execute({"image": IMAGE}).passed
    finally:
        block.close()


def test_contract_export_does_not_infer_and_preserves_existing_destinations(tmp_path, monkeypatch):
    block = inference()
    monkeypatch.setattr(
        cv2.dnn, "readNetFromONNX", lambda *a: pytest.fail("Export executed inference")
    )
    document = export_contract(block, tmp_path / "bundle")
    assert document == read_contract(tmp_path / "bundle/contract.json")
    assert block._net is None and block.last_result is None
    raw = (tmp_path / "bundle/model.onnx").read_bytes()
    with pytest.raises(FileExistsError):
        export_contract(block, tmp_path / "bundle")
    assert (tmp_path / "bundle/model.onnx").read_bytes() == raw
    assert (
        main(
            [
                "export-model",
                str(BASE / "pinned_single.json"),
                "--block",
                "infer",
                "--output",
                str(tmp_path / "cli"),
            ]
        )
        == 0
    )
    assert read_contract(tmp_path / "cli/contract.json")["revision"]


def test_explicit_export_can_pin_changed_settings_as_a_new_revision(tmp_path):
    block = inference()
    original = export_contract(block, tmp_path / "original")
    block.project_dir = tmp_path
    block.parameters.update(contract_path="original/contract.json", scale=2 / 255, model_path="")
    assert block.execute({"image": IMAGE}) is None and "settings differ" in block.error
    changed = export_contract(block, tmp_path / "changed")
    assert changed["revision"] != original["revision"]
    assert read_contract(tmp_path / "original/contract.json") == original
    block.parameters["contract_path"] = "changed/contract.json"
    try:
        assert block.execute({"image": IMAGE}).passed
    finally:
        block.close()


@pytest.mark.parametrize(
    "kind", ["revision", "duplicate", "nan", "schema", "traversal", "size", "bom"]
)
def test_invalid_manifest_and_artifact_schema_cannot_run(kind, tmp_path):
    block = inference()
    document = export_contract(block, tmp_path / "bundle")
    if kind == "revision":
        document["revision"] = "0" * 64
    elif kind == "schema":
        document["schema_version"] = True
    elif kind == "traversal":
        document["artifact"]["file"] = "../model.onnx"
    elif kind == "size":
        document["artifact"]["bytes"] = True
    if kind in ("schema", "traversal", "size"):
        document["revision"] = revision({k: v for k, v in document.items() if k != "revision"})
    raw = (json.dumps(document) + "\n").encode()
    if kind == "duplicate":
        raw = raw.replace(b'"schema_version": 1', b'"schema_version": 1, "schema_version": 1')
    elif kind == "nan":
        raw = raw.replace(b'"schema_version": 1', b'"schema_version": NaN')
    elif kind == "bom":
        raw = b"\xef\xbb\xbf" + raw
    (tmp_path / "bundle/contract.json").write_bytes(raw)
    block.project_dir = tmp_path
    block.parameters["contract_path"] = "bundle/contract.json"
    assert block.execute({"image": IMAGE}) is None and block._net is None
    assert not any(block.current_emissions.values())


@pytest.mark.parametrize(
    "path", ["../bundle.json", "C:/bundle.json", "a\\bundle.json", "/bundle.json", "a/../b", False]
)
def test_contract_path_validation_is_pure(path):
    block = inference(contract_path=path)
    with pytest.raises(ValidationError):
        block.validate_parameters()
    assert block._net is None


def test_availability_cpu_fallback_is_opt_in_and_native_errors_are_not_retried(monkeypatch):
    original = cv2.dnn.getAvailableTargets
    monkeypatch.setattr(
        cv2.dnn,
        "getAvailableTargets",
        lambda backend: [] if backend == cv2.dnn.DNN_BACKEND_CUDA else original(backend),
    )
    block = inference(engine="classic", backend="cuda", target="cuda")
    assert block.execute({"image": IMAGE}) is None and "unavailable" in block.error
    block.parameters["fallback_policy"] = "cpu_if_unavailable"
    try:
        receipt = block.execute({"image": IMAGE}).outputs["receipt"]["calls"][0]
        assert receipt["fallback"] and receipt["selected_execution"]["backend"] == "opencv"
        block.parameters["input_name"] = "missing"
        assert block.execute({"image": IMAGE}) is None and not any(block.current_emissions.values())
        assert block._net is None and not block._receipt_calls
    finally:
        block.close()


def test_save_as_and_xml_copy_both_bundle_files_and_omit_native_state(tmp_path):
    graph = Graph.load(BASE / "pinned_multi.json", builtin_registry())
    try:
        graph.run()
        graph.save(tmp_path / "saved.json")
        assert (tmp_path / "bundles/multi/model.onnx").read_bytes() == (
            BASE / "bundles/multi/model.onnx"
        ).read_bytes()
        export_project_xml(graph.to_dict(), tmp_path / "saved.xml")
        restored = Graph.from_dict(
            read_project_xml(tmp_path / "saved.xml"), builtin_registry(), project_dir=tmp_path
        )
        try:
            assert restored.blocks["multi"]._net is None
            restored.run()
            assert restored.blocks["classify"].last_result.passed
            assert restored.blocks["multi"].last_result.outputs["receipt"]["calls"][0][
                "contract_revision"
            ]
        finally:
            restored.close()
        conflict = tmp_path / "other/bundles/multi/model.onnx"
        conflict.parent.mkdir(parents=True)
        conflict.write_bytes(b"existing owner")
        with pytest.raises(FileExistsError):
            graph.save(tmp_path / "other/saved.json")
        assert conflict.read_bytes() == b"existing owner"
    finally:
        graph.close()


def test_exact_example_regeneration_preserves_source_and_pinned_assets(tmp_path):
    before = {
        p.relative_to(BASE): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in BASE.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    spec = importlib.util.spec_from_file_location("contract_example", BASE / "build_example.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = tmp_path
    module.build()
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    for path in BASE.glob("*.json"):
        assert json.loads((tmp_path / path.name).read_bytes()) == json.loads(path.read_bytes())
    assert all(
        hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest
        for name, digest in before.items()
    )
