"""Q/DQ arithmetic, pretrained fidelity, float boundary guards and portable projects."""

import copy
import hashlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.batch_inference import BatchImageInference
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.dnn_blocks import DNNInference
from opencv_dataflow.ml_common import selected_tensor, tensor_record
from opencv_dataflow.multi_inference import MultiInputInference
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket, frame
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.run_profiles import profile
from opencv_dataflow.tensor_comparison import CompareImageTensors
from opencv_dataflow.tickets import TicketClock, ticket_key

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/quantized_inference"
TRUTH = json.loads((BASE / "truth.json").read_bytes())


def comparison(reference, candidate, **parameters):
    b = CompareImageTensors(parameters=parameters)
    args = {
        "image": np.zeros((32, 40, 3), np.uint16),
        "reference": {"a": tensor_record(reference, 1000)},
        "candidate": {"b": tensor_record(candidate, 1000)},
    }
    r = b.execute(args)
    assert r is not None, b.error
    return b, r, args


def test_numeric_error_metrics_relative_reference_tolerance_hashes_and_no_class_event():
    a = np.float32([[0, 2], [-2, 4]])
    b = np.float32([[0.125, 2.25], [-2.5, 4]])
    block, r, args = comparison(
        a, b, task="tensor", absolute_tolerance=0.125, relative_tolerance=0.125
    )
    delta = b.astype(float) - a
    assert r.outputs["maximum_error"] == pytest.approx(abs(delta).max())
    assert r.outputs["rmse"] == pytest.approx(np.sqrt((delta * delta).mean()))
    assert r.outputs["report"]["bias"] == pytest.approx(delta.mean())
    assert r.outputs["report"]["violating_elements"] == 1 and not r.passed
    assert r.outputs["top1_equal"] is NO_VALUE
    assert (
        r.outputs["report"]["reference_float32_sha256"]
        == hashlib.sha256(a.astype("<f4").tobytes()).hexdigest()
    )
    assert r.graphics and r.drawings
    block.run_profile = profile("lean")
    lean = block.execute(args)
    assert not lean.graphics and not lean.drawings and lean.outputs["report"] == r.outputs["report"]
    np.testing.assert_array_equal(lean.outputs["image"], args["image"])


def test_classification_ties_topk_and_top1_have_independent_failure_reasons():
    _, r, _ = comparison(
        [[1, 1, 0]],
        [[0.99, 1, 0]],
        absolute_tolerance=0.02,
        relative_tolerance=0,
        top_k=1,
        minimum_topk_overlap=1,
    )
    assert not r.passed and r.outputs["report"]["violating_elements"] == 0
    assert {v["code"] for v in r.fail_reasons} == {
        "tensor_comparison.top1",
        "tensor_comparison.topk",
    }
    _, relaxed, _ = comparison(
        [[1, 1, 0]], [[0.99, 1, 0]], absolute_tolerance=0.02, require_top1=False
    )
    assert relaxed.passed and not relaxed.outputs["top1_equal"]


def test_softmax_uses_independent_stable_distributions_and_preserves_raw_hashes():
    _, r, _ = comparison(
        [[1000, 999, -1000]],
        [[1001, 1000, -999]],
        comparison_space="softmax",
        absolute_tolerance=0,
        relative_tolerance=0,
    )
    assert r.passed and r.outputs["maximum_error"] == 0
    assert (
        r.outputs["report"]["reference_float32_sha256"]
        != r.outputs["report"]["candidate_float32_sha256"]
    )


@pytest.mark.parametrize(
    "params",
    [
        {"absolute_tolerance": -1},
        {"relative_tolerance": float("inf")},
        {"top_k": True},
        {"task": "tensor", "comparison_space": "softmax"},
        {"minimum_topk_overlap": 1.1},
        {"plot_items": 51},
        {"require_top1": 1},
        {"reference_output_name": 4},
    ],
)
def test_comparison_validation_is_pure(params):
    with pytest.raises(ValidationError):
        CompareImageTensors(parameters=params).validate_parameters()


@pytest.mark.parametrize(
    "fault", ["shape", "dtype", "budget", "batch", "ticket", "grid", "recipe", "pyramid"]
)
def test_invalid_tensor_or_known_context_conflicts_emit_nothing(fault):
    ctx = {"ticket": TicketClock().issue("source"), "spatial": frame([40, 32], "source")}
    a = {"a": tensor_record([[1, 2, 3]], 100)}
    b = copy.deepcopy(a)
    args = {
        "image": ValuePacket(np.zeros((32, 40, 3), np.uint8), copy.deepcopy(ctx)),
        "reference": ValuePacket(a, copy.deepcopy(ctx)),
        "candidate": ValuePacket(b, copy.deepcopy(ctx)),
    }
    params = {}
    if fault == "shape":
        b["a"] = tensor_record([1, 2, 3], 100)
    elif fault == "dtype":
        b["a"]["dtype"] = "uint8"
    elif fault == "budget":
        params["maximum_elements"] = 2
    elif fault == "batch":
        a["a"] = b["a"] = tensor_record([[1, 2], [3, 4]], 100)
    elif fault == "ticket":
        args["candidate"].payload["ticket"] = TicketClock().issue("other")
    elif fault == "grid":
        args["candidate"].payload["spatial"]["to_root"][0][2] = 1
    elif fault == "recipe":
        args["candidate"].payload["recipe_ref"] = {"id": "part", "revision": 2, "digest": "b" * 64}
        args["reference"].payload["recipe_ref"] = {"id": "part", "revision": 1, "digest": "a" * 64}
    else:
        for k in ("reference", "candidate"):
            args[k].payload["pyramid"] = {
                "level": 0,
                "largest_size": [40, 32],
                "to_largest": np.eye(3).tolist(),
            }
        args["candidate"].payload["pyramid"]["level"] = 1
    block = CompareImageTensors(parameters=params)
    assert block.execute(args) is None and not block.current_emissions


def independent_scores(image, quantized=True, unsigned=False, scale=None, selected=False):
    rgb = image[..., ::-1].astype(np.float32)
    if selected:
        masked = np.zeros_like(rgb)
        masked[4:20, 6:26] = rgb[4:20, 6:26]
        rgb = masked
    rgb = cv2.resize(rgb, (16, 16))
    middle = 32767.5 if image.dtype == np.uint16 else 127.5
    prepared = (rgb - middle) * (1 / middle if scale is None else scale)
    coefficients = np.array(TRUTH["coefficient"], np.float32)
    if quantized:
        zero = TRUTH["unsigned_zero_point" if unsigned else "signed_zero_point"]
        limits = (0, 255) if unsigned else (-128, 127)
        q = np.clip(np.rint(prepared / np.float32(0.05)) + zero, *limits)
        prepared = (q - zero) * np.float32(0.05)
        ws = np.array(TRUTH["weight_scale"], np.float32)
        coefficients = np.rint(coefficients / ws) * ws
    return np.maximum(prepared * coefficients + np.array(TRUTH["bias"], np.float32), 0).mean(
        axis=(0, 1)
    )


@pytest.mark.parametrize("case", TRUTH["cases"])
@pytest.mark.parametrize("mode", ["debug", "lean"])
def test_original_projects_qdq_rounding_saturation_context_and_judgments(case, mode):
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    try:
        graph.set_run_profile(mode)
        graph.run()
        candidate = graph.blocks["candidate"]
        compare = graph.blocks["compare"]
        if case in ("raw_output", "unsupported_op"):
            assert (
                candidate.last_result is None
                and candidate._net is None
                and not candidate.current_emissions
            )
            assert (
                "DequantizeLinear" if case == "raw_output" else "empty network"
            ) in candidate.error
            assert compare.last_result is None and not compare.current_emissions
            return
        image = graph.blocks["source"].last_result.outputs["image"]
        for key in ("reference", "candidate"):
            block = graph.blocks[key]
            r = block.last_result
            assert r is not None, block.error
            expected = independent_scores(
                image,
                key == "candidate",
                case.startswith("unsigned"),
                0.5 if case == "saturation" else None,
                case == "selected",
            )
            np.testing.assert_allclose(
                selected_tensor(r.outputs["tensors"])[0],
                expected,
                atol=0.001 if case == "saturation" and key == "candidate" else 3e-6,
            )
            assert r.outputs["receipt"]["calls"][0]["artifact"]["sha256"]
        result = compare.last_result
        assert result is not None, compare.error
        assert result.passed == (case not in ("strict_tolerance", "saturation", "top1_change"))
        assert bool(result.graphics) == (mode == "debug") and bool(result.drawings) == (
            mode == "debug"
        )
        assert ticket_key(result.payloads["report"]["ticket"]) == ticket_key(
            graph.blocks["source"].last_result.outputs["ticket"]
        )
        np.testing.assert_array_equal(result.outputs["image"], image)
        assert graph.blocks["filmstrip"].last_result is not None
        if mode == "debug":
            assert np.any(graph.blocks["overlay"].last_result.outputs["image"] != image)
    finally:
        graph.close()


@pytest.mark.parametrize("engine", ["classic", "new", "auto"])
@pytest.mark.parametrize("cls", [DNNInference, MultiInputInference, BatchImageInference])
def test_raw_8bit_outputs_reject_for_all_adapters_and_recover_with_dequantized_outputs(engine, cls):
    params = {
        "model_path": str(BASE / "assets/raw.onnx"),
        "output_names": ["codes"],
        "engine": engine,
        "input_name": "image",
        "width": 16,
        "height": 16,
    }
    if cls is MultiInputInference:
        params = {k: v for k, v in params.items() if k in cls.defaults}
        params.update(active_inputs=1, input_1_name="image", input_1_shape=[1, 3, 16, 16])
        inputs = {"input_1": tensor_record(np.full((1, 3, 16, 16), 0.5, np.float32), 10000)}
    elif cls is BatchImageInference:
        inputs = {
            "images": FlowGroup((ValuePacket(np.zeros((16, 16, 3), np.uint8), {}),), ("one",))
        }
    else:
        inputs = {"image": np.zeros((16, 16, 3), np.uint8)}
    block = cls(parameters=params)
    try:
        assert (
            block.execute(inputs) is None
            and "DequantizeLinear" in block.error
            and block._net is None
        )
        assert not block.current_emissions
        block.parameters.update(
            model_path=str(BASE / "assets/signed.onnx"), output_names=["scores"]
        )
        assert block.execute(inputs) is not None, block.error
    finally:
        block.close()


def test_exact_regeneration_pins_and_sealed_save_as_xml(tmp_path):
    build = runpy.run_path(str(BASE / "build_example.py"))["build"]
    generated = tmp_path / "generated"
    build(generated)
    for path in generated.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (BASE / path.relative_to(generated)).read_bytes(), path
    with pytest.raises(FileExistsError):
        build(generated)
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest
    graph = Graph.load(BASE / "sealed.json", builtin_registry())
    try:
        graph.run()
        expected = graph.blocks["compare"].last_result.outputs["report"]
        graph.save(tmp_path / "saved.json")
        export_project_xml(graph.to_dict(), tmp_path / "saved.xml")
        for key in ("reference", "candidate"):
            for name in ("model.onnx", "contract.json"):
                assert (tmp_path / f"bundle_{key}" / name).read_bytes() == (
                    BASE / f"bundle_{key}" / name
                ).read_bytes()
    finally:
        graph.close()
    for graph in (
        Graph.load(tmp_path / "saved.json", builtin_registry()),
        Graph.from_dict(
            read_project_xml(tmp_path / "saved.xml"), builtin_registry(), project_dir=tmp_path
        ),
    ):
        try:
            assert graph.blocks["candidate"]._net is None
            graph.run()
            assert graph.blocks["compare"].last_result.outputs["report"] == expected
        finally:
            graph.close()


@pytest.mark.parametrize("case", TRUTH["real_cases"])
@pytest.mark.parametrize("mode", ["debug", "lean"])
def test_pinned_pretrained_exports_native_parity_actual_layer_types_and_declared_fidelity(
    case, mode
):
    _, kind, engine = case.split("_")
    if not (BASE / f".local/{kind}_{engine}/model.onnx").exists():
        pytest.skip("Explicit MobileNet preparation required")
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    try:
        graph.set_run_profile(mode)
        graph.run()
        block = graph.blocks["candidate"]
        r = block.last_result
        assert r is not None, block.error
        # Same native state, a direct second forward used only by the independent test.
        direct = block._net.forward(list(block._net.getUnconnectedOutLayersNames()))[0]
        np.testing.assert_array_equal(selected_tensor(r.outputs["tensors"]), direct)
        result = graph.blocks["compare"].last_result
        assert result is not None and result.passed
        report = result.outputs["report"]
        classes = report["classification"]
        assert classes["reference_top1"] == classes["candidate_top1"] == 967
        assert classes["reference_topk"] == [967, 968, 504, 925, 809]
        assert classes["topk_overlap"] == (
            0.6 if kind == "qdq" and engine == "classic" else 1 if kind == "qdq" else 0.8
        )
        assert 0.3 < report["mean_absolute_error"] < 0.7 and report["maximum_absolute_error"] < 2.5
        expected_hash = {
            "qdq": "41a36090dafe98f4ad8f9b7fe0b218c56ac3c031e547f0367c30655d2702bffe",
            "qoperator": "cc028fe6cae7bc11a4ff53cfc9b79c920e8be65ce33a904ec3e2a8f66d77f95f",
        }[kind]
        call = r.outputs["receipt"]["calls"][0]
        assert call["artifact"]["sha256"] == expected_hash and call["configuration_verified"]
        types = {
            block._net.getLayer(block._net.getLayerId(s)).type for s in block._net.getLayerNames()
        }
        if kind == "qoperator" or engine != "classic":
            assert any("Int8" in t for t in types)
        assert call["native_placement_verified"] is False
        assert graph.blocks["filmstrip"].last_result is not None
    finally:
        graph.close()


def test_preparation_reuse_and_conflict_are_offline_and_preserve_bundles(monkeypatch, tmp_path):
    data = runpy.run_path(str(BASE / "prepare_models.py"))
    monkeypatch.setattr(
        data["prepare"].__globals__["urllib"].request,
        "urlopen",
        lambda *a, **k: pytest.fail("Unexpected download"),
    )
    if not (BASE / ".local/qdq_new/model.onnx").exists():
        pytest.skip("Explicit MobileNet preparation required")
    assert len(data["prepare"]()) == 9
    folder = tmp_path / "reference_classic"
    folder.mkdir()
    original = b"preserved"
    (folder / "contract.json").write_bytes(original)
    with pytest.raises((ValueError, KeyError)):
        data["prepare"](directory=tmp_path)
    assert (folder / "contract.json").read_bytes() == original


def test_cold_offscreen_worker_process_shutdown():
    script = r"""
import sys
from pathlib import Path
from PySide6.QtCore import QCoreApplication,Qt
from opencv_dataflow.app import Runner
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
app=QCoreApplication([]);registry=builtin_registry();graph=Graph.load(Path(sys.argv[1]),registry)
try:
 worker=Runner(graph,'once',registry);errors=[];frames=[]
 worker.frame.connect(lambda:(frames.append(1),worker.ack.set()),Qt.ConnectionType.DirectConnection)
 worker.failed.connect(errors.append,Qt.ConnectionType.DirectConnection);worker.start()
 if not worker.wait(20000):
  worker.stop();worker.ack.set();worker.wait(5000);raise AssertionError('worker timeout')
 assert not errors,errors
 assert worker.iterations==len(frames)==1
 assert graph.blocks['compare'].last_result.passed
finally:graph.close()
"""
    p = subprocess.run(
        [sys.executable, "-c", script, str(BASE / "signed_new.json")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=35,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert p.returncode == 0, p.stdout + p.stderr


@pytest.mark.parametrize("engine", ["classic", "new", "auto"])
@pytest.mark.parametrize("signed", [True, False])
def test_binary_scale_half_ties_expose_one_code_native_difference_and_guard_raw_outputs(
    tmp_path, engine, signed
):
    import onnx
    from onnx import TensorProto as T
    from onnx import helper as h
    from onnx import numpy_helper as n

    dtype = np.int8 if signed else np.uint8
    zero = -7 if signed else 117
    nodes = [
        h.make_node("QuantizeLinear", ["image", "scale", "zero"], ["codes"]),
        h.make_node("DequantizeLinear", ["codes", "scale", "zero"], ["dequantized"]),
    ]
    model = h.make_model(
        h.make_graph(
            nodes,
            "binary_tie_reference",
            [h.make_tensor_value_info("image", T.FLOAT, [1, 1, 16, 32])],
            [
                h.make_tensor_value_info("dequantized", T.FLOAT, [1, 1, 16, 32]),
                h.make_tensor_value_info("codes", T.INT8 if signed else T.UINT8, [1, 1, 16, 32]),
            ],
            [
                n.from_array(np.float32(1 / 32), "scale"),
                n.from_array(np.array(zero, dtype), "zero"),
            ],
        ),
        opset_imports=[h.make_opsetid("", 13)],
        ir_version=8,
    )
    onnx.checker.check_model(model)
    path = tmp_path / "ties.onnx"
    path.write_bytes(model.SerializeToString())
    image = np.resize((np.arange(-200, 200, dtype=np.float32) + 0.5) / 32, (16, 32))
    expected = (
        np.clip(np.rint(image * 32) + zero, *((-128, 127) if signed else (0, 255))) - zero
    ) / 32
    block = DNNInference(
        parameters={
            "model_path": str(path),
            "engine": engine,
            "input_name": "image",
            "output_names": ["dequantized"],
            "width": 32,
            "height": 16,
            "color_format": "gray",
            "mean": [0.0],
            "std": [1.0],
            "scale": 1.0,
        }
    )
    try:
        r = block.execute({"image": image})
        assert r is not None, block.error
        # This pinned native build rounds the shifted value. Odd zero points expose
        # a one-code half-tie discrepancy from the ONNX reference's round-then-shift.
        shifted = (
            np.clip(np.rint(image * 32 + zero), *((-128, 127) if signed else (0, 255))) - zero
        ) / 32
        observed = selected_tensor(r.outputs["tensors"])[0, 0]
        np.testing.assert_array_equal(observed, shifted)
        assert np.max(np.abs(observed - expected)) == 1 / 32
        image = image + np.float32(0.25 / 32)
        expected = (
            np.clip(np.rint(image * 32) + zero, *((-128, 127) if signed else (0, 255))) - zero
        ) / 32
        np.testing.assert_array_equal(
            selected_tensor(block.execute({"image": image}).outputs["tensors"])[0, 0], expected
        )
        block.parameters["output_names"] = ["codes"]
        assert block.execute({"image": image}) is None and "Raw 8-bit" in block.error
        assert block._net is None and not block.current_emissions
    finally:
        block.close()


@pytest.mark.parametrize("engine", ["classic", "new", "auto"])
def test_empty_native_import_is_reported_before_input_binding(monkeypatch, engine):
    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda *a, **k: cv2.dnn.Net())
    block = DNNInference(
        parameters={"model_path": str(BASE / "assets/signed.onnx"), "engine": engine}
    )
    assert block.execute({"image": np.zeros((16, 16, 3), np.uint8)}) is None
    assert "empty network" in block.error and "missing" not in block.error
    assert block._net is None and not block.current_emissions


def test_comparison_failed_input_admission_retains_history_and_emits_nothing():
    block, r, args = comparison([[1, 2, 3]], [[1, 2, 3]])
    previous = block.last_inputs
    block.input_failure_policies["candidate"] = "reject_failed"
    args["candidate"] = ValuePacket(
        args["candidate"],
        {
            "outcome": {
                "model": "tool_outcome_v1",
                "passed": False,
                "chain_passed": False,
                "source": {"id": "prior"},
                "fail_reasons": [],
            }
        },
    )
    assert block.execute(args) is None and block.input_rejected and not block.current_emissions
    assert block.last_inputs is previous and block.last_result is r


def test_comparison_rejects_oversized_or_nested_values_before_array_conversion(monkeypatch):
    import opencv_dataflow.tensor_comparison as module

    def forbidden(*a, **k):
        pytest.fail("Invalid input reached array conversion")

    monkeypatch.setattr(module, "selected_tensor", forbidden)
    for values in ([1, 2, 3], [[1, 2]], [True], "123"):
        data = {
            "scores": {"model": "tensor_v1", "dtype": "float32", "shape": [1, 3], "values": values}
        }
        with pytest.raises(ValidationError, match="bounded flat"):
            module.read_tensor(data, "scores", 2)
