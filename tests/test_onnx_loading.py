"""Native engine discrimination, immutable snapshots, cleanup and cache receipts."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
import pytest

from opencv_dataflow.assistant_api import ProjectSession, validate_graph
from opencv_dataflow.core import Graph
from opencv_dataflow.dnn_blocks import DNNInference
from opencv_dataflow.ml_common import selected_tensor
from opencv_dataflow.onnx_loading import load_onnx_snapshot
from opencv_dataflow.project_xml import export_project_xml, read_project_xml

ROOT = Path(__file__).resolve().parents[1]
RGB = ROOT / "examples/assets/rgb_average.onnx"
NEW_ONLY = ROOT / "examples/opencv tools/dnn_detection/assets/proposal_heads.onnx"
EXAMPLES = ROOT / "examples/opencv tools/dnn_engines"


@pytest.mark.local_examples
def test_classic_uses_exact_frozen_bytes_and_cleans_temp_before_forward(monkeypatch, tmp_path):
    import opencv_dataflow.onnx_loading as module

    raw = RGB.read_bytes()
    original = cv2.dnn.readNetFromONNX
    opened = []

    def inspect(path, engine):
        assert isinstance(path, str) and engine == cv2.dnn.ENGINE_CLASSIC
        snapshot = Path(path)
        opened.append(snapshot)
        assert snapshot.read_bytes() == raw
        assert snapshot != RGB and snapshot.parent == tmp_path
        return original(path, engine)

    monkeypatch.setattr(
        module, "NamedTemporaryFile", lambda **kw: NamedTemporaryFile(dir=tmp_path, **kw)
    )
    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", inspect)
    network, evidence = load_onnx_snapshot(raw, cv2.dnn.ENGINE_CLASSIC)
    assert len(opened) == 1 and not opened[0].exists()
    assert not list(tmp_path.iterdir())
    network.setInput(np.full((1, 3, 32, 32), 0.25, np.float32))
    np.testing.assert_allclose(network.forward(), [[0.25, 0.25, 0.25]], atol=1e-6)
    assert evidence == {
        "model": "onnx_import_v1",
        "transport": "temporary_snapshot_file",
        "engine_argument": cv2.dnn.ENGINE_CLASSIC,
        "snapshot_bytes": len(raw),
        "temporary_file_retained": False,
        "native_engine_observed": False,
    }
    assert str(tmp_path) not in json.dumps(evidence)


@pytest.mark.local_examples
def test_cleanup_when_native_classic_import_rejects_and_no_hidden_auto_retry(monkeypatch, tmp_path):
    import opencv_dataflow.onnx_loading as module

    raw = NEW_ONLY.read_bytes()
    original = cv2.dnn.readNetFromONNX
    opened = []

    def inspect(path, engine):
        assert isinstance(path, str) and engine == cv2.dnn.ENGINE_CLASSIC
        opened.append(Path(path))
        assert Path(path).read_bytes() == raw
        return original(path, engine)

    monkeypatch.setattr(
        module, "NamedTemporaryFile", lambda **kw: NamedTemporaryFile(dir=tmp_path, **kw)
    )
    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", inspect)
    with pytest.raises(cv2.error):
        load_onnx_snapshot(raw, cv2.dnn.ENGINE_CLASSIC)
    assert len(opened) == 1 and not opened[0].exists()
    assert not list(tmp_path.iterdir())


@pytest.mark.local_examples
@pytest.mark.parametrize("engine", [cv2.dnn.ENGINE_AUTO, cv2.dnn.ENGINE_NEW])
def test_auto_and_new_stay_in_memory_without_temporary_files(monkeypatch, engine):
    import opencv_dataflow.onnx_loading as module

    def unexpected(**kwargs):
        pytest.fail("Memory import created a temporary file")

    monkeypatch.setattr(module, "NamedTemporaryFile", unexpected)
    network, evidence = load_onnx_snapshot(NEW_ONLY.read_bytes(), engine)
    assert set(network.getUnconnectedOutLayersNames()) == {"rows", "channels", "objectness", "ssd"}
    assert evidence["transport"] == "memory_buffer" and evidence["engine_argument"] == engine


@pytest.mark.local_examples
def test_native_discriminator_classic_rejects_new_only_model_wrapper_emits_nothing():
    raw = NEW_ONLY.read_bytes()
    # Native path selection is the independent discriminator: classic's importer
    # rejects this fixture; the new engine accepts it. Shape/name receipts alone
    # would not detect the OpenCV 5 vector-overload fallback bug.
    with pytest.raises(cv2.error):
        cv2.dnn.readNetFromONNX(str(NEW_ONLY), cv2.dnn.ENGINE_CLASSIC)
    assert cv2.dnn.readNetFromONNX(str(NEW_ONLY), cv2.dnn.ENGINE_NEW)
    block = DNNInference(
        parameters={"model_path": str(NEW_ONLY), "engine": "classic", "width": 64, "height": 64}
    )
    try:
        assert block.execute({"image": np.zeros((80, 160, 3), np.uint8)}) is None
        assert block.state == "error" and not block.current_emissions
        assert block._net is None and block._import is None
        block.parameters["engine"] = "new"
        result = block.execute({"image": np.zeros((80, 160, 3), np.uint8)})
        assert result is not None, block.error
        assert result.outputs["receipt"]["calls"][0]["model_import"]["transport"] == "memory_buffer"
        assert (
            result.outputs["receipt"]["calls"][0]["artifact"]["sha256"]
            == hashlib.sha256(raw).hexdigest()
        )
    finally:
        block.close()


@pytest.mark.local_examples
def test_cached_classic_import_is_reused_and_not_saved_with_project(monkeypatch):
    import opencv_dataflow.dnn_blocks as module

    original, calls = module.load_onnx_snapshot, []

    def counted(raw, engine):
        calls.append(engine)
        return original(raw, engine)

    monkeypatch.setattr(module, "load_onnx_snapshot", counted)
    graph = Graph()
    block = graph.add(
        DNNInference(
            parameters={"model_path": str(RGB), "engine": "classic", "width": 32, "height": 32}
        )
    )
    try:
        image = np.full((32, 32, 3), 100, np.uint8)
        first = block.execute({"image": image})
        assert first is not None, block.error
        net = block._net
        second = block.execute({"image": image})
        assert second is not None, block.error
        assert block._net is net and calls == [cv2.dnn.ENGINE_CLASSIC]
        receipt = second.outputs["receipt"]["calls"][0]
        assert (
            receipt["cache_reused"]
            and receipt["model_import"] == first.outputs["receipt"]["calls"][0]["model_import"]
        )
        assert "model_import" not in json.dumps(graph.to_dict())
        block.parameters["engine"] = "new"
        assert block.execute({"image": image}) is not None
        assert calls == [cv2.dnn.ENGINE_CLASSIC, cv2.dnn.ENGINE_NEW]
    finally:
        graph.close()
    assert block._net is None and block._import is None


@pytest.mark.local_examples
def test_read_snapshot_does_not_reopen_changed_source(monkeypatch, tmp_path):
    import opencv_dataflow.dnn_blocks as module

    path = tmp_path / "source.onnx"
    path.write_bytes(RGB.read_bytes())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    original = module.load_onnx_snapshot

    def change_after_hash(raw, engine):
        path.write_bytes(NEW_ONLY.read_bytes())
        assert hashlib.sha256(raw).hexdigest() == digest
        return original(raw, engine)

    monkeypatch.setattr(module, "load_onnx_snapshot", change_after_hash)
    block = DNNInference(
        parameters={"model_path": str(path), "engine": "classic", "width": 32, "height": 32}
    )
    try:
        result = block.execute({"image": np.full((32, 32, 3), 255, np.uint8)})
        assert result is not None, block.error
        np.testing.assert_allclose(
            selected_tensor(result.outputs["tensors"]), [[1, 1, 1]], atol=1e-6
        )
        assert result.outputs["receipt"]["calls"][0]["artifact"]["sha256"] == digest
        assert path.read_bytes() == NEW_ONLY.read_bytes()
    finally:
        block.close()


@pytest.mark.local_examples
def test_concurrent_imports_keep_engine_choices_independent():
    raw = RGB.read_bytes()

    def run(engine):
        net, evidence = load_onnx_snapshot(raw, engine)
        net.setInput(np.ones((1, 3, 32, 32), np.float32))
        result = net.forward().copy()
        return evidence, result

    before = dict(os.environ)
    with ThreadPoolExecutor(max_workers=2) as executor:
        classic, modern = list(executor.map(run, [cv2.dnn.ENGINE_CLASSIC, cv2.dnn.ENGINE_NEW]))
    np.testing.assert_allclose(classic[1], modern[1], atol=1e-6)
    assert classic[0]["transport"] == "temporary_snapshot_file"
    assert modern[0]["transport"] == "memory_buffer"
    assert dict(os.environ) == before


@pytest.mark.local_examples
@pytest.mark.parametrize(
    "name", ["classic_rgb", "new_rgb", "auto_rgb", "new_proposals", "classic_rejected"]
)
def test_native_projects_engine_evidence_numeric_results_and_validation(name):
    with ProjectSession.open(EXAMPLES / (name + ".json")) as session:
        assert validate_graph(session.graph)["valid"]
        session.graph.run()
        block = session.graph.blocks["infer"]
        if name == "classic_rejected":
            assert block.state == "error" and block.last_result is None
            assert not block.current_emissions and block._import is None
            assert not session.run_validation("engine")["summary"]["passed"]
            return
        assert block.state == "pass", block.error
        call = block.last_result.outputs["receipt"]["calls"][0]
        assert (
            call["model_import"]["engine_argument"]
            == {"classic_rgb": 1, "new_rgb": 2, "auto_rgb": 3, "new_proposals": 2}[name]
        )
        assert call["model_import"]["transport"] == (
            "temporary_snapshot_file" if name == "classic_rgb" else "memory_buffer"
        )
        assert not call["model_import"]["native_engine_observed"]
        if name.endswith("_rgb"):
            np.testing.assert_allclose(
                selected_tensor(block.last_result.outputs["tensors"]),
                [[220 / 255, 180 / 255, 30 / 255]],
                atol=1e-5,
            )
        else:
            assert session.graph.blocks["decode"].last_result.outputs["count"] == 3
        assert session.run_validation("engine")["summary"]["passed"]


@pytest.mark.local_examples
def test_fixture_hashes_regeneration_and_json_xml_save_as(tmp_path):
    expected = json.loads((EXAMPLES / "truth.json").read_bytes())
    for name, digest in expected["asset_sha256"].items():
        assert hashlib.sha256((EXAMPLES / "assets" / name).read_bytes()).hexdigest() == digest
    spec = importlib.util.spec_from_file_location("engine_example", EXAMPLES / "build_example.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    builder.HERE = fresh
    builder.build()
    for path in fresh.rglob("*"):
        if path.is_file():
            original = EXAMPLES / path.relative_to(fresh)
            if path.suffix == ".json" and "blocks" in json.loads(path.read_bytes()):
                # Reopened projects acquire new optional block defaults without migration.
                with ProjectSession.open(original) as session:
                    assert json.loads(path.read_bytes()) == session.graph.to_dict(), path
            else:
                assert path.read_bytes() == original.read_bytes(), path
    with pytest.raises(FileExistsError, match="preserved"):
        builder.build()
    with ProjectSession.open(EXAMPLES / "classic_rgb.json") as session:
        xml = tmp_path / "project.xml"
        definition = session.graph.to_dict()
        export_project_xml(definition, xml)
        assert read_project_xml(xml) == definition
        session.save(tmp_path / "project.json")
    with ProjectSession.open(tmp_path / "project.json") as session:
        assert session.run_validation("engine")["summary"]["passed"]
        assert session.graph.blocks["infer"].parameters["engine"] == "classic"


@pytest.mark.local_examples
def test_cold_classic_worker_process_exits_cleanly_without_accelerator_probe():
    script = r"""
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication
import cv2
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.validation_editor import ValidationWorker
def unexpected(backend):
    raise AssertionError("CPU import probed accelerators")
cv2.dnn.getAvailableTargets = unexpected
app = QApplication([])
registry = builtin_registry()
graph = Graph.load("examples/opencv tools/dnn_engines/classic_rgb.json", registry)
worker = ValidationWorker(graph.to_dict(), graph.validation_sets[0], registry, graph.project_dir)
worker.start()
assert worker.wait(20000)
assert not worker.error, worker.error
assert worker.report["summary"]["passed"]
worker.deleteLater()
graph.close()
QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
app.processEvents()
del worker, graph, app
print("classic CPU worker closed", flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", script],
        cwd=ROOT,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "classic CPU worker closed" in result.stdout
