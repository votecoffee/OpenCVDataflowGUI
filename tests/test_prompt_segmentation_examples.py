"""Pinned native prompted inference, independent arrays/masks and portable saved projects."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.assistant_api import ProjectSession, validate_graph
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.tickets import ticket_key

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "examples/opencv tools/prompted_segmentation"
MODEL = BASE / ".local/efficient_sam/model.onnx"
TRUTH = json.loads((BASE / "truth.json").read_bytes())
EXPECTED = {
    "coffee_point": (True, 9342, 1),
    "coffee_box": (True, 37821, 0),
    "coffee_background": (False, 12285, 2),
    "rectangle_box": (True, 1911, 1),
    "rectangle_linear": (True, 2000, 2),
    "quality_fail": (False, 9342, 1),
    "selected_out": (False, 0, 1),
    "mismatched_prompts": None,
}
native = pytest.mark.skipif(
    not MODEL.is_file(), reason="Prepare the optional pinned EfficientSAM model"
)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture(scope="module")
def references():
    assert (
        hashlib.sha256(MODEL.read_bytes()).hexdigest()
        == "4eb496e0a7259d435b49b66faf1754aa45a5c382a34558ddda9a8c6fe5915d77"
    )
    net = cv2.dnn.readNetFromONNX(str(MODEL), cv2.dnn.ENGINE_NEW)
    references = {}
    # Explicit independent source coordinates, labels and uint8-before-float resize.
    for case, path, points, labels in (
        ("coffee_point", ROOT / "examples/assets/coffee.png", [[300, 140]], [1]),
        (
            "coffee_background",
            ROOT / "examples/assets/coffee.png",
            [[300, 140], [300, 180]],
            [1, 0],
        ),
        ("rectangle_box", BASE / "assets/shapes.png", [[19, 19], [61, 71]], [2, 3]),
    ):
        image = cv2.imread(str(path))
        h, w = image.shape[:2]
        resized = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        rgb = resized[..., ::-1].astype(np.float32) / np.float32(255)
        coordinates = np.zeros((1, 1, 6, 2), np.float32)
        kinds = np.full((1, 1, 6), -1, np.float32)
        coordinates[0, 0, : len(points)] = np.asarray(points) * [1024 / w, 1024 / h]
        kinds[0, 0, : len(points)] = labels
        net.setInput(np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]), "batched_images")
        net.setInput(coordinates, "batched_point_coords")
        net.setInput(kinds, "batched_point_labels")
        masks, scores = net.forward(["output_masks", "iou_predictions"])
        references[case] = masks[0, 0].copy(), scores[0, 0].copy()
    return references


@native
@pytest.mark.parametrize("case", EXPECTED)
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_native_outcomes_counts_drawings_current_tickets_and_pixels(case, profile):
    with ProjectSession.open(BASE / (case + ".json")) as session:
        assert validate_graph(session.graph)["valid"]
        session.graph.set_run_profile(profile)
        session.graph.run()
        tool = session.graph.blocks["decode"]
        if EXPECTED[case] is None:
            assert tool.state == "error" and not tool.current_emissions
            assert "changed since inference" in tool.error
            return
        result = tool.last_result
        assert result is not None, tool.error
        assert (
            result.passed,
            result.outputs["pixel_count"],
            result.outputs["report"]["selected_index"],
        ) == EXPECTED[case]
        source = session.graph.blocks["source"].last_result
        np.testing.assert_array_equal(result.outputs["image"], source.outputs["image"])
        assert ticket_key(result.payloads["mask"]["ticket"]) == ticket_key(source.outputs["ticket"])
        assert result.payloads["mask"]["spatial"] == source.payloads["image"]["spatial"]
        if profile == "debug":
            if result.outputs["pixel_count"]:
                assert result.drawings
            view = session.graph.blocks["candidate_view"].last_output_images["value"]
            assert view.dtype == np.float32 and view.shape == (1024, 1024)
            final = session.graph.blocks["filmstrip"].last_result.outputs["image"]
            assert np.any(final != source.outputs["image"])
        else:
            assert not result.drawings
        if case.startswith("rectangle"):
            reference = cv2.imread(str(BASE / "assets/rectangle_truth.png"), 0) > 0
            mask = result.outputs["mask"] > 0
            dice = (
                2
                * np.count_nonzero(reference & mask)
                / (np.count_nonzero(reference) + np.count_nonzero(mask))
            )
            assert dice >= 0.97
        json.dumps(result.outputs["report"], allow_nan=False)


@native
@pytest.mark.parametrize("case", ["coffee_point", "coffee_background", "rectangle_box"])
def test_native_logits_quality_and_source_mask_match_independent_preprocessing_and_prompt_reference(
    case, references
):
    planes, scores = references[case]
    with ProjectSession.open(BASE / (case + ".json")) as session:
        session.graph.run()
        infer = session.graph.blocks["infer"].last_result
        for packet, expected in zip(infer.outputs["candidates"].items, planes, strict=True):
            np.testing.assert_allclose(packet.value, expected, atol=1e-5, rtol=1e-6)
        np.testing.assert_array_equal(infer.outputs["qualities"]["values"], scores)
        result = session.graph.blocks["decode"].last_result
        index = EXPECTED[case][2]
        w, h = result.outputs["image"].shape[1::-1]
        expected = cv2.resize(
            (planes[index] >= 0).astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST
        )
        np.testing.assert_array_equal(result.outputs["mask"], expected)
        if case == "coffee_background":
            assert [r["code"] for r in result.fail_reasons] == ["prompt_segmentation.prompts"]


@native
def test_warm_prompt_change_rebinds_all_slots_without_mutating_old_candidate_buffers():
    with ProjectSession.open(BASE / "coffee_point.json") as session:
        session.graph.run()
        infer = session.graph.blocks["infer"]
        old = infer.last_result.outputs["candidates"].items[0].value
        pixels = old.copy()
        first = infer.last_result.outputs["info"]["inference_revision"]
        prompts = session.graph.blocks["prompts"]
        prompts.parameters.update(points=[], point_labels=[], box=[170, 18, 245, 292])
        session.graph.blocks["decode"].parameters["minimum_quality"] = 2
        session.graph.run()
        assert infer.last_result.outputs["receipt"]["calls"][0]["cache_reused"]
        assert infer.last_result.outputs["info"]["inference_revision"] != first
        np.testing.assert_array_equal(old, pixels)
        result = session.graph.blocks["decode"].last_result
        assert not result.passed and result.outputs["pixel_count"] == 37821


@native
def test_model_contract_xml_portable_save_as_and_reopen(tmp_path):
    with ProjectSession.open(BASE / "rectangle_box.json") as session:
        definition = session.graph.to_dict()
        export_project_xml(definition, tmp_path / "program.xml")
        assert read_project_xml(tmp_path / "program.xml") == definition
        session.save(tmp_path / "program.json")
    assert (tmp_path / ".local/efficient_sam/model.onnx").read_bytes() == MODEL.read_bytes()
    for name in ("SOURCES.md", "MODEL-LICENSE.txt"):
        assert (tmp_path / name).read_bytes() == (BASE / name).read_bytes()
    with ProjectSession.open(tmp_path / "program.json") as session:
        session.graph.run()
        assert session.graph.blocks["decode"].last_result.outputs["pixel_count"] == 1911


def test_exact_regeneration_and_preservation_of_existing_projects_and_images(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(BASE))
    builder = module("prompt_builder", BASE / "build_example.py")
    builder.HERE = tmp_path
    builder.build()
    for original in [*BASE.glob("*.json"), *(BASE / "assets").glob("*.png")]:
        assert (tmp_path / original.relative_to(BASE)).read_bytes() == original.read_bytes(), (
            original
        )
    with pytest.raises(FileExistsError, match="preserved"):
        builder.build()


def test_wrong_weights_reject_without_download_or_publication(tmp_path, monkeypatch):
    preparer = module("prompt_prepare", BASE / "prepare_prompt_model.py")
    source = tmp_path / "wrong.onnx"
    source.write_bytes(b"preserve")
    monkeypatch.setattr(preparer.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network"))
    with pytest.raises(ValueError, match="pinned"):
        preparer.prepare(source, directory=tmp_path / "bundle")
    assert source.read_bytes() == b"preserve" and not (tmp_path / "bundle").exists()


@native
def test_preparation_is_idempotent_offline_and_native_engine_edit_requires_explicit_reseal(
    monkeypatch,
):
    preparer = module("prompt_prepare", BASE / "prepare_prompt_model.py")
    monkeypatch.setattr(preparer.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network"))
    paths = [MODEL, MODEL.with_name("contract.json")]
    before = [p.stat().st_mtime_ns for p in paths]
    assert preparer.prepare() == MODEL.parent
    assert before == [p.stat().st_mtime_ns for p in paths]
    with ProjectSession.open(BASE / "coffee_point.json") as session:
        session.graph.blocks["infer"].parameters["engine"] = "classic"
        session.graph.run()
        tool = session.graph.blocks["infer"]
        assert tool.state == "error" and not tool.current_emissions
        assert "pinned ONNX contract" in tool.error


@native
def test_cold_qt_worker_finishes_cpu_model_and_region_decode_then_shuts_down():
    script = r"""
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtWidgets import QApplication
import cv2
from opencv_dataflow.app import MainWindow, Runner
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
def unexpected(*args):
    raise AssertionError("CPU model probed accelerators")
cv2.dnn.getAvailableTargets = unexpected
app = QApplication([])
registry = builtin_registry()
graph = Graph.load("examples/opencv tools/prompted_segmentation/rectangle_box.json", registry)
from types import SimpleNamespace
mode = MainWindow.run_all_mode(SimpleNamespace(root_graph=graph))
assert mode == "once"
worker = Runner(graph, mode, registry)
worker.frame.connect(lambda: worker.ack.set(), Qt.ConnectionType.DirectConnection)
worker.start()
assert worker.wait(25000)
assert not worker.ended_with_error and worker.iterations == 1
result = graph.blocks["decode"].last_result
assert result.passed and result.outputs["pixel_count"] == 1911
worker.deleteLater()
graph.close()
QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
app.processEvents()
del worker, graph, app
print("prompted CPU worker closed", flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", script],
        cwd=ROOT,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=35,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "prompted CPU worker closed" in result.stdout
