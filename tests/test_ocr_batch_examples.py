"""Saved batch OCR, native CRNN, current failures and portable model/notice replay."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from opencv_dataflow.assistant_api import ProjectSession, validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.ocr import RecognizeText
from opencv_dataflow.project_xml import export_project_xml, read_project_xml

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/ocr_batch"
MODEL = BASE / ".local/crnn_en/model.onnx"
TRUTH = json.loads((BASE / "truth.json").read_bytes())
native = pytest.mark.skipif(
    not MODEL.exists(), reason="Run the example's explicit prepare_model.py"
)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize("case", list(TRUTH["cases"]))
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_projects_keep_order_confidence_and_current_failures(case, profile):
    if case == "native_serial" and not MODEL.exists():
        pytest.skip("Explicit CRNN preparation required")
    with ProjectSession.open(BASE / (case + ".json")) as session:
        assert validate_graph(session.graph)["valid"]
        session.graph.set_run_profile(profile)
        session.graph.run()
        block = session.graph.blocks["ocr"]
        r = block.last_result
        assert r is not None, block.error
        assert r.passed == TRUTH["cases"][case].get("passed", True)
        expected = (
            TRUTH["native_texts"]
            if case == "native_serial"
            else TRUTH["texts"]
            if case in ("blank_failure", "minimum_reads")
            else [TRUTH["texts"][i] for i in (0, 1, 2, 4)]
        )
        rows = r.outputs["reads"]
        assert [row["index"] for row in rows] == list(range(len(expected)))
        if case == "selected_roi":
            assert [row["text"] for row in rows] == ["ABA", "BA", "", ""]
            assert [row["status"] for row in rows] == ["read", "read", "skipped", "skipped"]
        elif case == "empty_selection":
            assert all(row["status"] == "skipped" for row in rows)
            assert not r.outputs["report"]["batch"]["native_calls"]
            assert block._net is None
        else:
            assert [row["text"] for row in rows] == expected
        assert r.outputs["count"] == sum(row["passed"] is True for row in rows)
        if case not in ("empty_selection", "confidence_gate"):
            assert r.outputs["count"] == (
                3 if case == "native_serial" else 2 if case == "selected_roi" else 4
            )
        if case == "confidence_gate":
            assert r.outputs["text"] == "" and all(
                row["status"] == "low_confidence" for row in rows
            )
        source = session.graph.blocks["source"].last_result
        np.testing.assert_array_equal(r.outputs["image"], source.outputs["image"])
        assert r.payloads["reads"]["ticket"] == source.payloads["image"]["ticket"]
        assert r.payloads["reads"]["spatial"] == source.payloads["image"]["spatial"]
        assert bool(r.drawings) == (profile == "debug" and case != "empty_selection")
        np.testing.assert_array_equal(
            session.graph.blocks["overlay"].last_result.outputs["image"],
            session.graph.blocks["filmstrip"].last_result.outputs["image"],
        )


@native
def test_native_baseline_agrees_with_the_separate_single_region_recognizer():
    image = cv2.imread(str(BASE / "assets/native_labels.png"))
    legacy = RecognizeText(
        parameters={
            "model_path": str(MODEL),
            "engine": "classic",
            "mean": [127.5],
            "scale": 1 / 127.5,
            "vocabulary": list("0123456789abcdefghijklmnopqrstuvwxyz"),
        }
    )
    regions = {"model": "ocr_regions_v1", "items": [{"corners": q} for q in TRUTH["native_quads"]]}
    try:
        result = legacy.execute({"image": image, "regions": regions})
        assert result is not None, legacy.error
        assert [row["text"] for row in result.outputs["reads"]] == TRUTH["native_texts"]
    finally:
        legacy.close()


def test_exact_regeneration_preserves_original_assets_and_existing_destinations(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(str(BASE))
    # Avoid another example's identically named authoring helper in the module cache.
    preparation = module("ocr_batch_preparation", BASE / "prepare_model.py")
    monkeypatch.setitem(sys.modules, "prepare_model", preparation)
    builder = module("ocr_batch_builder", BASE / "build_example.py")
    builder.HERE = tmp_path
    builder.build()
    for p in [*BASE.glob("*.json"), *(BASE / "assets").iterdir()]:
        assert (tmp_path / p.relative_to(BASE)).read_bytes() == p.read_bytes()
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / "assets" / name).read_bytes()).hexdigest() == digest
    (tmp_path / "assets/dynamic_tnc.onnx").write_bytes(b"preserve model")
    with pytest.raises(FileExistsError, match="preserved"):
        builder.build()


def test_wrong_pinned_weights_reject_without_network_or_publication(tmp_path, monkeypatch):
    prepare = module("ocr_prepare_bad", BASE / "prepare_model.py")
    monkeypatch.setattr(prepare.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network"))
    path = tmp_path / "wrong.onnx"
    path.write_bytes(b"keep")
    with pytest.raises(ValueError, match="pinned"):
        prepare.prepare(path, directory=tmp_path / "bundle")
    assert path.read_bytes() == b"keep" and not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("example", ["ocr_batch", "pinned_text_detection", "ssd_detection"])
def test_authoring_uses_its_own_settings_despite_a_foreign_helper(tmp_path, monkeypatch, example):
    monkeypatch.setitem(sys.modules, "prepare_model", SimpleNamespace(PARAMETERS={"foreign": True}))
    folder = BASE.parent / example
    builder = module("isolated_" + example, folder / "build_example.py")
    builder.HERE = tmp_path
    builder.build()
    for path in folder.glob("*.json"):
        assert json.loads((tmp_path / path.name).read_bytes()) == json.loads(path.read_bytes())


@native
def test_pinned_preparation_save_as_and_xml_replay(tmp_path, monkeypatch):
    prepare = module("ocr_prepare", BASE / "prepare_model.py")
    monkeypatch.setattr(prepare.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network"))
    paths = [MODEL, MODEL.with_name("contract.json")]
    before = [p.stat().st_mtime_ns for p in paths]
    assert prepare.prepare() == MODEL.parent and [p.stat().st_mtime_ns for p in paths] == before
    with ProjectSession.open(BASE / "native_serial.json") as session:
        session.save(tmp_path / "program.json")
        definition = session.graph.to_dict()
        export_project_xml(definition, tmp_path / "program.xml")
        assert read_project_xml(tmp_path / "program.xml") == definition
    assert (tmp_path / ".local/crnn_en/model.onnx").read_bytes() == MODEL.read_bytes()
    for name in ("SOURCES.md", "MODEL-LICENSE.txt"):
        assert (tmp_path / name).read_bytes() == (BASE / name).read_bytes()
    for graph in (
        Graph.load(tmp_path / "program.json", builtin_registry()),
        Graph.from_dict(definition, builtin_registry(), project_dir=tmp_path),
    ):
        try:
            graph.run()
            r = graph.blocks["ocr"].last_result
            assert r is not None and r.passed
            assert [row["text"] for row in r.outputs["reads"]] == TRUTH["native_texts"]
        finally:
            graph.close()
