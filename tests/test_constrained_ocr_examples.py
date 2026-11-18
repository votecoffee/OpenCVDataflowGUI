"""Native saved-project controls, independent probability truth and sealed asset replay."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.assistant_api import ProjectSession, validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.project_xml import export_project_xml, read_project_xml

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/constrained_ocr"
MODEL = BASE / ".local/crnn_en/model.onnx"
TRUTH = json.loads((BASE / "truth.json").read_bytes())
EXPECTED = {
    "beam_recovers": ("A", "read"),
    "greedy_blank": ("", "blank"),
    "repeat_blank": ("AA", "read"),
    "blank_dominant": ("A", "blank_dominant"),
    "unreachable_word": ("", "constraint_miss"),
    "confidence_limit": ("A", "low_confidence"),
    "lexicon_file": ("A", "read"),
    "alphabet_check": ("ABA", "alphabet_miss"),
    "lexicon_check": ("ABA", "lexicon_miss"),
    "ratio_guard": ("AA", "greedy_ratio"),
}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.mark.parametrize("case", list(TRUTH["cases"]))
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_native_outcomes_mass_source_context_and_optional_graphics(case, profile):
    if case == "native_lexicon" and not MODEL.exists():
        pytest.skip("Explicit CRNN preparation required")
    with ProjectSession.open(BASE / (case + ".json")) as session:
        assert validate_graph(session.graph)["valid"]
        session.graph.set_run_profile(profile)
        session.graph.run()
        block = session.graph.blocks["ocr"]
        result = block.last_result
        assert result is not None, block.error
        assert result.passed == TRUTH["cases"][case].get("passed", True)
        rows = result.outputs["reads"]
        if case == "native_lexicon":
            assert [r["text"] for r in rows] == TRUTH["native_texts"]
        else:
            assert (rows[0]["text"], rows[0]["status"]) == EXPECTED[case]
        if case in ("beam_recovers", "lexicon_file", "confidence_limit"):
            mass = TRUTH["alignment_mass"]["ambiguous"]["A"]
            assert np.exp(rows[0]["decoding"]["selected_log_probability"]) == pytest.approx(
                mass, abs=2e-7
            )
            assert rows[0]["confidence"] == pytest.approx(mass, abs=2e-7)
            assert rows[0]["decoding"]["greedy_text"] == ""
        if case == "repeat_blank":
            assert rows[0]["confidence"] == 1
            assert [(c["start_step"], c["stop_step"]) for c in rows[0]["characters"]] == [
                (0, 1),
                (2, 3),
            ]
        source = session.graph.blocks["source"].last_result
        np.testing.assert_array_equal(result.outputs["image"], source.outputs["image"])
        assert result.payloads["reads"]["ticket"] == source.payloads["image"]["ticket"]
        assert result.payloads["reads"]["spatial"] == source.payloads["image"]["spatial"]
        assert bool(result.drawings) == (profile == "debug")
        np.testing.assert_array_equal(
            session.graph.blocks["overlay"].last_result.outputs["image"],
            session.graph.blocks["filmstrip"].last_result.outputs["image"],
        )


def test_generator_is_independent_of_cached_helpers_and_preserves_different_assets(
    tmp_path, monkeypatch
):
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "prepare_model", SimpleNamespace(PARAMETERS={"foreign": True}))
    builder = module("constrained_builder", BASE / "build_example.py")
    builder.HERE = tmp_path
    builder.build()
    for path in [*BASE.glob("*.json"), *(BASE / "assets").iterdir()]:
        assert (tmp_path / path.relative_to(BASE)).read_bytes() == path.read_bytes()
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / "assets" / name).read_bytes()).hexdigest() == digest
    (tmp_path / "assets/probability_tnc.onnx").write_bytes(b"preserve model")
    with pytest.raises(FileExistsError, match="preserved"):
        builder.build()


def test_file_lexicon_is_copied_on_save_as_and_json_xml_replay(tmp_path):
    with ProjectSession.open(BASE / "lexicon_file.json") as session:
        session.save(tmp_path / "program.json")
        definition = session.graph.to_dict()
        export_project_xml(definition, tmp_path / "program.xml")
    assert (tmp_path / "assets/words.txt").read_bytes() == (BASE / "assets/words.txt").read_bytes()
    for graph in (
        Graph.load(tmp_path / "program.json", builtin_registry()),
        Graph.from_dict(
            read_project_xml(tmp_path / "program.xml"), builtin_registry(), project_dir=tmp_path
        ),
    ):
        try:
            graph.run()
            r = graph.blocks["ocr"].last_result
            assert r is not None and r.passed and r.outputs["text"] == "A"
        finally:
            graph.close()


@pytest.mark.skipif(not MODEL.exists(), reason="Explicit CRNN preparation required")
def test_native_preparation_is_offline_idempotent_and_sealed_save_as_reopens(tmp_path, monkeypatch):
    preparer = module("constrained_preparer", BASE / "prepare_model.py")
    monkeypatch.setattr(preparer.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network"))
    before = [p.stat().st_mtime_ns for p in (MODEL, MODEL.with_name("contract.json"))]
    assert preparer.prepare() == MODEL.parent
    assert before == [p.stat().st_mtime_ns for p in (MODEL, MODEL.with_name("contract.json"))]
    with ProjectSession.open(BASE / "native_lexicon.json") as session:
        session.save(tmp_path / "program.json")
    assert (tmp_path / ".local/crnn_en/model.onnx").read_bytes() == MODEL.read_bytes()
    with ProjectSession.open(tmp_path / "program.json") as session:
        session.graph.run()
        r = session.graph.blocks["ocr"].last_result
        assert r is not None and r.passed and r.outputs["text"] == "lot2048\npnax12\npass"
