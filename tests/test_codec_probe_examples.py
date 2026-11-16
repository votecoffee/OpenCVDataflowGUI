"""Saved preflight projects retain counts/negative diagnostics and their input assets."""

import hashlib
import json
import runpy
from pathlib import Path

import pytest

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/codec_probe"
TRUTH = json.loads((BASE / "truth.json").read_bytes())


@pytest.mark.parametrize("case", TRUTH["cases"])
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_file_and_encoder_preflights(case, profile, tmp_path):
    graph = Graph.load(BASE / f"{case}.json", builtin_registry())
    try:
        assert validate_graph(graph)["valid"]
        graph.set_run_profile(profile)
        graph.run()
        failed = case in ("page_limit", "invalid_header", "unsupported_encoder")
        for block in graph.processing_blocks():
            result = block.last_result
            assert result is not None, block.error
            assert result.passed is not failed and block.state == ("fail" if failed else "pass")
            assert result.outputs["info"]["pixel_decode_verified"] is False
            if case == "file_inventory":
                assert result.outputs["page_count"] == TRUTH["file_counts"][block.id]
                assert result.outputs["reader_available"] is True
                assert result.outputs["writer_available"] is NO_VALUE
            elif case == "encoder_support":
                assert result.outputs["writer_available"] is True
                assert (
                    result.outputs["page_count"] is result.outputs["reader_available"] is NO_VALUE
                )
            elif case == "page_limit":
                assert result.outputs["page_count"] == 3
                assert result.fail_reasons[0]["code"] == "image_page_count_out_of_range"
            elif case == "invalid_header":
                assert (
                    result.outputs["reader_available"] is True and result.outputs["page_count"] == 0
                )
                assert result.fail_reasons[0]["code"] == "image_header_unreadable"
            else:
                assert result.outputs["writer_available"] is False
                assert result.fail_reasons[0]["code"] == "image_writer_unavailable"
        assert "MIT" in graph.blocks["scope"].description
        graph.save(tmp_path / "saved.xml")
        restored = Graph.load(tmp_path / "saved.xml", builtin_registry())
        try:
            restored.run()
            assert all(
                b.last_result and b.last_result.passed is not failed
                for b in restored.processing_blocks()
            )
        finally:
            restored.close()
    finally:
        graph.close()


def test_source_hashes_and_project_regeneration_remain_identical():
    paths = sorted([*BASE.glob("*.json"), *BASE.glob("assets/*")])
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    runpy.run_path(str(BASE / "build_example.py"), run_name="__main__")
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / "assets" / name).read_bytes()).hexdigest() == digest
