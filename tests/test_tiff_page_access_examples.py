"""Saved indexed TIFF projects, original pixel references and reproducible asset bytes."""

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.tickets import ticket_key

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/tiff_page_access"
TRUTH = json.loads((BASE / "truth.json").read_bytes())


@pytest.mark.parametrize("case", TRUTH["cases"])
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_indexed_tiff_projects_and_repeat_cache_behavior(case, profile, tmp_path):
    graph = Graph.load(BASE / f"{case}.json", builtin_registry())
    try:
        assert validate_graph(graph)["valid"]
        graph.save(tmp_path / "copy.json")
        graph.set_run_profile(profile)
        tickets = set()
        for run in range(2):
            graph.run()
            block = graph.blocks["page"]
            if case in ("page_outside_stack", "workspace_limit"):
                assert block.state == "error" and block.error_kind == "invalid", block.error
                assert not block.current_emissions
                continue
            assert block.state == "pass", block.error
            result = block.last_result
            tickets.add(ticket_key(result.outputs["ticket"]))
            image = result.outputs["image"]
            index = 129 if case == "converted_page" else 256
            assert result.outputs["info"]["page_index"] == index
            assert result.outputs["page_count"] == 257
            if case == "converted_page":
                y, x = np.indices((96, 128))
                rgb = np.dstack(((x + y) % 256, (y * 3) % 256, (x + 129) % 256))
                np.testing.assert_allclose(image, rgb / 255, atol=3e-8)
                assert image.dtype == np.float32
            else:
                assert (
                    hashlib.sha256(image.tobytes()).hexdigest()
                    == TRUTH["pages"][index]["bgr_sha256"]
                )
            info = result.outputs["info"]["cache"]
            assert info["hit"] is (run == 1 and case != "no_cache")
            assert info["snapshot_reads"] == 1
        if case not in ("page_outside_stack", "workspace_limit"):
            assert len(tickets) == 2
    finally:
        graph.close()


def test_wired_counter_indices_override_configuration_with_normal_data_conversion():
    registry = builtin_registry()
    graph = Graph()
    counter = graph.add(registry.create("input.counter", parameters={"start": 129, "step": -126}))
    page = graph.add(
        registry.create(
            "input.tiff_page",
            parameters={"path": str(BASE / "assets/mixed-stack.tiff"), "page_index": 256},
        )
    )
    graph.connect(counter.id, "value", page.id, "page_index")
    try:
        for index in (129, 3):
            graph.run()
            assert page.last_result is not None, page.error
            assert page.last_result.outputs["info"]["page_index"] == index
            assert (
                hashlib.sha256(page.last_result.outputs["image"].tobytes()).hexdigest()
                == TRUTH["pages"][index]["bgr_sha256"]
            )
    finally:
        graph.close()


def test_reference_formulas_asset_hash_and_project_regeneration():
    paths = sorted([*BASE.glob("*.json"), *BASE.glob("assets/*")])
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    content = runpy.run_path(str(BASE / "build_example.py"), run_name="__main__")
    assert before == [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    for entry in TRUTH["pages"]:
        assert (
            hashlib.sha256(content["page"](entry["index"]).tobytes()).hexdigest()
            == entry["bgr_sha256"]
        )
    for name, expected in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / "assets" / name).read_bytes()).hexdigest() == expected
