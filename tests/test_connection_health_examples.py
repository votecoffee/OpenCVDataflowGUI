"""Public templates never connect implicitly; replay fixtures are explicitly trusted."""

import hashlib
import runpy
from pathlib import Path

import pytest

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.plugins import load_plugin

BASE = Path(__file__).resolve().parents[1] / "examples/industrial integration/connection_health"


@pytest.mark.parametrize("name", ["ros2", "mqtt", "ethernetip", "logix", "fanuc"])
def test_disabled_template_uses_builtins_and_never_probes(name, monkeypatch):
    graph = Graph.load(BASE / f"{name}_disabled.json", builtin_registry())
    try:
        assert validate_graph(graph)["valid"]
        block = graph.blocks["health"]
        monkeypatch.setattr(block, "probe_factory", lambda: pytest.fail("Implicit probe"))
        graph.run()
        assert not block.error and block.last_result.outputs["state"] == "disabled"
        assert "MIT" in graph.blocks["scope"].description
    finally:
        graph.close()


@pytest.mark.parametrize("state", ["ok", "warning", "alarm"])
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_offline_replay_is_explicitly_loaded_and_scheduled_without_images(state, profile):
    registry = builtin_registry()
    load_plugin(BASE / "plugin.py", registry)
    graph = Graph.load(BASE / f"offline_{state}.json", registry)
    try:
        assert validate_graph(graph)["valid"]
        graph.set_run_profile(profile)
        graph.run()
        block = graph.blocks["health"]
        assert not block.error and block.last_result.outputs["state"] == state
        assert block.last_result.passed == (state != "alarm")
        assert block.status_indicator()["state"] == state
    finally:
        graph.close()


def test_graph_regeneration_is_identical():
    paths = sorted(BASE.glob("*.json"))
    assert len(paths) == 8
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    runpy.run_path(str(BASE / "build_example.py"), run_name="__main__")
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths] == before
