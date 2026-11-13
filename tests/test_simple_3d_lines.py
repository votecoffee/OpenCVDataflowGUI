"""Portable line recipes retain numerical, missing-support and native graphic evidence."""

import json
from pathlib import Path

import pytest

pytest.importorskip("opencv_simple_3d_measure.line_fitting")
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.pointclouds import PointCloud

ROOT = Path(__file__).resolve().parents[1] / "examples/Simple Plugins/3d_measure/lines"


@pytest.mark.local_examples
@pytest.mark.parametrize("case", ["generated/good", "generated/wrong_gap", "generated/missing_target",
                                  "generated/nonparallel", "generated/shortened",
                                  "real_sections/frame000003", "real_sections/frame000013",
                                  "real_sections/frame000024"])
def test_line_projects_load_and_replay_through_the_installed_provider(case):
    graph = Graph.load(ROOT / case / "project.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["measure"]
        assert block.state in ("pass", "fail"), block.error
        report = block.last_result.outputs["report"]
        expected = json.loads((ROOT / case / "expected.json").read_bytes())
        assert json.loads(json.dumps(report["measurements"])) == expected["measurements"]
        assert report["lines"] == expected["lines"]
        assert block.last_result.outputs["lines"] == report["lines"]
        assert isinstance(block.last_result.outputs["fit"], PointCloud)
        assert block.last_result.outputs["fit"].colors is not None
        if case.endswith(("missing_target", "nonparallel", "frame000024")):
            assert block.last_result.outputs["value"] is NO_VALUE and block.state == "fail"
        if case.endswith("shortened"):
            assert report["measurements"][3]["value"] < 14.1
            assert block.state == "fail"
        block.parameters["render_graphics"] = False
        graph.run()
        assert block.last_result.outputs["fit"] is NO_VALUE
        assert block.last_result.outputs["report"] == report
    finally:
        graph.close()
