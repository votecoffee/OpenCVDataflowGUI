"""Minimum-zone results remain explicit, portable and independent of graphics."""

import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("opencv_simple_3d_measure.flatness")
pytest.importorskip("scipy.spatial")
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.pointclouds import PointCloud

ROOT = Path(__file__).resolve().parents[1] / "examples/Simple Plugins/3d_measure/flatness"


@pytest.mark.local_examples
@pytest.mark.parametrize("case", ["generated/good", "generated/deep_feature",
                                  "generated/geometry_budget", "generated/missing_surface",
                                  "real_tless/frame000003", "real_tless/frame000013",
                                  "real_tless/frame000024"])
def test_flatness_examples_keep_all_point_evidence_and_numeric_outputs(case):
    graph = Graph.load(ROOT / case / "project.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["measure"]
        assert block.state in ("pass", "fail"), block.error
        report = block.last_result.outputs["report"]
        expected = json.loads((ROOT / case / "expected.json").read_bytes())
        cloud = block.last_result.outputs["fit"]
        assert isinstance(cloud, PointCloud) and cloud.colors is not None
        for current, saved in zip(report["measurements"], expected["measurements"]):
            assert current["status"] == saved["status"] and current["reason"] == saved["reason"]
            assert current["value"] == (None if saved["value"] is None else
                                         pytest.approx(saved["value"], abs=1e-7))
            if "flatness" in current:
                assert current["flatness"]["population"] == (
                    "all_selected_points_including_datum_fit_exclusions")
        if case.endswith(("geometry_budget", "missing_surface")):
            assert block.last_result.outputs["value"] is NO_VALUE and block.state == "fail"
        elif case.endswith("deep_feature"):
            assert block.last_result.outputs["value"] == pytest.approx(.55, abs=1e-7)
            assert block.state == "fail"
        else:
            assert block.state == "pass"
            assert cloud.metadata["zone_status"]
        block.parameters["render_graphics"] = False
        graph.run()
        assert block.last_result.outputs["fit"] is NO_VALUE
        assert block.last_result.outputs["report"] == report
    finally:
        graph.close()


@pytest.mark.local_examples
def test_flatness_recipe_v3_survives_save_as_byte_exactly(tmp_path):
    graph = Graph.load(ROOT / "generated/good/project.json", builtin_registry())
    path = tmp_path / "saved/project.json"
    path.parent.mkdir()
    try:
        digest = graph.blocks["measure"].parameters["sha256"]
        graph.save(path)
    finally:
        graph.close()
    reopened = Graph.load(path, builtin_registry())
    try:
        block = reopened.blocks["measure"]
        copied = path.parent / block.parameters["recipe_path"]
        assert hashlib.sha256(copied.read_bytes()).hexdigest() == digest
        reopened.run()
        assert block.state == "pass", block.error
        assert block.last_result.outputs["value"] == pytest.approx(.4, abs=1e-7)
    finally:
        reopened.close()
