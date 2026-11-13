"""Optional plugin discovery and public native-cloud graph replay."""

import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("opencv_simple_3d_measure.dataflow")
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.pointclouds import PointCloud

ROOT = Path(__file__).resolve().parents[1] / "examples/Simple Plugins/3d_measure"


@pytest.mark.local_examples
@pytest.mark.parametrize("relative", [
    "generated/good", "generated/raised_feature", "generated/missing_surface",
    "real_tless/frame000003", "real_tless/frame000013", "real_tless/frame000024",
])
def test_public_measurements_replay_through_native_lazy_provider(relative):
    registry = builtin_registry()
    assert "3d.simple_measure" not in registry.types
    graph = Graph.load(ROOT / relative / "project.json", registry)
    try:
        graph.run()
        block = graph.blocks["measure"]
        assert block.state in ("pass", "fail"), block.error
        result = block.last_result
        assert isinstance(result.outputs["fit"], PointCloud)
        assert result.outputs["fit"].colors.shape == result.outputs["fit"].points.shape
        assert block.preview_output == "fit"
        expected = json.loads((ROOT / relative / "expected.json").read_bytes())
        assert json.loads(json.dumps(result.outputs["measurements"])) == expected["measurements"]
        if relative.endswith("missing_surface"):
            assert block.state == "fail" and result.outputs["value"] is NO_VALUE
            assert result.outputs["count"] == 1
        elif relative.endswith("raised_feature"):
            assert block.state == "fail"
            assert result.outputs["measurements"][2]["status"] == "out_of_limits"
        else:
            assert block.state == "pass"
        block.parameters["render_graphics"] = False
        graph.run()
        assert block.last_result.outputs["fit"] is NO_VALUE
        actual = json.loads(json.dumps(block.last_result.outputs["measurements"]))
        assert actual == expected["measurements"]
    finally:
        graph.close()


@pytest.mark.local_examples
def test_public_measurement_assets_and_raw_source_bytes_remain_pinned():
    manifest = json.loads((ROOT / "sha256.json").read_bytes())
    for relative, digest in manifest.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative
    source = ROOT / "real_tless/source"
    provenance = json.loads((source / "provenance.json").read_bytes())
    assert provenance["license"] == "CC-BY-4.0"
    for relative, digest in provenance["sha256"].items():
        assert hashlib.sha256((source / relative).read_bytes()).hexdigest() == digest
    for path in ROOT.rglob("*.json"):
        assert "OpenCVSimpleTools" not in path.read_text(encoding="utf-8")
