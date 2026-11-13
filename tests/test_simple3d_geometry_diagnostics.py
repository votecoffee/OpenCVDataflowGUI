import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import PointCloud


def test_supplied_pose_cloud_projects_preserve_rgb_units_hashes_and_never_match(monkeypatch):
    locate = pytest.importorskip("opencv_simple_3d_locate")

    def forbidden(*args, **kwargs):
        raise AssertionError("Stored supplied-pose review must never execute localization")

    monkeypatch.setattr(locate.Locator, "locate", forbidden)
    root = (Path(__file__).resolve().parents[1] / "examples/Simple Plugins/3d_locate"
            / "geometry_diagnostics")
    summary = json.loads((root / "summary.json").read_bytes())
    assert summary["cases"] == 6 and not summary["search_executed"]
    assert summary["variants"]["runtime_baseline"]["geometry_pass"] == 0
    assert summary["variants"]["triangle_visibility"]["geometry_pass"] == 1
    for name, digest in json.loads((root / "artifact_hashes.json").read_bytes()).items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    projects = sorted(root.glob("object*/project.json"))
    assert len(projects) == 6
    for path in projects:
        report = json.loads(path.with_name("report.json").read_bytes())
        assert report["scope"] == "ANNOTATED_POSE_ONLY_NOT_A_DETECTION"
        graph = Graph.load(path, builtin_registry())
        try:
            graph.run()
            for name in ("runtime_baseline", "triangle_visibility"):
                block = graph.blocks[name]
                assert not block.error and block.preview_output == "cloud"
                cloud = block.last_result.outputs["cloud"]
                assert isinstance(cloud, PointCloud)
                assert cloud.units == "mm" and cloud.frame == "camera"
                assert cloud.colors.shape == cloud.points.shape
                assert np.isfinite(cloud.points).all()
                assert len(np.unique(cloud.colors, axis=0)) >= 5
            assert "NOT A DETECTION" in graph.blocks["diagnostic_scope"].name
        finally:
            graph.close()
