import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import PointCloud


def test_real_camera_depth_reviews_preserve_maps_rgb_and_annotation_scope(monkeypatch):
    locate = pytest.importorskip("opencv_simple_3d_locate")

    def forbidden(*args, **kwargs):
        raise AssertionError("Stored camera-ray review must not run localization")

    monkeypatch.setattr(locate.Locator, "locate", forbidden)
    root = (Path(__file__).resolve().parents[1] / "examples/Simple Plugins/3d_locate"
            / "camera_depth_review")
    for name, digest in json.loads((root / "artifact_hashes.json").read_bytes()).items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    cases = sorted(root.glob("object*/project.json"))
    assert len(cases) == 6
    for path in cases:
        record = json.loads(path.with_name("report.json").read_bytes())
        assert record["annotation_only"] and not record["search_executed"]
        assert record["depth_source"] == "full_measured_frame"
        with np.load(path.with_name("camera_ray_maps.npz")) as data:
            states = data["states"]
            assert states.shape == data["residual"].shape == data["expected"].shape
            assert max(states.shape) <= 640
            for code, name in record["state_codes"].items():
                if int(code):
                    assert int((states == int(code)).sum()) == record["counts"][name]
            assert np.isnan(data["measured"][states == 4]).all()
            assert np.isnan(data["residual"][states == 4]).all()
        graph = Graph.load(path, builtin_registry())
        try:
            graph.run()
            block = graph.blocks["camera_depth"]
            assert not block.error and block.preview_output == "cloud"
            cloud = block.last_result.outputs["cloud"]
            assert isinstance(cloud, PointCloud) and cloud.frame == "camera" and cloud.units == "mm"
            assert cloud.colors.shape == cloud.points.shape
            assert "ANNOTATION ONLY" in graph.blocks["scope"].name
        finally:
            graph.close()
