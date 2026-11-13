import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.pointclouds import PointCloud


@pytest.mark.local_examples
def test_boundary_review_clouds_preserve_rejected_retained_and_unknown_evidence(monkeypatch):
    locate = pytest.importorskip("opencv_simple_3d_locate")

    def forbidden(*args, **kwargs):
        raise AssertionError("Cached evidence examples must not run localization")

    monkeypatch.setattr(locate.Locator, "locate", forbidden)
    root = (Path(__file__).resolve().parents[1] / "examples/Simple Plugins/3d_locate"
            / "boundary_verification/pose_reviews")
    for name, digest in json.loads((root / "artifact_hashes.json").read_bytes()).items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    cases = sorted(root.glob("*/project.json"))
    assert len(cases) == 3
    for path in cases:
        report = json.loads(path.with_name("report.json").read_bytes())
        assert report["review_only"] and not report["search_executed"]
        counts = report["counts"]
        assert sum(counts.values()) == report["sampled_boundary_pairs"] <= 8192
        assert counts["unknown"] > 0
        assert report["distinct_support_upper_bound"] == pytest.approx(
            1 - counts["continued"] / report["sampled_boundary_pairs"])
        rejected = path.parent.name == "rejected_false_pose"
        assert report["limit_exceeded"] == rejected
        assert report["runtime_accepted"] != rejected
        with np.load(path.with_name("boundary_pairs.npz")) as data:
            assert np.isnan(data["residual"][data["states"] == 0]).all()
        graph = Graph.load(path, builtin_registry())
        try:
            graph.run()
            block = graph.blocks["boundary"]
            assert not block.error and block.preview_output == "cloud"
            cloud = block.last_result.outputs["cloud"]
            assert isinstance(cloud, PointCloud) and cloud.frame == "camera" and cloud.units == "mm"
            assert cloud.colors.shape == cloud.points.shape
            assert np.any(np.all(cloud.colors == [255, 170, 35], axis=1))
        finally:
            graph.close()
