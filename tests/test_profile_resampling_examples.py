"""Saved common pixel/physical grids, independent source coordinates and gap support."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.profile_resampling import read_profile

BASE = Path(__file__).resolve().parents[1] / "examples/custom tools/profile_resampling"
pytestmark = pytest.mark.local_examples


@pytest.mark.parametrize("case", ["pixel_common", "physical_common", "masked_decimation"])
def test_saved_profile_grids_measurements_ticket_geometry_and_null_gaps(case):
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    try:
        graph.run()
        assert all(b.state == "pass" for b in graph.processing_blocks()), [
            (b.id, b.error) for b in graph.processing_blocks()
        ]
        a, b = (graph.blocks[f"resample_{i}"].last_result for i in (0, 1))
        pa, pb = a.outputs["profile"], b.outputs["profile"]
        assert pa["distances"] == pb["distances"]
        units = "mm" if case == "physical_common" else "px"
        assert pa["units"] == pb["units"] == units
        pitch = 1 if units == "mm" else 4
        np.testing.assert_allclose(np.diff(pa["distances"]), pitch)
        points_a, points_b = np.array(a.outputs["points"]), np.array(b.outputs["points"])
        np.testing.assert_allclose(points_a[:, 0], points_b[:, 0])
        np.testing.assert_allclose(points_a[:, 1], 16)
        np.testing.assert_allclose(points_b[:, 1], 48)
        assert (
            a.payloads["profile"]["ticket"]
            == b.payloads["profile"]["ticket"]
            == graph.blocks["capture"].last_result.payloads["image"]["ticket"]
        )
        for result in (a, b):
            assert result.outputs["selected_count"] > 0 and result.graphics
            read_profile(result.outputs["profile"])
        va = np.array(pa["channels"][0]["values"], float)
        vb = np.array(pb["channels"][0]["values"], float)
        assert points_a[int(np.nanargmax(va)), 0] == 320
        if case == "masked_decimation":
            middle = (points_a[:, 0] >= 220) & (points_a[:, 0] < 240)
            assert np.isnan(va[middle]).all() and np.isnan(vb[middle]).all()
            assert np.count_nonzero(np.isnan(va)) > np.count_nonzero(np.isnan(vb))
            assert (
                a.outputs["report"]["antialias_applied"]
                and not b.outputs["report"]["antialias_applied"]
            )
            x = points_a[:, 0]
            low = (
                18000 + 4000 * np.cos(2 * np.pi * x / 64) + 12000 * np.exp(-(((x - 320) / 16) ** 2))
            )
            valid = np.isfinite(va) & (x > 60) & (x < 450)
            assert np.sqrt(np.mean((va[valid] - low[valid]) ** 2)) < 3
            assert np.sqrt(np.mean((vb[valid] - low[valid]) ** 2)) > 1000
        else:
            assert np.isfinite(va).all() and np.isfinite(vb).all()
            np.testing.assert_allclose(va, vb, rtol=0.0005, atol=0.5)
    finally:
        graph.close()


def test_generator_reproduces_all_images_mapping_projects_and_truth(tmp_path):
    truth = json.loads((BASE / "truth.json").read_bytes())
    for name, digest in truth["asset_sha256"].items():
        assert hashlib.sha256((BASE / "assets" / name).read_bytes()).hexdigest() == digest
    spec = importlib.util.spec_from_file_location(
        "profile_resampling_example", BASE / "build_example.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = tmp_path
    module.build()
    for path in [
        *(BASE / "assets").iterdir(),
        BASE / "truth.json",
        *(BASE / (case + ".json") for case in truth["cases"]),
    ]:
        assert (tmp_path / path.relative_to(BASE)).read_bytes() == path.read_bytes(), path
