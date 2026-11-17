"""Known jitter, native fitting, review pixels and exact original measured masks."""

import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/stabilization_fill"
CASES = [
    "project",
    "shifted_grids",
    "changed_donor",
    "highdepth_mean",
    "no_consistent_neighbors",
    "distance_limited",
    "native_stabilization",
]
pytestmark = pytest.mark.local_examples


@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize("case", CASES)
def test_stabilization_fill_matches_world_pixels_and_preserves_current_measurement(case, profile):
    graph = Graph.load(BASE / f"{case}.json", builtin_registry())
    graph.set_run_profile(profile)
    try:
        graph.run()
        block = graph.blocks["fill"]
        assert block.last_result is not None, [(b.id, b.error) for b in graph.processing_blocks()]
        assert not any(b.state == "error" for b in graph.processing_blocks())
        result = block.last_result
        assert result.passed is (case not in ("no_consistent_neighbors", "distance_limited"))
        report = result.outputs["report"]
        prior = graph.blocks["corrected"].last_result
        current = prior.outputs["corrected"].items[0]
        measured = prior.outputs["coverage"].items[0].value != 0
        filled = result.outputs["filled_mask"] != 0
        np.testing.assert_array_equal(result.outputs["coverage"] != 0, measured)
        np.testing.assert_array_equal(result.outputs["image"][measured], current.value[measured])
        assert not result.outputs["image"][~measured].any()
        np.testing.assert_array_equal(result.outputs["display"][measured], current.value[measured])
        assert not np.any(measured & filled)
        assert result.payloads["display"]["ticket"] == current.payload["ticket"]
        assert result.payloads["image"]["spatial"] == current.payload["spatial"]
        assert result.payloads["display"]["stabilization_border_fill"]["review_only"]
        assert report["target_ticket"] == current.payload["ticket"]
        assert graph.blocks["filmstrip"].last_result is not None
        assert bool(result.drawings) is (profile == "debug")
        if case == "no_consistent_neighbors":
            assert not filled.any()
            assert all(row["state"] == "inconsistent_overlap" for row in report["neighbors"][1:])
            return
        if case == "distance_limited":
            assert 0 < report["fill_fraction"] < 1 and not report["complete"]
        elif case in ("shifted_grids", "native_stabilization"):
            assert report["fill_fraction"] >= 0.98 and not report["complete"]
            assert (
                report["remaining_pixels"] == 6
                if case == "shifted_grids"
                else report["remaining_pixels"] <= 8
            )
        else:
            assert filled.any() and report["complete"]
        world = cv2.imread(
            str(BASE / "assets" / ("world16.png" if case == "highdepth_mean" else "world.png")),
            cv2.IMREAD_UNCHANGED,
        )
        y, x = np.indices(measured.shape)
        if case == "native_stabilization":
            # Known world crop of the first raw capture; fitted root is anchored there.
            matrix = np.asarray(current.payload["spatial"]["to_root"])
            u = np.rint(matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2] + 4).astype(int)
            v = np.rint(matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2] + 6).astype(int)
            # Border regions are an independent slow ramp; compound nearest rounding is <= 2 units.
            np.testing.assert_allclose(
                result.outputs["display"][filled], world[v, u][filled], atol=2
            )
        else:
            offset = 1 if case == "shifted_grids" else 0
            expected = world[8:120, 8 - offset : 168 - offset]
            np.testing.assert_array_equal(result.outputs["display"][filled], expected[filled])
        if case == "changed_donor":
            assert report["neighbors"][1]["changed_halo_pixels"] > 0
            assert np.any(result.outputs["donor_index"][50:58, 156:159] == 2)
        if case == "highdepth_mean":
            assert result.outputs["display"].dtype == np.uint16
            assert result.outputs["display"].max() > 255
    finally:
        graph.close()


def test_original_jitter_assets_and_sealed_projects_regenerate_exactly(tmp_path):
    spec = importlib.util.spec_from_file_location("fill_fixtures", BASE / "build_example.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    truth = json.loads((BASE / "truth.json").read_bytes())
    names = [
        "truth.json",
        *[name + ".json" for name in CASES],
        *["assets/" + name for name in truth["asset_sha256"]],
    ]
    before = {name: (BASE / name).read_bytes() for name in names}
    module.HERE = tmp_path
    module.build()
    assert {name: (tmp_path / name).read_bytes() for name in names} == before
    assert {name: (BASE / name).read_bytes() for name in names} == before
    for name, expected in truth["asset_sha256"].items():
        assert hashlib.sha256(before["assets/" + name]).hexdigest() == expected
