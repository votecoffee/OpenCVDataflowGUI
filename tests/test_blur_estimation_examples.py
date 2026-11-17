"""Native block graphs against independently integrated blur and clean image truth."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE

BASE = Path(__file__).resolve().parents[1] / "examples/custom tools/motion_blur_estimation"
CASES = [
    "horizontal",
    "diagonal",
    "vertical",
    "mild_noise",
    "highdepth",
    "masked",
    "ambiguous",
    "flat",
    "sharp",
    "heavy_noise",
    "validation_mismatch",
]
pytestmark = pytest.mark.local_examples


@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize("case", CASES)
def test_motion_estimation_and_psf_wiring_restore_only_qualified_cases(case, profile):
    truth = json.loads((BASE / "truth.json").read_bytes())
    expected = truth["cases"][case]
    graph = Graph.load(BASE / f"{case}.json", builtin_registry())
    graph.set_run_profile(profile)
    try:
        graph.run()
        block, restoration = graph.blocks["estimate"], graph.blocks["restore"]
        assert block.last_result is not None, block.error
        assert not any(b.state == "error" for b in graph.processing_blocks())
        result = block.last_result
        assert result.passed is expected.get("passed", True), result.outputs["report"]
        if not expected.get("passed", True):
            assert result.outputs["report"]["status"] == expected["status"]
            assert result.outputs["psf"] is NO_VALUE and restoration.last_result is None
            assert graph.blocks["candidate"].last_result is not None
            return
        assert (
            abs(result.outputs["length"] - expected["length"]) <= truth["length_tolerance_pixels"]
        )
        delta = abs((result.outputs["angle"] - expected["angle"] + 90) % 180 - 90)
        assert delta <= truth["angle_tolerance_degrees"]
        assert restoration.last_result is not None, restoration.error
        clean = np.asarray(graph.blocks["truth"].last_result.outputs["image"], np.float64)
        blurred = np.asarray(graph.blocks["capture"].last_result.outputs["image"], np.float64)
        restored = np.asarray(restoration.last_result.outputs["image"], np.float64)
        selection = np.ones(clean.shape[:2], bool)
        if case == "masked":
            selection[:, 256:] = False
            np.testing.assert_array_equal(restored[:, 256:], blurred[:, 256:])
        before = float(np.mean((blurred[selection] - clean[selection]) ** 2))
        after = float(np.mean((restored[selection] - clean[selection]) ** 2))
        assert after / before < truth["maximum_restored_mse_ratio"], (case, before, after)
        assert (
            result.payloads["psf"]["ticket"]
            == graph.blocks["capture"].last_result.payloads["image"]["ticket"]
        )
        assert graph.blocks["restored"].last_result is not None
        assert bool(result.drawings) is (profile == "debug")
    finally:
        graph.close()


def test_rejected_later_acquisition_cannot_reuse_a_previous_accepted_psf():
    graph = Graph.load(BASE / "horizontal.json", builtin_registry())
    try:
        graph.run()
        restoration = graph.blocks["restore"]
        initial_count = restoration.timing.count
        assert initial_count == 1
        graph.blocks["capture"].parameters["path"] = "assets/flat.png"
        graph.run()
        assert graph.blocks["estimate"].last_result.outputs["psf"] is NO_VALUE
        assert restoration.timing.count == initial_count
        assert not any(restoration.current_emissions.values())
        graph.blocks["capture"].parameters["path"] = "assets/horizontal.png"
        graph.run()
        assert restoration.timing.count == initial_count + 1
        assert restoration.current_emissions["image"]
    finally:
        graph.close()


def test_original_motion_assets_and_graphs_regenerate_without_estimator_execution(tmp_path):
    spec = importlib.util.spec_from_file_location("blur_fixtures", BASE / "build_example.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    truth = json.loads((BASE / "truth.json").read_bytes())
    names = [
        "truth.json",
        *[case + ".json" for case in CASES],
        *["assets/" + name for name in truth["asset_sha256"]],
    ]
    before = {name: (BASE / name).read_bytes() for name in names}
    module.HERE = tmp_path
    module.build()
    assert {name: (tmp_path / name).read_bytes() for name in names} == before
    assert {name: (BASE / name).read_bytes() for name in names} == before
    for name, digest in truth["asset_sha256"].items():
        assert hashlib.sha256(before["assets/" + name]).hexdigest() == digest
