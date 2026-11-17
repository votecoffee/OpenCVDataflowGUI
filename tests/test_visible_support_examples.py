"""Native detector -> independently counted support -> clearance/downstream selection."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.outcomes import failed_input
from opencv_dataflow.payloads import packet_for

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/visible_support"
pytestmark = pytest.mark.local_examples


@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize("case", ["clear", "touching", "covered", "required_three"])
def test_native_candidate_support_and_actual_pick_are_independently_specified(profile, case):
    truth = json.loads((BASE / "truth.json").read_bytes())
    expected = truth["cases"][case]
    graph = Graph.load(BASE / f"{case}.json", builtin_registry())
    graph.set_run_profile(profile)
    try:
        graph.run()
        detector, support, pick = (graph.blocks[k] for k in ("detector", "support", "pick"))
        assert detector.state == "pass" and detector.last_result.outputs["count"] == 3, (
            detector.error
        )
        assert support.state == ("pass" if expected["expected_pass"] else "fail"), support.error
        result = support.last_result
        assert result.outputs["count"] == expected["accepted"]
        by_position = {tuple(m["bounds"][:2]): m for m in result.outputs["candidates"]}
        assert len(by_position) == 3
        for position, count in zip(
            expected["positions"], expected["matching_samples"], strict=True
        ):
            candidate = by_position[tuple(position)]
            evidence = candidate["support"]
            assert evidence["expected_samples"] == evidence["observed_samples"] == 616
            assert evidence["matching_samples"] == count
            assert evidence["visible_fraction"] == count / 616
            assert candidate["score"] >= 0.35
            assert evidence["accepted"] is (count == 616)
        if expected["selected_pose"] is None:
            assert pick.state == "inactive" and pick.input_rejected and pick.last_result is None
            assert failed_input(packet_for(result, "image"))
            assert [reason["code"] for reason in result.fail_reasons] == [
                "visible_support.too_few_candidates"
            ]
        else:
            assert pick.state == "pass", pick.error
            assert pick.last_result.outputs["pose"]["origin"] == expected["selected_pose"]
            assert pick.last_result.outputs["count"] == (
                1 if case == "touching" else expected["accepted"]
            )
        assert graph.blocks["filmstrip"].last_result is not None
        np.testing.assert_array_equal(
            result.outputs["image"], graph.blocks["source"].last_result.outputs["image"]
        )
        for port in ("image", "matches", "candidates", "count", "report"):
            assert (
                result.payloads[port]["ticket"]
                == graph.blocks["source"].last_result.outputs["ticket"]
            )
            assert result.payloads[port]["spatial"] == result.payloads["image"]["spatial"]
        assert bool(result.drawings) is (profile == "debug")
        if profile == "debug":
            assert "Overlaid 1 layers" in graph.blocks["overlay"].last_result.message
    finally:
        graph.close()


def test_original_images_hashes_and_all_saved_graphs_regenerate_exactly(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "visible_support_fixture", BASE / "build_example.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = (
        ["truth.json"]
        + [f"{case}.json" for case in module.CASES]
        + ["assets/template.png", *[f"assets/{case}.png" for case in module.CASES]]
    )
    original = {name: (BASE / name).read_bytes() for name in names}
    module.HERE = tmp_path
    module.build()
    assert {name: (tmp_path / name).read_bytes() for name in names} == original
    assert {name: (BASE / name).read_bytes() for name in names} == original
    truth = json.loads(original["truth.json"])
    for name, digest in truth["asset_sha256"].items():
        assert hashlib.sha256(original["assets/" + name]).hexdigest() == digest
