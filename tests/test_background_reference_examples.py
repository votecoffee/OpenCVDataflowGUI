"""Saved restoration programs, fresh sessions and exact first-image foreground masks."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.background_reference import load_reference
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parents[1] / "examples/opencv tools/background_reference"
TRUTH = json.loads((HERE / "truth.json").read_bytes())


@pytest.mark.parametrize("profile", ("debug", "lean"))
@pytest.mark.parametrize("case", list(TRUTH["cases"]))
def test_saved_programs_restore_immediately_or_reject_mismatch_without_current_events(
    case, profile, tmp_path
):
    registry = builtin_registry()
    graph = Graph.load(HERE / (case + ".json"), registry)
    expected = TRUTH["cases"][case]
    graph.set_run_profile(profile)
    try:
        if expected["prepare"]:
            graph.save(tmp_path / "prepare.json")
        for index in range(expected["runs"]):
            graph.run()
            if expected["prepare"]:
                assert all(b.state == "pass" for b in graph.processing_blocks())
                record, image, digest = load_reference(tmp_path / "capture.generated.json")
                assert record["camera_id"] == TRUTH["camera_id"]
                np.testing.assert_array_equal(
                    image, graph.blocks["capture"].last_result.outputs["image"]
                )
                assert digest == graph.blocks["save"].last_result.outputs["info"]["sha256"]
                continue
            block = graph.blocks["background"]
            assert block.state == expected["state"], block.error
            if expected["state"] == "error":
                assert block.error_kind == "invalid" and not any(block.current_emissions.values())
                assert not block.memory["streams"]
                continue
            result = block.last_result
            assert result.outputs["ready"] and result.outputs["activations"] == index + 1
            reference_file = HERE / block.parameters["reference_path"]
            record, clean, _ = load_reference(reference_file)
            current = graph.blocks["capture"].last_result
            expected_mask = (
                np.uint8(np.abs(np.asarray(current.outputs["image"], float) - clean) > 80) * 255
            )
            np.testing.assert_array_equal(result.outputs["mask"], expected_mask)
            np.testing.assert_array_equal(result.outputs["background"], clean)
            assert result.payloads["mask"]["ticket"] == current.payloads["image"]["ticket"]
            assert result.payloads["mask"]["spatial"] == current.payloads["image"]["spatial"]
            assert result.outputs["status"]["background_reference"]["id"] == record["id"]
            assert graph.blocks["verify"].last_result.outputs["value"] == 0
            assert all(b.state == "pass" for b in graph.processing_blocks())
            if profile == "lean":
                assert not result.graphics and not result.drawings
        # Independent reopened jobs replay references; no native state is serialized.
        if case in ("mog2", "knn", "float_mog2"):
            reopened = Graph.from_dict(graph.to_dict(), registry, project_dir=HERE)
            try:
                reopened.run()
                again = reopened.blocks["background"].last_result
                assert again.outputs["ready"] and again.outputs["activations"] == 1
                np.testing.assert_array_equal(
                    again.outputs["mask"], graph.blocks["background"].last_result.outputs["mask"]
                )
            finally:
                reopened.close()
    finally:
        graph.close()


def test_public_assets_and_graphs_regenerate_exactly_and_pinned_json_has_no_newline_conversion(
    tmp_path,
):
    spec = importlib.util.spec_from_file_location("background_example", HERE / "build_example.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = tmp_path
    module.build()
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (HERE / path.relative_to(tmp_path)).read_bytes(), path.name
    for name, digest in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((HERE / "assets" / name).read_bytes()).hexdigest() == digest
