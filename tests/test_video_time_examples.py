"""Exact media-time project selection, independent geometry and deliberate guard failure."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.tickets import ticket_key

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/video_time"
pytestmark = pytest.mark.local_examples


@pytest.mark.parametrize(
    "case", ["native_exact", "nearest", "at_or_before", "at_or_after", "error_guard"]
)
def test_saved_project_selects_known_frame_and_overlays_current_results(case):
    truth = json.loads((BASE / "truth.json").read_bytes())
    graph = Graph.load(BASE / f"{case}.json", builtin_registry())
    source = graph.blocks["video"]
    identities = set()
    try:
        for _ in range(2):
            graph.run()
            expected_index = truth["cases"][case]["index"]
            if expected_index is None:
                assert source.state == "error" and "Maximum time error" in source.error
                assert source._source_ticket is None and not any(source.current_emissions.values())
                assert graph.blocks["filmstrip"].last_result is None
                continue
            assert all(b.state == "pass" for b in graph.processing_blocks()), source.error
            result = source.last_result
            expected = truth["frames"][expected_index]
            assert (
                hashlib.sha256(result.outputs["image"].tobytes()).hexdigest() == expected["sha256"]
            )
            assert result.outputs["info"]["frame_index"] == expected_index
            assert result.outputs["info"]["timestamp_ms"] == pytest.approx(expected["timestamp_ms"])
            measured = graph.blocks["blobs"].last_result
            np.testing.assert_allclose(measured.outputs["centroids"], [expected["center"]])
            assert measured.outputs["data"]["components"][0]["area"] == expected["area"]
            assert measured.payloads["collection"]["ticket"] == result.payloads["image"]["ticket"]
            assert "Overlaid 1 layers" in graph.blocks["overlay"].last_result.message
            assert np.any(
                graph.blocks["filmstrip"].last_result.outputs["image"] != result.outputs["image"]
            )
            identities.add(ticket_key(result.outputs["ticket"]))
        if case != "error_guard":
            assert len(identities) == 2
    finally:
        graph.close()


def test_original_clip_and_complete_regeneration_match_exact_bytes(tmp_path):
    truth = json.loads((BASE / "truth.json").read_bytes())
    for name, digest in truth["asset_sha256"].items():
        assert hashlib.sha256((BASE / "assets" / name).read_bytes()).hexdigest() == digest
    spec = importlib.util.spec_from_file_location("time_seek_example", BASE / "build_example.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = tmp_path
    module.build()
    for original in [*BASE.glob("*.json"), *(BASE / "assets").glob("*.avi")]:
        assert (tmp_path / original.relative_to(BASE)).read_bytes() == original.read_bytes()
