"""Recorded bounded video groups, downstream per-frame inspection and fixed asset hashes."""

import hashlib
import json
import runpy
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import format_image
from opencv_dataflow.tickets import ticket_key

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/video_buffer"
TRUTH = json.loads((BASE / "truth.json").read_bytes())


@pytest.mark.parametrize("case", TRUTH["cases"])
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_buffer_projects_and_per_frame_inspection(case, profile, tmp_path):
    graph = Graph.load(BASE / f"{case}.json", builtin_registry())
    try:
        assert validate_graph(graph)["valid"]
        graph.save(tmp_path / "project.json")
        graph.set_run_profile(profile)
        graph.run()
        block = graph.blocks["decode"]
        if case in ("read_limit", "decoded_limit", "malformed"):
            assert block.state == "error" and block.error_kind == "invalid", block.error
            assert not block.current_emissions
            return
        assert block.state == "pass", block.error
        result = block.last_result
        indexes = [1, 3, 5, 7] if case == "selected_frames" else list(range(8))
        assert result.outputs["info"]["selected_indexes"] == indexes
        assert result.outputs["info"]["reader"]["closed"]
        assert result.outputs["info"]["selection_satisfied"] is (case != "short_clip")
        tickets = set()
        for index, packet in zip(indexes, result.outputs["frames"].items, strict=True):
            expected = cv2.imread(str(BASE / f"assets/frame_{index:02d}.png"))
            if case == "selected_frames":
                expected = format_image(expected, "rgb", "uint16")
            np.testing.assert_array_equal(packet.value, expected)
            assert packet.payload["video_frame"]["timestamp_ms"] == pytest.approx(index * 100)
            tickets.add(ticket_key(packet.payload["ticket"]))
        assert len(tickets) == len(indexes)
        if case == "group_inspection":
            parts = graph.blocks["parts"]
            assert parts.state == "pass", parts.error
            assert parts.timing.snapshot()["count"] == 8
            assert parts.last_result.outputs["count"] == 1
            assert parts.last_result.payloads["count"]["video_frame"]["frame_index"] == 7
    finally:
        graph.close()


def test_pinned_assets_pixels_and_project_regeneration_match_originals():
    paths = sorted([*BASE.glob("*.json"), *BASE.glob("assets/*")])
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    runpy.run_path(str(BASE / "build_example.py"), run_name="__main__")
    assert before == [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    for entry in TRUTH["annotations"]:
        image = cv2.imread(str(BASE / "assets" / entry["file"]))
        assert hashlib.sha256(image.tobytes()).hexdigest() == entry["bgr_sha256"]
        x, y, w, h = entry["bounds"]
        assert np.all(image[y : y + h, x : x + w] == (220, 235, 250))
    for name, expected in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / "assets" / name).read_bytes()).hexdigest() == expected
