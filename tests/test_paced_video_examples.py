"""Native timed file graphs, independent pixels/media indices and loop-reset identities."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.paced_video import MediaPacer

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/paced_video"
pytestmark = pytest.mark.local_examples


class Clock:
    def __init__(self):
        self.time = 10.0

    def now(self):
        return self.time

    def sleep(self, amount):
        self.time += amount


@pytest.mark.parametrize("case", ["native_paced", "strided_fast", "loop_reset"])
def test_saved_paced_source_pixels_media_times_geometry_tickets_and_loop_reset(case):
    truth = json.loads((BASE / "truth.json").read_bytes())
    options = truth["cases"][case]
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    source = graph.blocks["video"]
    clock = Clock()
    source._pacer = MediaPacer(clock.now, clock.sleep)
    indices = list(range(options["start"], options["end"], options["stride"]))
    old_identity = None
    try:
        for index in range(options["runs"]):
            frame_index = indices[index % len(indices)]
            cycle = index // len(indices)
            graph.run()
            assert all(b.state == "pass" for b in graph.processing_blocks()), [
                (b.id, b.error) for b in graph.processing_blocks()
            ]
            result = source.last_result
            info = result.outputs["info"]
            expected = truth["frames"][frame_index]
            assert (
                info["frame_index"] == frame_index
                and info["timestamp_ms"] == pytest.approx(expected["timestamp_ms"])
            )
            assert (
                hashlib.sha256(result.outputs["image"].tobytes()).hexdigest() == expected["sha256"]
            )
            assert info["playback"]["cycle"] == cycle
            assert result.outputs["cycle_start"] is (index % len(indices) == 0)
            assert clock.time == pytest.approx(
                10 + index * options["stride"] / truth["fps"] / options["speed"]
            )
            tracked = graph.blocks["track"].last_result
            active = tracked.outputs["report"]["active"][0]
            np.testing.assert_allclose(active["center"], expected["center"], atol=0.6)
            if result.outputs["cycle_start"]:
                assert active["track_id"] != old_identity and tracked.outputs["activations"] == 1
            else:
                assert active["track_id"] == old_identity
            old_identity = active["track_id"]
            assert tracked.payloads["tracks"]["ticket"] == result.payloads["image"]["ticket"]
            if tracked.outputs["count"]:
                assert "Overlaid 1 layers" in graph.blocks["overlay"].last_result.message
                assert np.any(
                    graph.blocks["filmstrip"].last_result.outputs["image"]
                    != result.outputs["image"]
                )
        assert source.exhausted is (case != "loop_reset")
    finally:
        graph.close()


def test_movie_hash_and_full_regeneration_keep_original_assets_and_project_bytes(tmp_path):
    truth = json.loads((BASE / "truth.json").read_bytes())
    assert (
        hashlib.sha256((BASE / "assets/moving_part.avi").read_bytes()).hexdigest()
        == truth["asset_sha256"]["moving_part.avi"]
    )
    spec = importlib.util.spec_from_file_location("paced_video_example", BASE / "build_example.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = tmp_path
    module.build()
    for path in [
        BASE / "assets/moving_part.avi",
        BASE / "truth.json",
        *(BASE / (case + ".json") for case in truth["cases"]),
    ]:
        assert (tmp_path / path.relative_to(BASE)).read_bytes() == path.read_bytes(), path
