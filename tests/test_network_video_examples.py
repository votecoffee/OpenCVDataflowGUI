"""Pinned loopback examples: deterministic assets, current tickets and selected overlays."""

import hashlib
import importlib.util
import json
import shutil
import threading
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parents[1] / "examples/opencv tools/network_video"
pytestmark = pytest.mark.local_examples


def module(name):
    spec = importlib.util.spec_from_file_location(f"network_example_{name}", HERE / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_original_assets_and_projects_regenerate_exactly_in_a_fresh_folder(tmp_path):
    truth = json.loads((HERE / "truth.json").read_bytes())
    for name, digest in truth["asset_sha256"].items():
        assert hashlib.sha256((HERE / "assets" / name).read_bytes()).hexdigest() == digest
    builder = module("build_example")
    builder.HERE = tmp_path
    shutil.copyfile(HERE / "serve_stream.py", tmp_path / "serve_stream.py")
    builder.build()
    for original in [*HERE.glob("*.json"), *(HERE / "assets").glob("*.jpg")]:
        assert (tmp_path / original.relative_to(HERE)).read_bytes() == original.read_bytes()


@pytest.mark.parametrize(
    "name,route,runs",
    [
        ("live_stream", "/stream", 6),
        ("reconnect_reset", "/drop", 10),
    ],
)
def test_saved_native_projects_match_part_centers_reset_identity_and_final_overlays(
    name, route, runs
):
    server = module("serve_stream").make_server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    graph = Graph.load(HERE / f"{name}.json", builtin_registry())
    # The native stream is ephemeral in this test; installed CLI checks cover owned service startup.
    graph.set_program({"enabled": False})
    source = graph.blocks["video"]
    source.parameters["url"] = f"http://127.0.0.1:{server.server_port}{route}"
    identity, grid = None, None
    truth = json.loads((HERE / "truth.json").read_bytes())
    try:
        for index in range(runs):
            graph.run()
            assert all(b.state == "pass" for b in graph.processing_blocks()), [
                (b.id, b.state, b.error) for b in graph.blocks.values()
            ]
            result = source.last_result
            info = result.outputs["info"]
            expected_index = index % (4 if route == "/drop" else 8)
            assert info["frame_index"] == index
            assert info["connection_frame_index"] == expected_index
            measured = graph.blocks["blobs"].last_result.outputs["centroids"][0]
            np.testing.assert_allclose(
                measured, truth["frames"][expected_index]["center"], atol=0.05
            )
            track = graph.blocks["track"].last_result
            active = track.outputs["report"]["active"][0]
            current_grid = source.current_emissions["image"][0].payload["spatial"]
            if info["stream_start"]:
                assert active["track_id"] != identity and current_grid != grid
                assert track.outputs["activations"] == 1
            else:
                assert active["track_id"] == identity and current_grid == grid
            identity, grid = active["track_id"], current_grid
            assert (
                graph.blocks["track"].current_emissions["tracks"][0].payload["ticket"]
                == (source.current_emissions["image"][0].payload["ticket"])
            )
            assert "Overlaid 1 layers" in graph.blocks["overlay"].last_result.message
            assert graph.blocks["filmstrip"].last_result is not None
    finally:
        graph.close()
        server.stopped.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
