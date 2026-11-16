"""Saved image sequence: actual native detections against independent motion annotations."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.image_tracking import angle_delta

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/image_tracking"
pytestmark = pytest.mark.local_examples


@pytest.mark.parametrize("case", ["oriented_gaps", "expire_reacquire"])
def test_annotated_sequence_pose_uncertainty_expiry_tickets_and_overlay(case):
    truth = json.loads((BASE / "truth.json").read_bytes())
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    identity = None
    try:
        for index, expected in enumerate(truth["frames"]):
            graph.run()
            block = graph.blocks["track"]
            assert block.state == "pass", block.error
            result = block.last_result
            report = result.outputs["report"]
            assert report["timestamp_ms"] == expected["timestamp_ms"]
            if case == "expire_reacquire" and index in (14, 15):
                assert not report["active"] and result.outputs["count"] == 0
                if index == 14:
                    assert report["expired"] == [{"id": 1, "reason": "missed_frames"}]
                continue
            track = report["active"][0]
            if index == 0:
                identity = track["track_id"]
            assert (track["track_id"] == identity) is (case == "oriented_gaps" or index < 16)
            np.testing.assert_allclose(track["center"], expected["center"], atol=0.4)
            assert abs(angle_delta(track["angle_degrees"], expected["angle_degrees"], 180)) < 2
            np.testing.assert_allclose(
                track["size"], np.array([40, 18]) * expected["scale"], rtol=0.08
            )
            assert track["observed"] is expected["visible"]
            covariance = np.asarray(track["covariance"])
            assert np.linalg.eigvalsh(covariance).min() > -1e-8
            assert track["confidence"] <= 1
            if not expected["visible"]:
                assert track["confidence"] < 1
            emitted = result.outputs["tracks"]["items"]
            if emitted:
                assert (
                    emitted[0]["pose"]["target_payload"]["ticket"]
                    == graph.blocks["capture"].last_result.payloads["image"]["ticket"]
                )
                assert "Overlaid 1 layers" in graph.blocks["overlay"].last_result.message
                # The rendered final result contains graphics even in hidden-part frames.
                overlay = graph.blocks["filmstrip"].last_result.outputs["image"]
                source = graph.blocks["capture"].last_result.outputs["image"]
                assert np.any(overlay != source[..., None])
        assert graph.blocks["capture"].exhausted
    finally:
        graph.close()


def test_fixture_hashes_and_generator_reproduce_every_asset_and_project(tmp_path):
    truth = json.loads((BASE / "truth.json").read_bytes())
    for name, expected in truth["asset_sha256"].items():
        assert hashlib.sha256((BASE / "assets" / name).read_bytes()).hexdigest() == expected
    spec = importlib.util.spec_from_file_location(
        "image_tracking_example", BASE / "build_example.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = tmp_path
    module.build()
    paths = [
        *(BASE / "assets").glob("*.png"),
        BASE / "truth.json",
        *(BASE / (name + ".json") for name in truth["cases"]),
    ]
    for path in paths:
        assert (tmp_path / path.relative_to(BASE)).read_bytes() == path.read_bytes(), path
