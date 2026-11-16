"""Saved detector-call budgets, entry/loss/freshness and annotated pixel/ticket checks."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/detector_refresh"
pytestmark = pytest.mark.local_examples


@pytest.mark.parametrize("case", ["periodic", "every_frame", "freshness_limit"])
def test_saved_refresh_budget_entry_loss_expiry_and_final_graphics(case, monkeypatch):
    truth = json.loads((BASE / "truth.json").read_bytes())
    options = truth["cases"][case]
    period = options["period"]
    graph = Graph.load(BASE / (case + ".json"), builtin_registry())
    calls = []
    native = graph.blocks["detector"].process

    def detect(inputs):
        calls.append(1)
        return native(inputs)

    monkeypatch.setattr(graph.blocks["detector"], "process", detect)
    try:
        for index, expected in enumerate(truth["frames"]):
            graph.run()
            tracker = graph.blocks["track"]
            assert tracker.state == "pass", tracker.error
            result = tracker.last_result
            report = result.outputs["report"]
            assert len(calls) == index // period + 1 == report["detector_calls"]
            assert report["refresh"] is (index % period == 0)
            assert graph.blocks["detector"].state == ("pass" if index % period == 0 else "inactive")
            active = report["active"]
            if case == "freshness_limit" and index in (3, 7, 11, 15):
                assert not active and all(t["reason"] == "detector_age" for t in report["expired"])
                continue
            expected_objects = [
                v
                for v in expected["objects"]
                if v["label"] != "B" or index >= (5 if period == 1 else 8)
            ]
            for obj in expected_objects:
                nearest = min(
                    active, key=lambda t: np.linalg.norm(np.array(t["center"]) - obj["center"])
                )
                np.testing.assert_allclose(nearest["center"], obj["center"], atol=0.5)
                assert nearest["observed"]
                assert nearest["measurement_source"] == (
                    "detector" if index % period == 0 else "local_flow"
                )
            if case != "freshness_limit" and index in (9, 10):
                assert active[0]["measurement_source"] == "prediction"
            if case != "freshness_limit" and index == 11:
                assert report["expired"] == [{"id": 1, "reason": "missed_frames"}]
            for item in result.outputs["tracks"]["items"]:
                assert (
                    item["pose"]["target_payload"]["ticket"]
                    == graph.blocks["capture"].last_result.payloads["image"]["ticket"]
                )
                assert item["detector_age_frames"] <= options["age"]
            if result.outputs["count"]:
                assert "Overlaid 1 layers" in graph.blocks["overlay"].last_result.message
                original = graph.blocks["capture"].last_result.outputs["image"]
                composed = graph.blocks["filmstrip"].last_result.outputs["image"]
                assert np.any(composed != original[..., None])
        assert len(calls) == (4 if period == 4 else 16) and graph.blocks["capture"].exhausted
    finally:
        graph.close()


def test_generator_and_every_original_asset_are_byte_reproducible(tmp_path):
    truth = json.loads((BASE / "truth.json").read_bytes())
    for name, digest in truth["asset_sha256"].items():
        assert hashlib.sha256((BASE / "assets" / name).read_bytes()).hexdigest() == digest
    spec = importlib.util.spec_from_file_location(
        "detector_refresh_example", BASE / "build_example.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = tmp_path
    module.build()
    for path in [
        *(BASE / "assets").glob("*.png"),
        BASE / "truth.json",
        *(BASE / (name + ".json") for name in truth["cases"]),
    ]:
        assert (tmp_path / path.relative_to(BASE)).read_bytes() == path.read_bytes(), path
