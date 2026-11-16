"""Recorded MIL motion, explicit failure/reseed and reproducible original assets."""

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
from opencv_dataflow.payloads import NO_VALUE

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/mil_box_tracking"
TRUTH = json.loads((BASE / "truth.json").read_bytes())


@pytest.mark.parametrize("case", TRUTH["cases"])
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_mil_clip_localization_loss_and_work_rejection(case, profile, tmp_path):
    cv2.setRNGSeed(156)
    graph = Graph.load(BASE / f"{case}.json", builtin_registry())
    graph.save(tmp_path / "program.json")
    records = []
    try:
        assert validate_graph(graph)["valid"]
        graph.set_run_profile(profile)
        while not graph.sources_exhausted():
            assert len(records) < 10
            graph.run()
            block = graph.blocks["track"]
            if case == "work_limit":
                assert block.state == "error" and block.error_kind == "invalid", block.error
                assert not block.current_emissions and not block.memory["streams"]
                records.append(None)
                continue
            result = block.last_result
            assert result is not None, block.error
            status = result.outputs["status"]
            records.append((status, result.outputs["ready"], result.passed))
            assert result.score is None and not status["native_confidence_available"]
            if result.outputs["ready"]:
                assert result.outputs["bounds"][2:] == [32, 32]
                if case in ("translation", "region_seed"):
                    expected = TRUTH["annotations"][len(records) - 1]["bounds"]
                    error = np.linalg.norm(np.array(result.outputs["bounds"][:2]) - expected[:2])
                    assert error <= TRUTH["accepted_center_error_limit_pixels"], error
                assert result.outputs["pose"]["track_id"] == status["track_id"]
            else:
                assert result.outputs["pose"] is NO_VALUE
                assert result.outputs["bounds"] is NO_VALUE
            assert bool(result.drawings) == (profile == "debug")
            assert result.outputs["overlay"] is NO_VALUE
        if case in ("translation", "region_seed"):
            assert len(records) == 7 and all(record[2] for record in records)
            assert [record[1] for record in records] == [False, False] + [True] * 5
        elif case == "template_reacquisition":
            assert len(records) == 8
            assert records[4][0]["phase"] == "lost" and not records[4][2]
            assert records[5][0]["phase"] == "initialized"
            assert records[0][0]["track_id"] != records[5][0]["track_id"]
            assert records[-1][1] and records[-1][2]
        elif case == "wrong_region":
            assert records[1][0]["reason"] == "center_outside_region"
            assert not any(record[1] for record in records)
            assert not any(record[2] for record in records[1:])
    finally:
        graph.close()


def test_mil_assets_annotations_and_project_regeneration_are_reproducible():
    paths = sorted([*BASE.glob("*.json"), *BASE.glob("assets/*")])
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    runpy.run_path(str(BASE / "build_example.py"), run_name="__main__")
    assert before == [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    patch = cv2.imread(str(BASE / "assets/template.png"), 0)
    for annotation in TRUTH["annotations"]:
        image = cv2.imread(str(BASE / "assets" / annotation["file"]), 0)
        x, y, w, h = annotation["bounds"]
        np.testing.assert_array_equal(image[y : y + h, x : x + w], patch)
    for name, expected in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((BASE / "assets" / name).read_bytes()).hexdigest() == expected
