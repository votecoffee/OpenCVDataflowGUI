"""Qualify HOG mining from actual background windows and independent holdouts."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.assistant_api import ProjectSession, make_registry, validate_graph
from opencv_dataflow.payloads import FlowGroup, ValuePacket

EXAMPLES = Path(__file__).resolve().parents[1] / "examples/opencv tools/hog_detector_training"
ASSETS = EXAMPLES / "assets"
TRAIN = "ml.hog_detector_train"


def read(name):
    result = cv2.imread(str(ASSETS / (name + ".png")), cv2.IMREAD_GRAYSCALE)
    assert result is not None
    return result


def group(images):
    return FlowGroup(
        tuple(ValuePacket(image, {}) for image in images), tuple(str(i) for i in range(len(images)))
    )


def crops():
    return {
        "positives": group([read(f"positive-{i}") for i in range(6)]),
        "negatives": group([read(f"negative-{i}") for i in range(6)]),
    }


def new(**parameters):
    return make_registry().create(TRAIN, parameters=parameters)


def model(name):
    return json.loads((ASSETS / f"{name}.json").read_bytes())


@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_training_replays_crop_and_background_groups_and_reproduces_models(profile):
    truth = json.loads((EXAMPLES / "truth.json").read_bytes())
    with ProjectSession.open(EXAMPLES / "train.json") as session:
        session.graph.set_run_profile(profile)
        assert validate_graph(session.graph)["valid"]
        result = session.run(all_images=True)
        assert result["passed"] and len(result["runs"]) == 6
        block = session.graph.blocks["train"]
        actual = block.last_result.outputs
        assert actual["baseline_model"] == model("baseline")
        assert actual["model"] == model("detector")
        assert actual["report"] == truth["training"]
        assert len(actual["report"]["selected_mined_negatives"]) == 20
        assert actual["report"]["mining_candidates_before"] == [46, 43]
        assert actual["report"]["mining_candidates_after"] == [0, 0]
        assert actual["baseline_model"]["revision"] != actual["model"]["revision"]


def test_mining_scores_match_native_search_and_independent_oriented_dot_product():
    truth = json.loads((EXAMPLES / "truth.json").read_bytes())["training"]
    coeff = np.array(model("baseline")["coefficients"], np.float32)
    native = cv2.HOGDescriptor(
        (32, 32),
        (16, 16),
        (8, 8),
        (8, 8),
        9,
        _gammaCorrection=True,
        _nlevels=1,
    )
    native.setSVMDetector(coeff)
    positive = native.compute(read("positive-0")).ravel().astype(np.float64)
    negative = native.compute(read("negative-1")).ravel().astype(np.float64)
    assert positive @ coeff[:-1] + coeff[-1] > 0
    assert negative @ coeff[:-1] + coeff[-1] < 0
    scenes = [read("mining-a"), read("mining-b")]
    scores = []
    for scene in scenes:
        locations, margins = native.detect(scene, hitThreshold=0, winStride=(8, 8))
        scores.append(
            {tuple(map(int, xy)): float(s) for xy, s in zip(locations, margins, strict=True)}
        )
    for entry in truth["selected_mined_negatives"]:
        index = entry["background"]
        point = tuple(entry["origin_xy"])
        descriptor = native.compute(scenes[index], winStride=(8, 8), locations=[point])
        assert descriptor.size == len(coeff) - 1
        margin = descriptor.ravel().astype(np.float64) @ coeff[:-1] + coeff[-1]
        assert margin == pytest.approx(entry["baseline_margin"], abs=2e-6)
        assert scores[index][point] == pytest.approx(entry["baseline_margin"], abs=2e-6)


@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize("name", ["heldout_parts", "heldout_second", "heldout_background"])
def test_heldout_full_frame_recall_and_false_positives(name, profile):
    truth = json.loads((EXAMPLES / "truth.json").read_bytes())
    expected = sorted(truth["heldout_boxes"][name])
    with ProjectSession.open(EXAMPLES / f"{name}.json") as session:
        session.graph.set_run_profile(profile)
        assert session.run()["passed"]
        actual = {}
        for key in ("before", "after"):
            results = session.graph.blocks[key].last_result
            actual[key] = sorted(
                [v["geometry"]["bbox"] for v in results.outputs["collection"]["items"]]
            )
        assert actual["after"] == expected
        assert len(actual["before"]) == len(expected) + 1
        assert set(map(tuple, expected)).issubset(map(tuple, actual["before"]))


@pytest.mark.parametrize("backgrounds", [False, True])
def test_optional_mining_and_zero_candidates_keep_baseline(backgrounds):
    block = new(mining_threshold=1000)
    inputs = crops()
    if backgrounds:
        inputs["backgrounds"] = read("mining-a")
    block.execute(inputs)
    assert block.state == "pass", block.error
    output = block.last_result.outputs
    assert output["baseline_model"] == output["model"]
    report = output["report"]
    assert report["selected_mined_negatives"] == []
    assert report["mining_candidates_before"] == ([0] if backgrounds else [])


@pytest.mark.parametrize(
    "key,image,message",
    [
        ("backgrounds", "positive-1", "Duplicate images"),
        ("positives", "query-a", "exact HOG window"),
        ("negatives", "positive-1", "Duplicate images"),
    ],
)
def test_invalid_or_overlapping_split_stops_before_native_training(
    monkeypatch, key, image, message
):
    block = new()
    inputs = crops()
    inputs[key] = read(image) if key == "backgrounds" else group([read(image), read("negative-0")])
    monkeypatch.setattr(
        cv2.ml, "SVM_create", lambda: pytest.fail("Invalid data reached native fit")
    )
    block.execute(inputs)
    assert block.state == "error"
    assert message in block.error
    assert not block.current_emissions


def test_mining_budget_is_checked_before_either_native_fit(monkeypatch):
    block = new(maximum_mining_windows=1)
    monkeypatch.setattr(
        cv2.ml, "SVM_create", lambda: pytest.fail("Mining budget reached native fit")
    )
    block.execute({**crops(), "backgrounds": read("mining-a")})
    assert block.state == "error" and "window allowance" in block.error


def test_saved_project_copies_both_models_and_replays(tmp_path):
    path = tmp_path / "comparison.json"
    with ProjectSession.open(EXAMPLES / "heldout_parts.json") as session:
        session.save(path)
    for name in ("baseline.json", "detector.json"):
        assert (tmp_path / "assets" / name).read_bytes() == (ASSETS / name).read_bytes()
    with ProjectSession.open(path) as session:
        assert session.run()["passed"]
        assert session.graph.blocks["after"].last_result.outputs["count"] == 2
