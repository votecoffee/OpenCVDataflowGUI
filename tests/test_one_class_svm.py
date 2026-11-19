"""Native one-class parity, threshold behavior, and portable image graphs."""

import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.assistant_api import ProjectSession, make_registry
from opencv_dataflow.classifier_common import model_revision
from opencv_dataflow.one_class_svm import (
    model_bytes,
    native_predictions,
    read_one_class,
)
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import FlowGroup, ValuePacket
from opencv_dataflow.tickets import ticket_key

EXAMPLES = Path(__file__).resolve().parents[1] / "examples/opencv tools/one_class_svm"
ASSETS = EXAMPLES / "assets"
SOURCE_MODEL = ASSETS / "reference.json"
TRAIN = "ml.one_class_svm_train"
PREDICT = "ml.one_class_svm_predict"
SAMPLES = [[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3], [3, 1], [3, 2], [3, 3]]


def create(kind, **parameters):
    return make_registry().create(kind, parameters=parameters)


@pytest.mark.parametrize("nu", [0.1, 0.21])
def test_native_one_class_raw_matches_independent_kernel(nu):
    block = create(TRAIN, gamma=0.1, nu=nu)
    block.execute({"samples": SAMPLES})
    assert block.state == "pass", block.error
    model = block.last_result.outputs["model"]
    vectors = np.array(model["support_vectors"], dtype=np.float64)
    test = np.array([[2, 2.1], [1.7, 2], [8, 9], [-10, 2]], dtype=np.float32)
    delta = test[:, None, :].astype(np.float64) - vectors[None, :, :]
    reference = np.exp(-0.1 * np.square(delta).sum(axis=2)) @ model["alpha"] - model["rho"]
    native = cv2.ml.SVM_create()
    native.setType(cv2.ml.SVM_ONE_CLASS)
    native.setKernel(cv2.ml.SVM_RBF)
    native.setGamma(0.1)
    native.setNu(nu)
    native.setTermCriteria((cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 5000, 1e-6))
    assert native.train(np.array(SAMPLES, np.float32), cv2.ml.ROW_SAMPLE, np.ones(9, np.int32))
    expected = native.predict(test, flags=cv2.ml.StatModel_RAW_OUTPUT)[1].ravel()
    raw, labels = native_predictions(read_one_class(model), test)
    np.testing.assert_array_equal(raw, expected)
    np.testing.assert_allclose(raw, reference, rtol=1e-5, atol=2e-7)
    assert labels.tolist() == [True, True, False, False]


@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_training_graph_reproduces_the_portable_model(profile):
    with ProjectSession.open(EXAMPLES / "train.json") as session:
        session.graph.set_run_profile(profile)
        data = session.run(all_images=True)
        assert data["passed"] and len(data["runs"]) == 20
        result = session.graph.blocks["train"].last_result
        assert result.outputs["report"]["samples"] == 20
        assert model_bytes(result.outputs["model"]) == SOURCE_MODEL.read_bytes().replace(
            b"\r\n", b"\n"
        )


@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize(
    "name,state",
    [("normal", "pass"), ("defect", "fail"), ("tilted_good", "fail"), ("strict_cutoff", "fail")],
)
def test_reloaded_inspection_retains_ticket_original_and_margin(name, state, profile):
    truth = json.loads((EXAMPLES / "truth.json").read_bytes())["results"][name]
    with ProjectSession.open(EXAMPLES / f"{name}.json") as session:
        session.graph.set_run_profile(profile)
        summary = session.run()
        assert summary["passed"] == (state == "pass")
        block = session.graph.blocks["inspect"]
        assert block.state == state
        result = block.last_result
        assert result.outputs["report"] == {k: v for k, v in truth.items() if k != "state"}
        assert result.outputs["anomaly_score"] == -truth["raw_margin"]
        assert result.outputs["accepted"] == (state == "pass")
        source = session.graph.blocks["image"].last_result
        np.testing.assert_array_equal(result.outputs["image"], source.outputs["image"])
        assert ticket_key(result.payloads["report"]["ticket"]) == ticket_key(
            source.payloads["image"]["ticket"]
        )
        if state == "fail":
            assert result.fail_reasons[0]["code"] == "svm.anomaly"
        assert bool(result.drawings) == (profile == "debug")


def test_same_length_wrong_hog_settings_fail_before_scoring():
    with ProjectSession.open(EXAMPLES / "wrong_features.json") as session:
        assert not session.run()["passed"]
        block = session.graph.blocks["inspect"]
        assert block.state == "error" and "preprocessing differs" in block.error
        assert not block.current_emissions


def test_cutoff_equality_rejects_and_revision_follows_gate():
    trainer = create(TRAIN, gamma=0.1)
    trainer.execute({"samples": SAMPLES})
    assert trainer.state == "pass", trainer.error
    model = trainer.last_result.outputs["model"]
    block = create(PREDICT)
    block.execute({"model": model, "samples": [2, 2]})
    raw = block.last_result.outputs["report"]["raw_margin"]
    assert raw > 0
    strict = copy.deepcopy(model)
    strict["settings"]["minimum_inlier_margin"] = raw
    with pytest.raises(ValidationError, match="revision mismatch"):
        read_one_class(strict)
    strict["revision"] = model_revision(strict)
    assert strict["revision"] != model["revision"]
    block.execute({"model": strict, "samples": [2, 2]})
    assert block.state == "fail"
    assert block.last_result.outputs["report"]["native_inlier"]
    assert block.last_result.outputs["anomaly_score"] == -raw


def test_group_contract_and_required_diversity_are_enforced():
    trainer = create(TRAIN)
    samples = FlowGroup(
        tuple(
            ValuePacket(row, {"feature_contract": {"dimension": i}})
            for i, row in enumerate(SAMPLES[:3])
        ),
        ("a", "b", "c"),
    )
    trainer.execute({"samples": samples})
    assert trainer.state == "error" and "different preprocessing" in trainer.error
    trainer.execute({"samples": [[1, 2], [1, 2], [2, 1]]})
    assert trainer.state == "error" and "three distinct" in trainer.error


def test_saved_model_is_copied_with_save_as(tmp_path):
    path = tmp_path / "normal-copy.json"
    with ProjectSession.open(EXAMPLES / "normal.json") as session:
        session.save(path)
    assert (tmp_path / "assets/reference.json").read_bytes() == SOURCE_MODEL.read_bytes()
    with ProjectSession.open(path) as session:
        assert session.run()["passed"]


def test_writer_round_trip_preserves_existing_file(tmp_path):
    model = json.loads(SOURCE_MODEL.read_bytes())
    path = tmp_path / "reference.json"
    saver = create("ml.one_class_svm_save", path=str(path))
    saver.execute({"model": model})
    assert saver.state == "pass"
    raw = path.read_bytes()
    saver.execute({"model": model})
    assert saver.state == "error" and path.read_bytes() == raw
    loader = create("ml.one_class_svm_load", path=str(path))
    loader.execute({})
    assert loader.state == "pass" and loader.last_result.outputs["model"] == model
    path.write_bytes(raw.replace(b'"rho":', b'"rho":0,"rho":', 1))
    loader.execute({})
    assert loader.state == "error" and "Duplicate" in loader.error


def test_fixture_asset_hashes():
    recorded = json.loads((EXAMPLES / "truth.json").read_bytes())
    for name, sha in recorded["assets_sha256"].items():
        raw = (ASSETS / name).read_bytes()
        if name.endswith(".json"):
            raw = raw.replace(b"\r\n", b"\n")
        assert hashlib.sha256(raw).hexdigest() == sha
