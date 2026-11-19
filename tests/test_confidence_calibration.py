"""Independent calibration checks and native saved-project replay."""

import copy
import math
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.assistant_api import ProjectSession, make_registry, validate_graph
from opencv_dataflow.confidence_calibration import (
    audit_calibration,
    centered_scores,
    fit_calibration,
)
from opencv_dataflow.ml_common import tensor_record
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.tickets import ticket_key

EXAMPLES = Path(__file__).resolve().parents[1] / "examples/opencv tools/confidence_calibration"
CONTRACT = {"classes": ["a", "b"], "score_kind": "logits", "score_source": "a" * 64}
FIT = {
    **CONTRACT,
    "training_groups": ["train"],
    "minimum_temperature": 0.05,
    "maximum_temperature": 20.0,
}
GATE = {"minimum_confidence": 0.74, "minimum_margin": 0.49}
MODEL = "ml.apply_confidence_calibration"


def population(prefix, gap=4):
    return [
        {
            "id": f"{prefix}-{winner}-{i}",
            "capture_group": f"{prefix}-{winner}",
            "label": CONTRACT["classes"][winner if i < 3 else 1 - winner],
            "scores": [gap, 0] if winner == 0 else [0, gap],
        }
        for winner in (0, 1)
        for i in range(4)
    ]


def test_binary_temperature_matches_analytic_solution():
    records = population("fit")
    model, report = fit_calibration(records, **FIT)
    assert model["temperature"] == pytest.approx(4 / math.log(3), abs=1e-9)
    assert report["calibrated_nll"] == pytest.approx(-0.75 * math.log(0.75) - 0.25 * math.log(0.25))
    assert report["raw_nll"] > report["calibrated_nll"]
    assert fit_calibration(records[::-1], **FIT)[0] == model


def test_known_and_ambiguous_metrics_match_independent_reference():
    fit = fit_calibration(population("fit"), **FIT)[0]
    test = population("test")
    test.append({"id": "tie", "capture_group": "uncertain", "label": None, "scores": [0, 0]})
    report = audit_calibration(test, fit, **CONTRACT, **GATE, bins=10)
    assert report["calibrated"]["ece"] < 1e-12
    assert report["calibrated"]["brier"] == pytest.approx(2 * 0.75 * 0.25)
    assert report["raw"]["ece"] == pytest.approx(1 / (1 + math.exp(-4)) - 0.75)
    assert [report[k] for k in ("known", "known_accepted", "ambiguous", "ambiguous_accepted")] == [
        8,
        8,
        1,
        0,
    ]
    assert report["accepted_accuracy"] == pytest.approx(0.75)
    assert report["calibrated"]["bins"][7]["count"] == 8


@pytest.mark.parametrize("split", ["train", "fit"])
def test_reject_overlap_with_training_or_calibration(split):
    records = population(split)
    if split == "train":
        records[0]["capture_group"] = "train"
        with pytest.raises(ValidationError, match="overlap"):
            fit_calibration(records, **FIT)
    model = fit_calibration(population("fit"), **FIT)[0]
    with pytest.raises(ValidationError, match="overlap"):
        audit_calibration(records, model, **CONTRACT, **GATE, bins=10)


@pytest.mark.parametrize(
    "values,kind",
    [
        ([float("nan"), 1], "logits"),
        ([False, 1], "logits"),
        ([0, 1], "probabilities"),
        ([0.49, 0.4], "probabilities"),
        ([1], "logits"),
    ],
)
def test_invalid_score_vectors_raise_validation_errors(values, kind):
    with pytest.raises(ValidationError):
        centered_scores(values, 2, kind)


def test_positive_probabilities_and_shifted_logits_are_equivalent():
    logits = [999999.0, 999995.0]
    raw = [1 / (1 + math.exp(-4)), 1 / (1 + math.exp(4))]
    np.testing.assert_allclose(
        centered_scores(logits, 2, "logits"), centered_scores(raw, 2, "probabilities"), atol=1e-12
    )


@pytest.mark.parametrize(
    "option,pattern",
    [
        ({"classes": ["b", "a"]}, "class order"),
        ({"score_kind": "probabilities"}, "Score kind"),
        ({"score_source": "b" * 64}, "preprocessing differs"),
    ],
)
def test_incompatible_model_cannot_emit_deployed_label(option, pattern):
    model = fit_calibration(population("fit"), **FIT)[0]
    block = make_registry().create(MODEL, parameters={**CONTRACT, "calibration": model, **option})
    scores = [0.97, 0.03] if option.get("score_kind") == "probabilities" else [4, 0]
    block.execute({"tensors": {"scores": tensor_record(scores, 2)}})
    assert block.state == "error"
    assert pattern in block.error
    assert not block.current_emissions


def test_tampered_model_and_multiframe_tensor_are_refused():
    model = fit_calibration(population("fit"), **FIT)[0]
    block = make_registry().create(MODEL, parameters={**CONTRACT, "calibration": model})
    model["temperature"] = 1.0
    block.execute(
        {
            "calibration": model,
            "tensors": {"scores": tensor_record([4, 0], 2)},
        }
    )
    assert block.state == "error" and "revision mismatch" in block.error
    block.execute({"tensors": {"scores": tensor_record([[4, 0], [0, 4]], 4)}})
    assert block.state == "error" and "single-image" in block.error


@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize("name,state", [("known_red", "pass"), ("ambiguous", "fail")])
def test_saved_projects_run_native_dnn_and_emit_safe_ticketed_decision(name, state, profile):
    with ProjectSession.open(EXAMPLES / f"{name}.json") as session:
        assert validate_graph(session.graph)["valid"]
        session.graph.set_run_profile(profile)
        receipt = session.run()
        block = session.graph.blocks["decision"]
        result = block.last_result
        assert result is not None
        assert block.state == state and receipt["passed"] == (state == "pass")
        ticket = result.payloads["decision"]["ticket"]
        assert ticket_key(ticket) == ticket_key(
            session.graph.blocks["dnn"].last_result.payloads["tensors"]["ticket"]
        )
        if state == "pass":
            assert result.outputs["label"] == "red"
        else:
            assert result.outputs["label"] is NO_VALUE
            assert not block.current_emissions.get("label")
            assert {r["code"] for r in result.fail_reasons} == {
                "classification.low_confidence",
                "classification.small_margin",
            }


def test_dnn_receipt_detects_wrong_values_and_preprocessing_revision():
    with ProjectSession.open(EXAMPLES / "known_red.json") as session:
        assert session.run()["passed"]
        native = session.graph.blocks["dnn"]
        tensors = copy.deepcopy(native.last_result.outputs["tensors"])
        tensors["scores"]["values"].reverse()
        block = session.graph.blocks["decision"]
        block.execute({"tensors": tensors, "receipt": native.last_result.outputs["receipt"]})
        assert block.state == "error" and "receipt hash" in block.error
        native.parameters["scale"] *= 0.5
        assert not session.run()["passed"]
        assert block.state == "error" and "preprocessing differs" in block.error


@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_held_out_example_reduces_nll_ece_and_rejects_all_ambiguous_images(profile):
    with ProjectSession.open(EXAMPLES / "reliability.json") as session:
        session.graph.set_run_profile(profile)
        assert session.run()["passed"]
        results = session.graph.blocks["evaluation"].last_result.outputs
        report = results["report"]
        assert report["known"] == 40 and report["ambiguous"] == 4
        assert report["calibrated"]["nll"] < report["raw"]["nll"]
        assert report["calibrated"]["ece"] < report["raw"]["ece"]
        assert report["known_coverage"] == 0.5
        assert report["accepted_accuracy"] == 0.9
        assert report["ambiguous_accepted"] == 0
        if profile == "debug":
            assert results["plot"].shape == (410, 480, 3)
        else:
            assert results["plot"] is NO_VALUE
