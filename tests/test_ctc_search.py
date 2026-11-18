"""Exhaustive independent alignment sums, constraint semantics and bounded search evidence."""

import copy
import itertools
import math

import numpy as np
import pytest

from opencv_dataflow import ctc_search
from opencv_dataflow.ocr_constraints import RecognizeConstrainedText, compile_plan
from opencv_dataflow.parameter_validation import ValidationError


def plan(tokens=("A", "B"), **changes):
    p = {**copy.deepcopy(RecognizeConstrainedText.defaults), **changes}
    return compile_plan(list(tokens), p, p["lexicon"], "fixture")


def exhaustive(probabilities, blank=0):
    totals = {}
    for path in itertools.product(range(probabilities.shape[1]), repeat=len(probabilities)):
        # Independent definition: merge adjacent symbols, then remove the blank.
        merged = [c for i, c in enumerate(path) if i == 0 or path[i - 1] != c]
        labels = tuple(c for c in merged if c != blank)
        weight = math.prod(float(probabilities[t, c]) for t, c in enumerate(path))
        totals[labels] = totals.get(labels, 0) + weight
    return totals


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("blank", [0, 1, 2])
def test_wide_beam_and_exact_forward_match_every_exhaustive_sequence(seed, blank):
    probabilities = np.random.default_rng(seed).dirichlet(np.ones(3), size=5)
    p = plan(blank_index=blank, beam_width=256, maximum_alternatives=16)
    logp, _ = ctc_search.log_scores(probabilities, 3, "probability", 10)
    expected = exhaustive(probabilities, blank)
    for labels, value in expected.items():
        actual = ctc_search.forward_score(logp, labels, blank, {"used": 0, "maximum": 100000})
        assert math.exp(actual) == pytest.approx(value, abs=2e-14)
    complete, evidence = ctc_search.prefix_search(logp, p)
    assert set(complete) == set(expected)
    best, alternatives, characters, info = ctc_search.decode(logp, p, ())
    winner = min(expected, key=lambda key: (-expected[key], key))
    assert best["token_ids"] == list(winner)
    assert best["probability"] == pytest.approx(expected[winner], abs=2e-14)
    assert sum(expected.values()) == pytest.approx(1, abs=1e-14)
    assert not evidence["beam_pruned"]
    assert best["confidence"] == pytest.approx(expected[winner] ** (1 / max(1, len(winner))))
    assert [c["class_id"] for c in characters] == list(winner)
    assert all(c["start_step"] < c["stop_step"] for c in characters)
    assert info["candidates_rescored"] == len(expected)


def test_prefix_can_beat_the_best_path_when_each_greedy_step_is_blank():
    values = np.array([[0.6, 0.4], [0.6, 0.4], [0.6, 0.4]])
    logp, _ = ctc_search.log_scores(values, 2, "probability", 10)
    best, _, chars, _ = ctc_search.decode(logp, plan(tokens=("A",), beam_width=16), ())
    assert best["text"] == "A"
    expected = exhaustive(values)
    assert best["probability"] == pytest.approx(expected[(1,)])
    assert best["probability"] > expected[()]
    assert chars[0]["token"] == "A"


def test_alphabet_and_lexicon_preserve_original_mass_instead_of_renormalizing():
    probabilities = np.array([[0.1, 0.2, 0.7], [0.1, 0.2, 0.7]])
    logp, _ = ctc_search.log_scores(probabilities, 3, "probability", 10)
    p = plan(allowed_characters="A", lexicon=["A"])
    best, _, _, _ = ctc_search.decode(logp, p, (2,))
    assert best["text"] == "A"
    assert best["probability"] == pytest.approx(0.2 * 0.2 + 0.1 * 0.2 + 0.2 * 0.1)
    assert best["probability"] < 0.1


def test_repeat_requires_blank_and_multicharacter_tokenizations_are_distinct():
    logp, _ = ctc_search.log_scores(
        np.array([[0.1, 0.5, 0.4], [0.1, 0.5, 0.4]]), 3, "probability", 10
    )
    p = plan(tokens=("A", "AA"), lexicon=["AA"])
    best, alternatives, _, _ = ctc_search.decode(logp, p, ())
    assert best["text"] == "AA" and best["token_ids"] == [2]
    # Two time steps cannot emit A,A: a separating blank would need a third step.
    assert ctc_search.forward_score(logp, (1, 1), 0, {"used": 0, "maximum": 1000}) == -float("inf")
    assert all(row["token_ids"] != [1, 1] for row in alternatives)


def test_zero_probabilities_unreachable_lexicon_and_extreme_logits_are_finite_public_values():
    logp, _ = ctc_search.log_scores(np.array([[1.0, 0.0, 0.0]]), 3, "probability", 10)
    best, alternatives, _, _ = ctc_search.decode(logp, plan(lexicon=["A"]), ())
    assert best is None and alternatives == []
    logp, _ = ctc_search.log_scores(np.array([[3e38, -3e38, -3e38]], np.float32), 3, "softmax", 10)
    best, _, _, _ = ctc_search.decode(logp, plan(lexicon=["A"]), ())
    assert best is not None and np.isfinite(best["log_probability"])
    assert best["probability"] == 0 and best["confidence"] == 0


def test_class_beam_length_and_alignment_limits_are_observable_and_bounded():
    values = np.full((5, 3), 1 / 3)
    logp, _ = ctc_search.log_scores(values, 3, "probability", 10)
    p = plan(beam_width=1, token_top_k=1, maximum_output_tokens=1)
    _, evidence = ctc_search.prefix_search(logp, p)
    assert evidence["beam_pruned"] and evidence["class_pruned"]
    for field in (
        "maximum_search_expansions",
        "maximum_candidate_prefixes",
        "maximum_alignment_elements",
    ):
        with pytest.raises(ValidationError):
            ctc_search.decode(logp, plan(**{field: 1}), ())


@pytest.mark.parametrize(
    "values,activation",
    [
        (np.zeros((1, 3)), "probability"),
        (np.array([[np.nan, 0, 1]]), "softmax"),
        (np.ones((2, 2, 2)), "probability"),
        (np.array([[0.1, 0.2, 0.7]]), "unknown"),
    ],
)
def test_invalid_scores_reject(values, activation):
    with pytest.raises(ValidationError):
        ctc_search.log_scores(values, 3, activation, 10)
