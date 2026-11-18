"""Actual native batches with visible constraints, current failures and portable evidence."""

import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.ocr_constraints import RecognizeConstrainedText
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/ocr_batch"
TRUTH = json.loads((BASE / "truth.json").read_bytes())


def tool(**changes):
    data = json.loads((BASE / "dynamic_batch.json").read_bytes())
    p = next(row["parameters"] for row in data["blocks"] if row["id"] == "ocr")
    block = RecognizeConstrainedText(parameters={**p, **changes})
    block.project_dir = BASE
    return block


def args():
    source = cv2.imread(str(BASE / "assets/stripes.png"), cv2.IMREAD_GRAYSCALE)
    context = {
        "spatial": frame([300, 180], "camera"),
        "ticket": TicketClock().issue("camera"),
        "recipe_ref": {"id": "label", "revision": 1, "digest": "1" * 64},
    }
    return {"image": ValuePacket(source, context)}


def run(arguments=None, **parameters):
    block = tool(**parameters)
    try:
        result = block.execute(args() if arguments is None else arguments)
        assert result is not None, block.error
        return result
    finally:
        block.close()


@pytest.mark.parametrize("decoder", ["CTC-greedy", "CTC-prefix-beam-search"])
def test_native_reads_keep_order_exact_sequence_scores_and_source_payload(decoder):
    arguments = args()
    result = run(
        arguments, decoder=decoder, allowed_characters="AB", lexicon=["ABA", "BA", "AAA", "BBB"]
    )
    assert result.passed and result.outputs["text"] == "ABA\nBA\nAAA\nBBB"
    for row in result.outputs["reads"]:
        assert row["decoding"]["selected_token_ids"]
        assert row["confidence"] == pytest.approx(
            np.exp(
                row["decoding"]["selected_log_probability"]
                / len(row["decoding"]["selected_token_ids"])
            )
        )
        assert row["alternatives"][0]["text"] == row["text"]
        assert row["context"]["ticket"] == arguments["image"].payload["ticket"]
        assert row["context"]["spatial"] == arguments["image"].payload["spatial"]
        assert row["characters"] and all(
            c["stop_step"] > c["start_step"] for c in row["characters"]
        )
    assert result.outputs["report"]["model"] == "constrained_ocr_report_v1"
    np.testing.assert_array_equal(result.outputs["image"], arguments["image"].value)


def test_greedy_verifies_raw_text_instead_of_deleting_disallowed_symbols():
    result = run(decoder="CTC-greedy", quadrilaterals=[TRUTH["quads"][0]], allowed_characters="A")
    row = result.outputs["reads"][0]
    assert not result.passed and row["text"] == "ABA" and row["status"] == "alphabet_miss"
    assert result.outputs["text"] == ""
    result = run(decoder="CTC-greedy", quadrilaterals=[TRUTH["quads"][0]], lexicon=["BA"])
    assert not result.passed and result.outputs["reads"][0]["status"] == "lexicon_miss"


def test_closed_lexicon_cannot_turn_a_blank_into_an_unqualified_passing_word():
    result = run(quadrilaterals=[TRUTH["quads"][3]], lexicon=["ABA", "BA"])
    row = result.outputs["reads"][0]
    assert not result.passed and row["status"] == "blank_dominant"
    assert row["text"] and row["decoding"]["greedy_text"] == ""
    assert row["decoding"]["blank_log_probability"] >= row["decoding"]["selected_log_probability"]
    assert any(reason["code"] == "ocr.blank_dominant" for reason in result.fail_reasons)


def test_probability_ratio_can_reject_a_forced_but_nonblank_alphabet_read():
    result = run(
        quadrilaterals=[TRUTH["quads"][0]], allowed_characters="A", minimum_greedy_ratio=0.1
    )
    row = result.outputs["reads"][0]
    assert not result.passed and row["status"] == "greedy_ratio"
    assert row["decoding"]["greedy_text"] == "ABA" and "B" not in row["text"]
    assert row["decoding"]["relative_to_greedy"] < 0.1


def test_unreachable_completed_word_is_an_ordinary_current_failure():
    result = run(quadrilaterals=[TRUTH["quads"][0]], lexicon=["ABABABAB"])
    assert not result.passed
    assert result.outputs["reads"][0]["status"] == "constraint_miss"
    assert result.outputs["reads"][0]["alternatives"] == []
    assert result.outputs["text"] == "" and result.outputs["count"] == 0


def test_file_lexicon_is_pinned_and_changes_are_not_silently_consumed(tmp_path):
    path = tmp_path / "words.txt"
    raw = b"ABA\nBA\nAAA\nBBB\n"
    path.write_bytes(raw)
    parameters = {"lexicon_path": str(path), "lexicon_sha256": hashlib.sha256(raw).hexdigest()}
    assert run(**parameters).passed
    path.write_bytes(b"ABA\n")
    block = tool(**parameters)
    assert block.execute(args()) is None and "digest" in block.error
    assert not block.current_emissions and block._net is None
    path.write_bytes(raw)
    assert block.execute(args()) is not None
    block.close()


@pytest.mark.parametrize(
    "parameters",
    [
        {"beam_width": 0},
        {"maximum_search_expansions": 1},
        {"maximum_candidate_prefixes": 1},
        {"maximum_alignment_elements": 1},
        {"maximum_lexicon_characters": 1, "lexicon": ["ABA"]},
        {"lexicon": ["ABA", "ABA"]},
        {"allowed_characters": "A", "lexicon": ["AB"]},
        {"lexicon_path": "unknown.txt", "lexicon_sha256": ""},
    ],
)
def test_bad_controls_emit_nothing_and_recover(parameters):
    block = tool(**parameters)
    assert block.execute(args()) is None
    assert not block.current_emissions
    valid = tool()
    block.parameters = copy.deepcopy(valid.parameters)
    valid.close()
    assert block.execute(args()) is not None, block.error
    block.close()


def test_empty_roi_does_not_read_lexicon_or_load_model():
    arguments = args()
    arguments["mask"] = np.zeros((180, 300), np.uint8)
    block = tool(model_path="missing.onnx", lexicon_path="missing.txt", lexicon_sha256="1" * 64)
    result = block.execute(arguments)
    assert result is not None and not result.passed
    assert not result.outputs["report"]["batch"]["native_calls"] and block._net is None
    block.close()
