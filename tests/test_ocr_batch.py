"""Independent CTC confidence, real native batch rows, provenance and bounded failure checks."""

import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.images import ImageArray
from opencv_dataflow.ocr_batch import RecognizeTextBatch, greedy_ctc
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.run_profiles import PROFILES
from opencv_dataflow.tickets import TicketClock

BASE = Path(__file__).resolve().parents[1] / "examples/opencv tools/ocr_batch"
TRUTH = json.loads((BASE / "truth.json").read_bytes())


def tool(**changes):
    definition = json.loads((BASE / "dynamic_batch.json").read_bytes())
    parameters = next(b["parameters"] for b in definition["blocks"] if b["id"] == "ocr")
    block = RecognizeTextBatch(parameters={**parameters, **changes})
    block.project_dir = BASE
    return block


def args():
    image = cv2.imread(str(BASE / "assets/stripes.png"), cv2.IMREAD_GRAYSCALE)
    context = {
        "spatial": frame([300, 180], "camera"),
        "ticket": TicketClock().issue("source"),
        "recipe_ref": {"id": "label", "revision": 1, "digest": "1" * 64},
        "part_data": {"lot": "A"},
    }
    return {"image": ValuePacket(image, context)}


def execute(arguments=None, **parameters):
    block = tool(**parameters)
    result = block.execute(args() if arguments is None else arguments)
    assert result is not None, block.error
    block.close()
    return block, result


def test_greedy_path_blank_repeats_and_confidence_have_independent_reference():
    scores = np.array(
        [[0.1, 0.8, 0.1], [0.1, 0.6, 0.3], [0.9, 0.05, 0.05], [0.1, 0.7, 0.2], [0.1, 0.1, 0.8]],
        np.float32,
    )
    text, confidence, chars = greedy_ctc(
        scores, ["A", "B"], blank=0, activation="probability", maximum_steps=10
    )
    assert text == "AAB"
    assert confidence == pytest.approx((0.8 * 0.7 * 0.8) ** (1 / 3), abs=1e-7)
    assert [(c["start_step"], c["stop_step"]) for c in chars] == [(0, 2), (3, 4), (4, 5)]
    assert [c["token"] for c in chars] == ["A", "A", "B"]
    reordered = scores[:, [1, 0, 2]]
    assert (
        greedy_ctc(reordered, ["A", "B"], blank=1, activation="probability", maximum_steps=10)[0]
        == text
    )


def test_extreme_logits_ties_multicharacter_tokens_and_all_blank_are_explicit():
    values = np.array([[3e38, -3e38, -3e38], [-3e38, 3e38, -3e38], [0, 0, 0]], np.float32)
    text, confidence, chars = greedy_ctc(
        values, ["AB", "C"], blank=0, activation="softmax", maximum_steps=3
    )
    assert text == "AB" and confidence == 1 and len(chars) == 1
    text, confidence, chars = greedy_ctc(
        np.array([[1, 0, 0]], np.float32),
        ["A", "B"],
        blank=0,
        activation="probability",
        maximum_steps=1,
    )
    assert (text, confidence, chars) == ("", None, [])


@pytest.mark.parametrize("change", ["shape", "classes", "steps", "probability", "nan", "blank"])
def test_invalid_ctc_contract_is_rejected(change):
    scores = np.array([[0.1, 0.8, 0.1]], np.float32)
    tokens = ["A", "B"]
    blank = 0
    maximum = 3
    if change == "shape":
        scores = scores[None]
    elif change == "classes":
        tokens = ["A"]
    elif change == "steps":
        scores = np.repeat(scores, 4, axis=0)
    elif change == "probability":
        scores[0, 0] = 0.5
    elif change == "nan":
        scores[0, 0] = np.nan
    else:
        blank = 3
    with pytest.raises(ValidationError):
        greedy_ctc(scores, tokens, blank=blank, activation="probability", maximum_steps=maximum)


@pytest.mark.parametrize("layout", ["TNC", "NTC"])
@pytest.mark.parametrize("tail", ["variable", "pad_repeat"])
def test_native_rows_keep_order_and_padding_never_becomes_a_read(layout, tail):
    arguments = args()
    _, result = execute(
        arguments,
        ctc_layout=layout,
        model_path=f"assets/dynamic_{layout.lower()}.onnx",
        tail_policy=tail,
    )
    assert result.passed and result.outputs["text"] == "ABA\nBA\nAAA\nBBB"
    assert [r["index"] for r in result.outputs["reads"]] == [0, 1, 2, 3]
    assert [r["text"] for r in result.outputs["reads"]] == ["ABA", "BA", "AAA", "BBB"]
    calls = result.outputs["report"]["batch"]["native_calls"]
    assert [r["submitted_shape"][0] for r in calls] == ([3, 1] if tail == "variable" else [3, 3])
    assert result.outputs["report"]["batch"]["padding_rows"] == (0 if tail == "variable" else 2)
    for row in result.outputs["reads"]:
        assert row["context"]["ticket"] == arguments["image"].payload["ticket"]
        assert row["context"]["spatial"] == arguments["image"].payload["spatial"]
        assert row["context"]["part_data"] == {"lot": "A"}
        assert row["crop_context"]["spatial"]["image_size"] == [5, 3]
    np.testing.assert_array_equal(result.outputs["image"], arguments["image"].value)
    assert result.payloads["reads"]["spatial"] == arguments["image"].payload["spatial"]


def test_native_batch_matches_each_independent_single_crop():
    _, batch = execute()
    single = []
    for corners in [TRUTH["quads"][i] for i in (0, 1, 2, 4)]:
        _, result = execute(quadrilaterals=[corners], batch_size=1)
        single.append(result.outputs["reads"][0])
    for together, alone in zip(batch.outputs["reads"], single, strict=True):
        assert together["text"] == alone["text"]
        assert together["confidence"] == pytest.approx(alone["confidence"], abs=1e-7)
        assert together["characters"] == alone["characters"]


def test_rgb_uint16_source_depth_and_format_survive_grayscale_inference():
    arguments = args()
    source = arguments["image"].value.astype(np.uint16) * 257
    arguments["image"].value = ImageArray(np.repeat(source[..., None], 3, axis=2), "rgb")
    _, result = execute(arguments, scale=1 / 65535, background=65535)
    assert result.outputs["text"] == "ABA\nBA\nAAA\nBBB"
    assert result.outputs["image"].dtype == np.uint16
    assert result.outputs["image"].color_format == "rgb"
    np.testing.assert_array_equal(result.outputs["image"], arguments["image"].value)


def test_native_failures_are_current_results_and_aggregate_policy_is_explicit():
    _, failed = execute(quadrilaterals=TRUTH["quads"])
    assert not failed.passed and failed.outputs["count"] == 4
    assert failed.outputs["reads"][3]["status"] == "blank"
    assert failed.outputs["reads"][3]["fail_reasons"][0]["code"] == "ocr.blank"
    assert failed.outputs["report"]["failed_region_indices"] == [3]
    _, allowed = execute(quadrilaterals=TRUTH["quads"], aggregate="minimum_reads", minimum_reads=4)
    assert allowed.passed and allowed.outputs["count"] == 4
    _, low = execute(minimum_confidence=0.99999)
    assert not low.passed and low.outputs["count"] == 0
    assert all(row["status"] == "low_confidence" for row in low.outputs["reads"])
    assert low.outputs["text"] == "" and all(row["text"] for row in low.outputs["reads"])


def test_skipped_regions_keep_indices_and_empty_selection_never_opens_a_model(monkeypatch):
    arguments = args()
    mask = np.zeros((180, 300), np.uint8)
    mask[:75] = 255
    arguments["mask"] = ValuePacket(mask, copy.deepcopy(arguments["image"].payload))
    _, result = execute(arguments)
    assert result.outputs["count"] == 2 and result.passed
    assert [r["status"] for r in result.outputs["reads"]] == ["read", "read", "skipped", "skipped"]
    block = tool(model_path="missing.onnx", vocabulary=[])
    arguments["mask"].value[:] = 0
    monkeypatch.setattr(block, "infer_tensors", lambda *a, **k: pytest.fail("inference"))
    empty = block.execute(arguments)
    assert (
        empty is not None
        and not empty.passed
        and empty.outputs["report"]["batch"]["native_calls"] == []
    )
    assert block._net is None
    block.close()


def test_external_region_wire_context_and_payload_corners_remain_source_bound():
    arguments = args()
    context = copy.deepcopy(arguments["image"].payload)
    rows = [{"id": f"word{i}", "corners": corners} for i, corners in enumerate(TRUTH["quads"][:3])]
    arguments["regions"] = ValuePacket(
        {
            "model": "ocr_regions_v1",
            "items": rows,
            "image_size": [300, 180],
            "context": copy.deepcopy(context),
        },
        context,
    )
    _, result = execute(arguments)
    assert [row["region"]["id"] for row in result.outputs["reads"]] == ["word0", "word1", "word2"]
    assert result.outputs["count"] == 3
    arguments["regions"].payload["ticket"] = TicketClock().issue("other")
    block = tool()
    assert block.execute(arguments) is None and not block.current_emissions
    assert block.execute(args()) is not None
    block.close()


def test_vocabulary_file_is_digest_checked_and_observed_once(tmp_path):
    path = tmp_path / "tokens.txt"
    path.write_bytes(b"A\nB\n")
    _, result = execute(
        vocabulary=[],
        vocabulary_path=str(path),
        vocabulary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    assert result.passed
    path.write_bytes(b"B\nA\n")
    block = tool(
        vocabulary=[],
        vocabulary_path=str(path),
        vocabulary_sha256=hashlib.sha256(b"A\nB\n").hexdigest(),
    )
    assert block.execute(args()) is None and "digest" in block.error
    block.close()


@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_run_profile_does_not_change_numeric_reads(profile):
    block = tool()
    block.run_profile = PROFILES[profile]
    result = block.execute(args())
    assert result is not None, block.error
    assert result.outputs["count"] == 4 and result.passed
    assert bool(result.drawings) == (profile == "debug")
    block.close()


@pytest.mark.parametrize(
    "setting,value",
    [
        ("ctc_layout", "NTC"),
        ("maximum_steps", 1),
        ("maximum_images", 2),
        ("maximum_source_pixels", 1),
        ("maximum_tensor_elements", 1),
        ("max_output_elements", 1),
        ("tail_policy", "reject"),
        ("input_shape", [1, 1, 3, 5]),
        ("vocabulary", ["A"]),
    ],
)
def test_bad_shapes_budgets_and_vocab_emit_nothing_then_recover(setting, value):
    block = tool(**{setting: value})
    assert block.execute(args()) is None, (setting, block.error)
    assert not block.current_emissions
    valid = tool()
    block.parameters = copy.deepcopy(valid.parameters)
    valid.close()
    assert block.execute(args()) is not None, block.error
    block.close()
