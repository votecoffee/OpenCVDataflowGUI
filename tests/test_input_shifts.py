"""Paired image perturbations, current observation identities and independent mask statistics."""

import copy

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.input_shifts import pack
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock


@pytest.fixture
def make():
    blocks = []

    def create(kind="ml.image_input_shift", **p):
        block = builtin_registry().create(kind, parameters=p)
        blocks.append(block)
        return block

    yield create
    for block in blocks:
        block.close()


def shifted(make, *, image=None, truth=None, **parameters):
    image = np.arange(64, dtype=np.uint8).reshape(8, 8) if image is None else image
    truth = np.zeros(image.shape[:2], np.uint8) if truth is None else truth
    return make(**parameters).execute({"image": image, "truth": truth})


def observation(make, result, *, prediction=None, passed=True, score=0.1):
    payload = result.payloads["image"]
    prediction = result.outputs["truth"] if prediction is None else prediction
    values = {
        "image": ValuePacket(result.outputs["image"], payload),
        "prediction": ValuePacket(prediction, payload),
        "truth": ValuePacket(result.outputs["truth"], payload),
        "support": ValuePacket(result.outputs["support"], payload),
        "report": ValuePacket(
            {"reference_revision": "a" * 64, "decision_revision": "b" * 64}, payload
        ),
        "passed": passed,
        "score": ValuePacket(score, payload),
    }
    block = make("ml.capture_shift_observation")
    result = block.execute(values)
    assert result is not None, block.error
    return result.outputs["observation"], values


def test_baseline_byte_identity_depth_encoding_and_known_inverse_pose(make):
    image = np.uint16(np.arange(64).reshape(8, 8) * 1000)
    baseline = shifted(make, image=image)
    assert baseline is not None
    np.testing.assert_array_equal(baseline.outputs["image"], image)
    assert baseline.outputs["image"].dtype == np.uint16
    truth = np.zeros((8, 8), np.uint8)
    truth[2:4, 2:4] = 255
    posed = shifted(make, image=image, truth=truth, variant="pose", translation_x=2)
    expected = np.zeros_like(truth)
    expected[2:4, 4:6] = 255
    np.testing.assert_array_equal(posed.outputs["truth"], expected)
    assert np.count_nonzero(posed.outputs["support"]) == 48
    assert posed.payloads["image"]["spatial"]["to_root"][0][2] == -2


def test_photometric_clipping_and_native_blur_support(make):
    image = np.full((8, 8), 200, np.uint8)
    result = shifted(make, image=image, variant="exposure", gain=2, offset=-0.1)
    assert np.all(result.outputs["image"] == 255)
    blurred = shifted(make, image=image, variant="blur", blur_kernel=3, blur_sigma=1)
    expected = cv2.GaussianBlur(
        image.astype(float) / 255, (3, 3), 1, borderType=cv2.BORDER_CONSTANT
    )
    np.testing.assert_array_equal(
        blurred.outputs["image"], np.rint(expected * 255).astype(np.uint8)
    )
    assert np.count_nonzero(blurred.outputs["support"]) == 36


def test_ninety_degree_pose_has_independent_rotated_pixel_and_truth_reference(make):
    image = np.arange(64, dtype=np.uint8).reshape(8, 8)
    truth = np.zeros((8, 8), np.uint8)
    truth[2:4, 2:4] = 255
    result = shifted(make, image=image, truth=truth, variant="rotation", angle_degrees=90)
    np.testing.assert_array_equal(result.outputs["image"], np.rot90(image))
    np.testing.assert_array_equal(result.outputs["truth"], np.rot90(truth))


@pytest.mark.parametrize(
    "p",
    [
        {"gain": 2},
        {"variant": "x", "blur_kernel": 2},
        {"variant": "x", "blur_sigma": 1},
        {"variant": "x", "scale": 0},
        {"variant": 3},
        {"expected_pass": "yes"},
        {"blur_kernel": True},
    ],
)
def test_invalid_shift_parameters(make, p):
    with pytest.raises(ValueError):
        make(**p).validate_parameters()


def test_invalid_float_and_truth_grid_ticket_are_contained(make):
    assert shifted(make, image=np.full((8, 8), 2, np.float32)) is None
    clock = TicketClock()
    block = make()
    values = {
        "image": ValuePacket(np.zeros((8, 8), np.uint8), {"ticket": clock.issue("a")}),
        "truth": ValuePacket(np.zeros((8, 8), np.uint8), {"ticket": clock.issue("b")}),
    }
    assert block.execute(values) is None and "ticket" in block.error
    values["truth"] = ValuePacket(
        np.zeros((8, 8), np.uint8), {"spatial": frame([8, 8], root_id="b")}
    )
    values["image"] = ValuePacket(
        np.zeros((8, 8), np.uint8), {"spatial": frame([8, 8], root_id="a")}
    )
    assert block.execute(values) is None and "grid" in block.error


def test_capture_current_mask_support_scores_and_retained_truth(make):
    truth = np.zeros((8, 8), np.uint8)
    truth[2:4, 2:4] = 255
    source = shifted(make, truth=truth, expected_pass=False)
    row, values = observation(make, source, passed=False, score=3.2)
    assert row["passed"] is False and row["score"] == 3.2 and row["score_kind"] == "raw"
    assert row["truth_retention"] == 1
    capture = make("ml.capture_shift_observation", score_kind="probability")
    assert capture.execute(values) is None and "score" in capture.error
    cropped = shifted(make, truth=truth, variant="cropped", expected_pass=False, translation_x=6)
    _, values = observation(make, source, passed=False)
    for key in ("image", "truth", "support"):
        values[key] = ValuePacket(cropped.outputs[key], cropped.payloads[key])
    for key in ("prediction", "score", "report"):
        values[key].payload = copy.deepcopy(cropped.payloads["image"])
    assert make("ml.capture_shift_observation").execute(values) is None


@pytest.mark.parametrize(
    "mutation",
    ["pixels", "truth_pixels", "variant", "ticket", "frame", "model_support", "no_judgment"],
)
def test_invalid_observation_associations_emit_no_record(make, mutation):
    result = shifted(make)
    _, values = observation(make, result)
    if mutation == "pixels":
        values["image"].value = values["image"].value.copy()
        values["image"].value[0, 0] += 1
    elif mutation == "truth_pixels":
        values["truth"].value = values["truth"].value.copy()
        values["truth"].value[0, 0] = 255
    elif mutation == "variant":
        values["score"].payload = copy.deepcopy(values["score"].payload)
        values["score"].payload["evaluation_shift"]["variant"] = "other"
    elif mutation == "ticket":
        clock = TicketClock()
        values["image"].payload = {**values["image"].payload, "ticket": clock.issue("a")}
        values["prediction"].payload = {**values["prediction"].payload, "ticket": clock.issue("b")}
    elif mutation == "frame":
        values["prediction"].payload = {
            **values["prediction"].payload,
            "spatial": frame([8, 8], root_id="x"),
        }
        values["image"].payload = {**values["image"].payload, "spatial": frame([8, 8], root_id="y")}
    elif mutation == "model_support":
        values["model_support"] = ValuePacket(np.zeros((8, 8), np.uint8), result.payloads["image"])
    else:
        values.pop("passed")
    block = make("ml.capture_shift_observation")
    assert block.execute(values) is None and not block.current_emissions


def test_prediction_tool_outcome_can_supply_judgment_without_fabricated_confidence(make):
    source = shifted(make)
    _, values = observation(make, source, passed=False)
    values.pop("passed")
    values["prediction"].payload = {
        **values["prediction"].payload,
        "outcome": {
            "model": "tool_outcome_v1",
            "passed": False,
            "chain_passed": False,
            "source": {"id": "inspection"},
        },
    }
    block = make("ml.capture_shift_observation")
    result = block.execute(values)
    assert result is not None and result.passed and result.outputs["observation"]["passed"] is False


def paired(make, *, defective=False, bad=False):
    truth = np.zeros((8, 8), np.uint8)
    if defective:
        truth[2:4, 2:4] = 255
    baseline = shifted(make, truth=truth, expected_pass=not defective)
    variant = shifted(
        make, truth=truth, expected_pass=not defective, variant="pose", translation_x=2
    )
    a, _ = observation(make, baseline, passed=not defective, score=0.1)
    prediction = np.zeros((8, 8), np.uint8) if bad else None
    b, _ = observation(
        make, variant, prediction=prediction, passed=defective if bad else not defective, score=0.4
    )
    return [a, b]


def test_paired_pose_mapping_truth_iou_raw_score_deltas_and_lean_metrics(make):
    rows = paired(make, defective=True)
    block = make("ml.evaluate_input_shifts", required_variants=["baseline", "pose"])
    result = block.execute(
        {"observations": FlowGroup(tuple(ValuePacket(r, {}) for r in rows), ("a", "b"))}
    )
    assert result is not None and result.passed and result.score is None
    report = result.outputs["report"]
    assert report["summary"]["truth_iou"] == 1 and report["summary"]["baseline_agreement_iou"] == 1
    assert report["records"][1]["score_delta"] == pytest.approx(0.3)
    assert (
        result.outputs["false_reject_rate"] is NO_VALUE and result.outputs["false_accept_rate"] == 0
    )
    block.parameters["maximum_score_drop"] = 0.25
    assert block.execute({"observations": rows}).passed is False
    block.parameters["score_direction"] = "higher"
    block.set_graphics_mode("off")
    result = block.execute({"observations": rows})
    assert result.passed and result.outputs["plot"] is NO_VALUE
    assert report["generated_shifts_are_new_captures"] is False


@pytest.mark.parametrize("defective", [False, True])
def test_false_rejection_or_acceptance_and_localization_failures(make, defective):
    rows = paired(make, defective=defective, bad=True)
    result = make("ml.evaluate_input_shifts", required_variants=["baseline", "pose"]).execute(
        {"observations": rows}
    )
    assert result is not None and not result.passed
    summary = result.outputs["report"]["summary"]
    assert summary["false_accept_rate" if defective else "false_reject_rate"] == 0.5
    if defective:
        assert summary["truth_iou"] == 0.5 and summary["baseline_agreement_iou"] == 0.5


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "identity", "baseline", "label", "mask", "budget", "support"],
)
def test_reject_incomplete_mixed_or_malformed_pairs(make, mutation):
    rows = paired(make, defective=True)
    p = {"required_variants": ["baseline", "pose"]}
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "identity":
        rows[1]["model_revision"] = "different"
    elif mutation == "baseline":
        rows[0]["shift"]["reference_to_image"][0][2] = 1
    elif mutation == "label":
        rows[1]["shift"]["expected_pass"] = True
    elif mutation == "mask":
        rows[1]["prediction"] = "ff"
    elif mutation == "budget":
        p["maximum_total_pixels"] = 1
    else:
        rows[1]["support"] = pack(np.zeros((8, 8), bool))
    block = make("ml.evaluate_input_shifts", **p)
    assert block.execute({"observations": rows}) is None and not block.current_emissions
