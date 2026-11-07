"""Check fixed-profile validation of a partial identified board observation."""

import copy
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock, ticket_key

_PROJECT = (
    Path(__file__).parents[1]
    / "examples/custom tools/validate_charuco_calibration/validate_charuco_calibration.json"
)


def _graph(profile="debug"):
    graph = Graph.load(_PROJECT, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    return graph


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_separate_view_uses_only_observed_board_ids(profile):
    graph = _graph(profile)
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    observation = graph.blocks["observe"].last_result.outputs["observation"]
    result = graph.blocks["validate"].last_result
    report = result.outputs["report"]
    ids = observation["corner_ids"]
    assert 6 <= len(ids) < 24 and 0 not in ids
    assert [point["corner_id"] for point in report["points"]] == ids
    assert [point["grid"] for point in report["points"]] == [[i % 6, i // 6] for i in ids]
    assert report["worst_corner_id"] == ids[report["worst_point_index"]]
    assert report["board"] == observation["board"]
    assert report["translation_units"] == "m"
    np.testing.assert_allclose(report["translation_vector"], [-0.148, -0.088, 0.306], atol=0.003)
    assert result.outputs["rms"] < 0.15
    assert result.outputs["max_error"] >= result.outputs["rms"]
    assert [len(layer.objects) for layer in result.outputs["result_graphics"].layers] == [
        len(ids)
    ] * 2
    overlay = graph.blocks["overlay"].last_result.outputs["image"]
    assert overlay.shape == (480, 640, 3) and np.any(overlay[..., 0] != overlay[..., 1])


@pytest.fixture(scope="module")
def sample():
    graph = _graph()
    return {
        "image": graph.blocks["source"].last_result.outputs["image"],
        "calibration": graph.blocks["profile"].last_result.outputs["profile"],
        "observation": graph.blocks["observe"].last_result.outputs["observation"],
    }


@pytest.mark.parametrize(
    ("problem", "message"),
    [
        ("size", "dimensions must match"),
        ("board_points", "invalid matched points"),
        ("frame", "same coordinate frame"),
        ("transform", "same coordinate frame"),
        ("capture", "same captured image"),
    ],
)
def test_validator_rejects_observations_incompatible_with_the_image(sample, problem, message):
    inputs = copy.deepcopy(sample)
    if problem == "size":
        inputs["observation"]["image_size"][0] -= 1
    elif problem == "board_points":
        inputs["observation"]["object_points"][0][0] += 0.03
    elif problem == "frame":
        inputs["image"] = ValuePacket(inputs["image"], {"spatial": frame([640, 480], "camera-a")})
        inputs["observation"] = ValuePacket(
            inputs["observation"], {"spatial": frame([640, 480], "camera-b")}
        )
    elif problem == "transform":
        source_frame = frame([640, 480])
        observation_frame = frame([640, 480])
        observation_frame["to_root"][0][2] = 3.0
        inputs["image"] = ValuePacket(inputs["image"], {"spatial": source_frame})
        inputs["observation"] = ValuePacket(inputs["observation"], {"spatial": observation_frame})
    else:
        clock = TicketClock(session="validation")
        image_ticket = clock.issue("source", mode="fixed", value="repeated.png")
        observation_ticket = clock.issue("source", mode="fixed", value="repeated.png")
        assert ticket_key(image_ticket) == ticket_key(observation_ticket)
        inputs["image"] = ValuePacket(inputs["image"], {"ticket": image_ticket})
        inputs["observation"] = ValuePacket(inputs["observation"], {"ticket": observation_ticket})
    validate = builtin_registry().create("calibration.validate_lens")
    assert validate.execute(inputs) is None
    assert message in validate.error


def test_failed_detector_status_suppresses_reprojection_measurements(sample):
    inputs = dict(sample)
    inputs["observation"] = ValuePacket(
        inputs["observation"],
        {
            "outcome": {
                "model": "tool_outcome_v1",
                "source": {"id": "observe"},
                "passed": False,
                "chain_passed": False,
            }
        },
    )
    validate = builtin_registry().create("calibration.validate_lens")
    result = validate.execute(inputs)
    assert result is not None and not result.passed
    assert result.fail_reasons[0]["code"] == "calibration_validation_observation_failed"
    assert result.outputs["rms"] is result.outputs["max_error"] is NO_VALUE


def test_altered_profile_reports_the_corner_id_that_fails(sample):
    inputs = dict(sample)
    inputs["calibration"] = {**inputs["calibration"], "distortion": [0.6, 0, 0, 0, 0]}
    validate = builtin_registry().create(
        "calibration.validate_lens",
        parameters={"max_rms_pixels": 0.35, "max_point_error_pixels": 0.65},
    )
    result = validate.execute(inputs)
    assert result is not None and not result.passed, validate.error
    reasons = {reason["code"]: reason["message"] for reason in result.fail_reasons}
    assert "calibration_validation_rms_exceeds_limit" in reasons
    worst = result.outputs["report"]["worst_corner_id"]
    assert f"Corner ID {worst} error" in reasons["calibration_validation_point_exceeds_limit"]
