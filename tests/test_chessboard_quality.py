"""Check native chessboard measurements and quality-gated image emission."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame, metadata
from opencv_dataflow.result_graphics import ACCEPTED, REJECTED
from opencv_dataflow.tickets import TicketClock

TYPE_ID = "calibration.chessboard_quality"
_EXAMPLE = Path(__file__).parents[1] / "examples/custom tools/chessboard_quality"


def _fixture():
    return cv2.imdecode(
        np.frombuffer((_EXAMPLE / "assets/checkerboard.png").read_bytes(), np.uint8),
        cv2.IMREAD_GRAYSCALE,
    )


def _small_board(pitch):
    rows, columns = np.indices((4, 4))
    cells = np.where((rows + columns) % 2, 232, 24).astype(np.uint8)
    board = np.repeat(np.repeat(cells, pitch, axis=0), pitch, axis=1)
    board = np.pad(board, pitch, constant_values=232)
    yy, xx = np.indices((3, 3))
    points = np.stack(((xx + 2) * pitch - 0.5, (yy + 2) * pitch - 0.5), axis=-1)
    return board, points.reshape(-1, 2).tolist()


def test_matches_native_profiles_and_preserves_color_depth_and_frame():
    gray = _fixture()
    pixels = np.repeat(gray[..., None], 3, axis=2).astype(np.uint16) * 257
    context = frame([384, 288], "checkerboard")
    image = ImageArray(pixels, "rgb", {"spatial": context})
    block = builtin_registry().create(TYPE_ID)
    block.exposed_outputs.append("result_graphics")
    result = block.execute({"image": image})
    assert result and result.passed, block.error
    output = result.outputs["image"]
    assert output.dtype == np.uint16 and image_format(output) == "rgb"
    assert metadata(output)["spatial"] == context
    np.testing.assert_array_equal(output, pixels)
    points = np.array(result.outputs["corners"], dtype=np.float32)
    report = result.outputs["report"]
    counts = []
    for axis in ("horizontal", "vertical"):
        native, profiles = cv2.estimateChessboardSharpness(
            gray, (9, 6), points, 0.8, axis == "vertical"
        )
        data = report["directions"][axis]
        assert (data["average_width_pixels"], data["average_black"], data["average_white"]) == (
            native[:3]
        )
        np.testing.assert_array_equal(data["profiles"], profiles)
        counts.append(len(profiles))
    markers = result.outputs["result_graphics"].layers[0].objects
    assert len(markers) == sum(counts) == 67
    assert all(marker.color == ACCEPTED for marker in markers)


def test_directional_blur_fails_only_the_affected_edge_orientation():
    registry = builtin_registry()
    blurred = cv2.GaussianBlur(_fixture(), (1, 19), 0, sigmaY=2.4)
    horizontal = registry.create(TYPE_ID, parameters={"orientation": "horizontal"})
    horizontal.exposed_outputs.append("result_graphics")
    failed = horizontal.execute({"image": blurred})
    assert failed and not failed.passed, horizontal.error
    assert failed.outputs["image"] is NO_VALUE and failed.outputs["max_width"] > 5
    assert failed.fail_reasons[0]["code"] == "chessboard_transition_too_wide"
    assert all(
        marker.color == REJECTED for marker in failed.outputs["result_graphics"].layers[0].objects
    )
    vertical = registry.create(TYPE_ID, parameters={"orientation": "vertical"})
    passed = vertical.execute({"image": blurred, "corners": failed.outputs["corners"]})
    assert passed and passed.passed, vertical.error
    assert passed.outputs["max_width"] < 3
    assert set(passed.outputs["report"]["directions"]) == {"vertical"}


def test_contrast_and_brightness_failures_even_when_width_check_is_disabled():
    gray = np.rint(_fixture() * 0.4 + 60).astype(np.uint8)
    block = builtin_registry().create(
        TYPE_ID,
        parameters={
            "max_transition_pixels": 0,
            "min_contrast": 100,
            "max_black": 64,
            "min_white": 160,
        },
    )
    result = block.execute({"image": gray})
    assert result and not result.passed, block.error
    assert result.outputs["image"] is NO_VALUE and result.outputs["min_contrast"] < 100
    assert {r["code"] for r in result.fail_reasons} == {
        "chessboard_low_contrast",
        "chessboard_brightness_outside_limits",
    }


def test_missing_board_and_unmeasurable_profiles_have_distinct_status():
    block = builtin_registry().create(TYPE_ID, parameters={"columns": 3, "rows": 3})
    missing = block.execute({"image": np.full((128, 128), 125, dtype=np.uint8)})
    assert missing and not missing.passed
    assert missing.outputs["report"]["status"] == "chessboard_not_found"
    assert missing.outputs["image"] is NO_VALUE
    board, points = _small_board(6)
    tiny = block.execute({"image": board, "corners": points})
    assert tiny and not tiny.passed, block.error
    assert tiny.fail_reasons[0]["code"] == "chessboard_profiles_unavailable"
    assert tiny.outputs["max_width"] is NO_VALUE
    assert tiny.outputs["report"]["directions"]["horizontal"]["status"] == "unmeasured"
    board, points = _small_board(32)
    good = block.execute({"image": board, "corners": points})
    assert good and good.passed, block.error
    assert [len(d["profiles"]) for d in good.outputs["report"]["directions"].values()] == [2, 2]


def test_wired_corners_reject_wrong_frame_or_capture():
    board, points = _small_board(32)
    size = list(board.shape[::-1])
    spatial = frame(size, "camera")
    clock = TicketClock(session="test")
    ticket = clock.issue("camera", "fixed", 9)
    later = clock.issue("camera", "fixed", 9)
    source = ImageArray(board, "gray", {"spatial": spatial, "ticket": ticket})
    block = builtin_registry().create(TYPE_ID, parameters={"columns": 3, "rows": 3})
    for payload, message in (
        ({"spatial": frame(size, "different-camera")}, "coordinate frame"),
        ({"spatial": spatial, "ticket": later}, "same captured image"),
    ):
        result = block.execute({"image": source, "corners": ValuePacket(points, payload)})
        assert result is None and message in block.error


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_example_compares_and_can_reject_the_blurred_capture(profile):
    graph = Graph.load(_EXAMPLE / "chessboard_quality.json", builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    sharp = graph.blocks["sharp"].last_result.outputs
    defocused = graph.blocks["defocused"]
    blurred = defocused.last_result.outputs
    assert sharp["max_width"] < 3 < blurred["max_width"]
    assert metadata(sharp["image"])["ticket"] == metadata(blurred["image"])["ticket"]
    defocused.parameters["max_transition_pixels"] = 3
    graph.run()
    assert defocused.state == "fail"
    assert defocused.last_result.outputs["image"] is NO_VALUE
    assert "image" not in defocused.last_output_images
