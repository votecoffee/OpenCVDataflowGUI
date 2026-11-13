"""Native encoded round trips and fail-safe recorder configuration boundaries."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.runner import main as run_project
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.video_files import VideoFile, WriteVideo
from opencv_dataflow.video_qualification import CASES, main, qualify_video_codec


@pytest.mark.parametrize("case", CASES)
def test_native_codec_count_rate_color_and_moving_part_order(tmp_path, case):
    report = qualify_video_codec(tmp_path, case)
    assert report["status"] == "passed", report
    assert all(report["checks"].values())
    assert len(report["frames"]) == 8
    assert len({f["ticket"] for f in report["frames"]}) == 8
    assert report["submission"]["submission_status"] == "submitted_unverified"
    assert report["submission"]["frame_size"] == [128, 96]
    assert report["submission"]["backend"] == (
        "CV_MJPEG" if case == "opencv_mjpeg_bgr" else "FFMPEG"
    )
    assert all(f["timestamp_source"] == "decoder" for f in report["frames"])
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("encoding", ["rgb", "rgba", "bgra", "hsv", "lab", "gray"])
@pytest.mark.parametrize("color", ["bgr", "gray"])
def test_lossless_writer_uses_tagged_encoding_and_reports_alpha_policy(tmp_path, encoding, color):
    bgr = np.zeros((32, 64, 3), np.uint8)
    bgr[:, :32] = (255, 0, 0)
    bgr[:, 32:] = (0, 0, 255)
    if encoding == "gray":
        pixels = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        expected_bgr = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
    else:
        code = {
            "rgb": cv2.COLOR_BGR2RGB,
            "rgba": cv2.COLOR_BGR2RGBA,
            "bgra": cv2.COLOR_BGR2BGRA,
            "hsv": cv2.COLOR_BGR2HSV,
            "lab": cv2.COLOR_BGR2Lab,
        }[encoding]
        pixels = cv2.cvtColor(bgr, code)
        expected_bgr = cv2.cvtColor(
            pixels,
            {
                "rgb": cv2.COLOR_RGB2BGR,
                "rgba": cv2.COLOR_RGBA2BGR,
                "bgra": cv2.COLOR_BGRA2BGR,
                "hsv": cv2.COLOR_HSV2BGR,
                "lab": cv2.COLOR_Lab2BGR,
            }[encoding],
        )
        if encoding in ("rgba", "bgra"):
            pixels[..., 3] = 0  # Discard alpha without flattening invisible pixels to black.
    expected = expected_bgr if color == "bgr" else cv2.cvtColor(expected_bgr, cv2.COLOR_BGR2GRAY)
    original = ImageArray(pixels, encoding)
    path = tmp_path / "encoded.avi"
    writer = WriteVideo(
        parameters={"path": str(path), "codec": "FFV1", "backend": "ffmpeg", "color": color}
    )
    reader = VideoFile(parameters={"path": str(path), "color_format": color})
    ticket = TicketClock().issue("camera")
    context = {"ticket": ticket, "spatial": frame([64, 32], "camera")}
    try:
        result = writer.execute({"image": ValuePacket(original, context)})
        assert result is not None and result.passed, writer.error
        assert result.payloads["info"]["ticket"] == ticket
        info = result.outputs["info"]
        assert info["input_color_format"] == encoding
        assert info["color_format"] == color
        assert info["alpha_discarded"] == (encoding in ("rgba", "bgra"))
        writer.close()
        decoded = reader.execute({})
        assert decoded is not None, reader.error
        np.testing.assert_array_equal(decoded.outputs["image"], expected)
        np.testing.assert_array_equal(original, pixels)
    finally:
        writer.close()
        reader.close()


@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize(
    "parameters,reason",
    [
        ({"color": "gray"}, "grayscale recording is not reliable"),
        ({"fps": 12.5}, "whole-number FPS"),
    ],
)
def test_actual_opencv_mjpeg_backend_rejects_known_silent_corruption(
    tmp_path, monkeypatch, automatic, parameters, reason
):
    if automatic:
        native = cv2.VideoWriter
        monkeypatch.setattr(
            cv2,
            "VideoWriter",
            lambda path, backend, *rest: native(path, cv2.CAP_OPENCV_MJPEG, *rest),
        )
    path = tmp_path / "unsupported.avi"
    block = WriteVideo(
        parameters={
            "path": str(path),
            "backend": "auto" if automatic else "opencv_mjpeg",
            **parameters,
        }
    )
    try:
        assert block.execute({"image": np.zeros((32, 64, 3), np.uint8)}) is None
        assert reason in block.error
        assert block.error_kind == "invalid"
        assert not path.exists()
        assert block._writer is None
    finally:
        block.close()


def test_unsupported_codec_backend_cleans_reservation_and_can_recover(tmp_path):
    path = tmp_path / "recovered.avi"
    block = WriteVideo(parameters={"path": str(path), "backend": "opencv_mjpeg", "codec": "FFV1"})
    try:
        assert block.execute({"image": np.zeros((32, 64, 3), np.uint8)}) is None
        assert "FFV1" in block.error and "opencv_mjpeg" in block.error
        assert not path.exists() and block._writer is None
        block.parameters["codec"] = "MJPG"
        assert (
            block.execute({"image": np.zeros((32, 64, 3), np.uint8)}).outputs["info"][
                "frames_submitted"
            ]
            == 1
        )
    finally:
        block.close()
    assert path.stat().st_size > 0


@pytest.mark.parametrize(
    "parameters,image",
    [
        ({"fps": 0}, np.zeros((32, 64), np.uint8)),
        ({"fps": float("inf")}, np.zeros((32, 64), np.uint8)),
        ({"backend": "unknown"}, np.zeros((32, 64), np.uint8)),
        ({"codec": "custom", "custom_fourcc": "bad"}, np.zeros((32, 64), np.uint8)),
        ({"codec": "custom", "custom_fourcc": "a\nbc"}, np.zeros((32, 64), np.uint8)),
        ({}, np.zeros((31, 64), np.uint8)),
        ({}, np.zeros((32, 64), np.uint16)),
        ({}, np.zeros((32, 64), np.float32)),
    ],
)
def test_invalid_recording_inputs_never_create_a_file(tmp_path, parameters, image):
    path = tmp_path / "invalid.avi"
    block = WriteVideo(parameters={"path": str(path), **parameters})
    try:
        assert block.execute({"image": image}) is None
        assert block.error
        assert block.last_result is None and block._count == 0
        assert not path.exists()
    finally:
        block.close()


def test_custom_fourcc_and_failed_midstream_activation_preserve_recording(tmp_path):
    path = tmp_path / "custom.avi"
    block = WriteVideo(
        parameters={
            "path": str(path),
            "backend": "ffmpeg",
            "codec": "custom",
            "custom_fourcc": "FFV1",
            "fps": 12.5,
        }
    )
    frames = [np.full((32, 64, 3), value, np.uint8) for value in (15, 60, 220)]
    try:
        assert block.execute({"image": frames[0]}).passed
        block.parameters["fps"] = 20
        assert block.execute({"image": frames[1]}) is None
        assert "settings or frame dimensions changed" in block.error
        block.parameters["fps"] = 12.5
        assert block.execute({"image": frames[2]}).outputs["info"]["frames_submitted"] == 2
        block.close()
        retained = path.read_bytes()
        assert block.execute({"image": frames[1]}) is None
        assert "FileExistsError" in block.error
        assert path.read_bytes() == retained
    finally:
        block.close()
    reader = VideoFile(parameters={"path": str(path)})
    try:
        np.testing.assert_array_equal(reader.execute({}).outputs["image"], frames[0])
        np.testing.assert_array_equal(reader.execute({}).outputs["image"], frames[2])
        assert reader.exhausted
    finally:
        reader.close()


def test_qualification_command_writes_report_and_preserves_existing_directory(tmp_path):
    destination = tmp_path / "receipt"
    assert main(["--output-dir", str(destination), "--case", "ffv1_bgr", "--case", "ffv1_bgr"]) == 0
    report = destination / "report.json"
    original = report.read_bytes()
    data = json.loads(original)
    assert data["passed"] and len(data["cases"]) == 1
    with pytest.raises(SystemExit) as raised:
        main(["--output-dir", str(destination)])
    assert raised.value.code == 2 and report.read_bytes() == original


def test_qualification_detects_incorrect_pixels_and_frame_order(tmp_path, monkeypatch):
    native = VideoFile.process

    def wrong_frame(self, inputs):
        result = native(self, inputs)
        result.outputs["image"] = np.roll(result.outputs["image"], 9, axis=1)
        return result

    monkeypatch.setattr(VideoFile, "process", wrong_frame)
    report = qualify_video_codec(tmp_path, "ffv1_bgr")
    assert report["status"] == "failed"
    assert report["checks"]["frame_count_and_eof"]
    assert not report["checks"]["pixels"] and not report["checks"]["order"]


def test_unavailable_native_codec_is_not_a_passing_qualification(tmp_path, monkeypatch):
    native = cv2.VideoWriter
    monkeypatch.setattr(
        cv2,
        "VideoWriter",
        lambda path, backend, *rest: native(path, cv2.CAP_OPENCV_MJPEG, *rest),
    )
    destination = tmp_path / "unavailable"
    assert main(["--output-dir", str(destination), "--case", "ffv1_bgr"]) == 1
    report = json.loads((destination / "report.json").read_text(encoding="utf-8"))
    assert not report["passed"] and report["cases"][0]["status"] == "unavailable"
    assert not (destination / "ffv1_bgr.avi").exists()


@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_public_record_and_replay_projects_have_exact_native_pixels(tmp_path, profile):
    folder = Path(__file__).parents[1] / "examples/opencv tools/video_codec_roundtrip"
    registry = builtin_registry()
    recording = Graph.load(folder / "project.json", registry)
    replay = Graph.load(folder / "replay.json", registry)
    try:
        recording.blocks["record"].parameters["path"] = str(tmp_path / "recorded.avi")
        replay.blocks["source"].parameters["path"] = str(tmp_path / "recorded.avi")
        # Absolute image paths keep the published graph relocatable when copied.
        recording.blocks["source"].parameters["paths"] = [
            str(folder / name) for name in recording.blocks["source"].parameters["paths"]
        ]
        record_path, replay_path = tmp_path / "record.json", tmp_path / "replay.json"
        recording.save(record_path)
        replay.save(replay_path)
        for project in (record_path, replay_path):
            assert run_project([str(project), "--all-images", "--profile", profile]) == 0
        decoder = VideoFile(parameters={"path": str(tmp_path / "recorded.avi")})
        try:
            for name in recording.blocks["source"].parameters["paths"]:
                np.testing.assert_array_equal(
                    decoder.execute({}).outputs["image"], cv2.imread(name)
                )
            assert decoder.exhausted
        finally:
            decoder.close()
    finally:
        recording.close()
        replay.close()
