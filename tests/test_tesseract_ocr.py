"""Qualify word geometry, language pins, and bounded OCR failure behavior."""

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.parameter_validation import ValidationError
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tesseract_ocr import TesseractOCR, _executable, _tsv_words
from opencv_dataflow.tickets import TicketClock

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/tesseract_ocr"
EXPECTED = "PUMP 7 VALVE 24 LOT 305"
PIN = "7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2"
TSV = b"level\tleft\ttop\twidth\theight\tconf\ttext\n"


@pytest.fixture
def runtime():
    try:
        return _executable()
    except ValidationError:
        pytest.skip("Tesseract executable unavailable")


def iou(a, b):
    x, y, w, h = a
    u, v, p, q = b
    intersection = max(0, min(x + w, u + p) - max(x, u)) * max(0, min(y + h, v + q) - max(y, v))
    return intersection / (w * h + p * q - intersection)


@pytest.mark.parametrize(("profile", "workers"), [("debug", 1), ("lean", 3)])
def test_saved_graph_recognizes_and_locates_six_words(runtime, profile, workers):
    graph = Graph.load(EXAMPLE / "tesseract_ocr.json", builtin_registry())
    truth = json.loads((EXAMPLE / "truth.json").read_text("utf-8"))["words"]
    try:
        graph.set_run_profile(profile)
        graph.set_branch_workers(workers)
        graph.run()
        ocr = graph.blocks["ocr"]
        result = ocr.last_result
        assert result and result.passed, ocr.error
        assert result.outputs["text"] == EXPECTED
        words = result.outputs["words"]["items"]
        assert len(words) == len(truth) == 6
        for item, expected in zip(words, truth, strict=True):
            assert item["text"] == expected["text"]
            assert item["crop_index"] == expected["crop_index"]
            assert 0.90 < item["confidence"] <= 1
            assert iou(item["bounds"], expected["bounds"]) > 0.9
        assert result.payloads["words"]["ticket"]["value"] == "labels.png"
        assert result.payloads["words"]["spatial"]["root_id"] == "source:image"
        assert result.outputs["report"]["language_sha256"] == {"eng": PIN}
        assert graph.blocks["filmstrip"].last_result.passed
        np.testing.assert_array_equal(
            result.outputs["image"], graph.blocks["source"].last_result.outputs["image"]
        )
        if profile == "debug":
            assert len(result.drawings["Recognized words"].objects) == 12
        else:
            assert not result.drawings
    finally:
        graph.close()


def test_roi_filters_source_and_confidence_threshold(runtime):
    image = cv2.imread(str(EXAMPLE / "assets/labels.png"))
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[85:170, :300] = 255
    p = {"layout": "line", "tessdata_path": str(EXAMPLE / "assets/tessdata")}
    result = TesseractOCR(parameters=p).execute({"image": image, "mask": mask})
    assert result and result.outputs["text"] == "VALVE 24"
    assert all(85 <= item["bounds"][1] < 170 for item in result.outputs["words"]["items"])
    strict = TesseractOCR(parameters={**p, "minimum_confidence": 100})
    rejected = strict.execute({"image": image, "mask": mask})
    assert rejected and not rejected.passed and rejected.outputs["words"]["items"] == []
    assert rejected.fail_reasons[0]["code"] == "tesseract.no_words"


def test_pin_and_same_ticket_checks_run_before_the_executable(monkeypatch):
    def must_not_run():
        pytest.fail("Invalid input reached executable selection")

    monkeypatch.setattr("opencv_dataflow.tesseract_ocr._executable", must_not_run)
    image = np.zeros((30, 40), np.uint8)
    wrong = TesseractOCR(
        parameters={
            "tessdata_path": str(EXAMPLE / "assets/tessdata"),
            "language_sha256": {"eng": "0" * 64},
        }
    )
    assert wrong.execute({"image": image}) is None
    assert "SHA-256 differs" in wrong.error
    clock = TicketClock("test")
    spatial = frame([40, 30], "camera")
    issued = [
        {"spatial": spatial, "ticket": clock.issue("test", "fixed", key)} for key in ("a", "b")
    ]
    regions = {
        "model": "ocr_regions_v1",
        "image_size": [40, 30],
        "items": [{"corners": [[1, 1], [30, 1], [30, 22], [1, 22]]}],
    }
    block = TesseractOCR()
    assert (
        block.execute(
            {
                "image": ValuePacket(image, issued[0]),
                "regions": ValuePacket(regions, issued[1]),
            }
        )
        is None
    )
    assert "another ticket" in block.error


@pytest.mark.parametrize(
    "row",
    [
        b"5\t1\t1\t5\t2\tnan\tword\n",
        b"5\t1\t1\t80\t2\t99.9\tword\n",
        b"5\t0\t1\t5\t2\t102.0\tword\n",
    ],
)
def test_rejects_malformed_engine_geometry_and_confidence(row):
    with pytest.raises(ValidationError, match="malformed"):
        list(_tsv_words(TSV + row, (24, 36), 30))


def test_timeout_reports_a_processing_error(monkeypatch):
    monkeypatch.setattr("opencv_dataflow.tesseract_ocr._executable", lambda: Path("tesseract.exe"))

    def timed_out(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr("opencv_dataflow.tesseract_ocr.subprocess.run", timed_out)
    block = TesseractOCR()
    assert block.execute({"image": np.zeros((20, 40), np.uint8)}) is None
    assert "exceeded the OCR time limit" in block.error
