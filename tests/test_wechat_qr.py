"""Qualify the WeChat binding, known labels and model provenance."""

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.graphics_layers import RESULT_GRAPHICS
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/wechat_qr"
TYPE = "inspect.read_wechat_qr"


@pytest.fixture(autouse=True)
def small_cpu_pool():
    threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        yield
    finally:
        cv2.setNumThreads(threads)


@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize("with_models", [False, True])
def test_saved_comparison(profile, with_models):
    path = EXAMPLE / ("with_models.json" if with_models else "qr_comparison.json")
    graph = Graph.load(path, builtin_registry())
    graph.set_run_profile(profile)
    graph.set_branch_workers(2)
    try:
        if with_models and not all(
            (EXAMPLE / graph.blocks["cnn"].parameters[name]).exists()
            for name in graph.blocks["cnn"].project_path_parameters
        ):
            pytest.skip("Run the example's prepare_models.py for native ONNX checks")
        expected = [
            {"LOT-A17": [57, 70, 78, 91]},
            {"LOT-A17": [59, 72, 101, 114], "LOT-B24": [339, 201, 381, 243]},
        ]
        for index, truths in enumerate(expected):
            graph.run()
            counts = {"standard": 0, "without_models": 1}
            if with_models:
                counts["cnn"] = index + 1
            for key, count in counts.items():
                node = graph.blocks[key]
                result = node.last_result
                assert result and result.passed, node.error
                assert result.outputs["count"] == count
                codes = {code["text"]: code for code in result.outputs["codes"]}
                assert set(codes) <= truths.keys()
                if key == "cnn":
                    assert set(codes) == truths.keys()
                    info = result.outputs["info"]
                    assert info["models_sha256"] == {
                        role: node.parameters[f"{role}_sha256"]
                        for role in ("detector", "super_resolution")
                    }
                for text, code in codes.items():
                    left, top, right, bottom = truths[text]
                    np.testing.assert_allclose(
                        code["corners"],
                        [[left, top], [right, top], [right, bottom], [left, bottom]],
                        atol=2,
                        rtol=0,
                    )
                if key != "standard":
                    assert block_entry(node).path[:2] == (
                        "OpenCV",
                        "WeChat QR Code Detection (wechat_qrcode)",
                    )
        assert graph.sources_exhausted()
    finally:
        graph.close()


def test_selection_no_read_ticket_payload_and_drawings():
    image = np.full((300, 460), 255, np.uint8)
    encoder = cv2.QRCodeEncoder_create()
    for x, text in ((45, "LOT-A17"), (321, "LOT-B24")):
        encoded = encoder.encode(text)
        label = cv2.resize(encoded, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        size = label.shape[0]
        image[173 : 173 + size, x : x + size] = label
    image = ImageArray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA), "rgba")
    block = builtin_registry().create(TYPE, parameters={"draw_overlay": False})
    block.exposed_outputs.append(RESULT_GRAPHICS)
    block.set_graphics_mode("off")
    spatial = frame([460, 300], "camera")
    ticket = TicketClock("qrcode").issue("camera", "fixed", "test")
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[:, 300:] = 255
    result = block.execute(
        {
            "image": ValuePacket(image, {"ticket": ticket, "spatial": spatial}),
            "mask": ValuePacket(mask, {"spatial": spatial}),
        }
    )
    assert result and result.passed, block.error
    assert [record["text"] for record in result.outputs["codes"]] == ["LOT-B24"]
    assert result.payloads["codes"]["ticket"] == ticket
    assert result.payloads["codes"]["spatial"] == spatial
    assert result.outputs["overlay"] is NO_VALUE
    layers = result.outputs[RESULT_GRAPHICS].layers
    outline, x_arrow, y_arrow, label = layers[0].objects
    assert outline.closed and label.text == "LOT-B24"
    assert x_arrow.kind == y_arrow.kind == "arrow"
    assert x_arrow.points[1][0] > x_arrow.points[0][0]
    masked = block.execute({"image": image, "mask": np.zeros_like(mask)})
    assert masked is not None and not masked.passed and not block.error
    assert masked.fail_reasons[0]["code"] == "code_read_count_below_minimum"


@pytest.mark.parametrize(
    ("params", "error"),
    [
        ({"scale_factor": 0.5}, "requires a CNN"),
        ({"scale_factor": 0.001}, "between 0.01 and 1"),
        ({"detector_model_path": "missing.onnx"}, "64-character"),
        ({"detector_model_path": "missing.onnx", "detector_sha256": "0" * 64}, "existing detector"),
        (
            {"detector_model_path": "old.caffemodel", "detector_sha256": "0" * 64},
            "requires an ONNX",
        ),
    ],
)
def test_invalid_models_and_scale_are_block_errors(params, error):
    block = builtin_registry().create(TYPE, parameters=params)
    image = np.zeros((88, 88), np.uint8)
    assert block.execute({"image": image}) is None
    assert error in block.error


def test_model_hash_checked_before_native_constructor(tmp_path, monkeypatch):
    path = tmp_path / "detector.onnx"
    path.write_bytes(b"first revision")
    pinned = hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_bytes(b"updated revision")
    block = builtin_registry().create(
        TYPE, parameters={"detector_model_path": str(path), "detector_sha256": pinned}
    )
    image = np.zeros((104, 104), np.uint8)

    def unexpected(*args):
        raise AssertionError("An unverified artifact reached the native constructor")

    with monkeypatch.context() as patch:
        patch.setattr(cv2.wechat_qrcode, "WeChatQRCode", unexpected)
        assert block.execute({"image": image}) is None
        assert "SHA-256 differs" in block.error
    block.parameters.update(detector_model_path="", detector_sha256="")
    result = block.execute({"image": image})
    assert result is not None and not result.passed and not block.error
