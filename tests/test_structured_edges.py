"""Verify structured-edge model identity, source geometry, and marked boundaries."""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.structured_edges import MODEL_SHA256
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/structured_edges"


@pytest.fixture(scope="module")
def model():
    path = Path(os.getenv("OPENCV_STRUCTURED_EDGE_MODEL", EXAMPLE / ".local/models/model.yml.gz"))
    if not path.is_file():
        pytest.skip("Run the example's prepare_model.py or set OPENCV_STRUCTURED_EDGE_MODEL")
    return path.resolve()


@pytest.fixture(scope="module")
def registry():
    return builtin_registry()


def test_wrong_model_fails_before_native_loader(tmp_path, monkeypatch, registry):
    wrong = tmp_path / "model.yml.gz"
    wrong.write_bytes(b"not the reference forest")

    def unexpected_import(*args):
        pytest.fail("Incorrect bytes reached OpenCV's model reader")

    monkeypatch.setattr(cv2.ximgproc, "createStructuredEdgeDetection", unexpected_import)
    block = registry.create("cv.structured_edges", parameters={"model_path": str(wrong)})
    assert block.execute({"image": np.zeros((20, 35), np.uint8)}) is None
    assert "differs from the qualified" in block.error


def test_model_and_float_validation(tmp_path, registry):
    block = registry.create(
        "cv.structured_edges", parameters={"model_path": str(tmp_path / "missing.yml.gz")}
    )
    assert block.execute({"image": np.zeros((20, 35), np.uint8)}) is None
    assert "model is missing" in block.error
    hdr = np.full((20, 35, 3), 1.5, np.float32)
    assert block.execute({"image": hdr}) is None
    assert "values must lie in [0,1]" in block.error


def test_native_parity_with_rgba16_and_selected_source_context(model, registry):
    image = cv2.imread(str(EXAMPLE / "assets/color_target.png"))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Use an odd-width crop with ignored alpha and uint16 color.
    wide = rgb[1:, 1:].astype(np.uint16) * 257
    source = ImageArray(
        np.concatenate((wide, np.zeros((*wide.shape[:2], 1), np.uint16)), axis=-1), "rgba"
    )
    expected_rgb = rgb[1:, 1:].astype(np.float32) / 255
    detector = cv2.ximgproc.createStructuredEdgeDetection(str(model))
    response = detector.detectEdges(expected_rgb)
    orientation = detector.computeOrientation(response)
    thinned = detector.edgesNms(response, orientation, r=3, s=4, m=1.25, isParallel=False)
    assert float(response.max()) > 1
    height, width = source.shape[:2]
    spatial = frame([width, height], "camera")
    ticket = TicketClock("structured").issue("camera", "fixed", "color")
    mask = np.zeros((height, width), np.uint8)
    mask[:, :160] = 255
    region = {
        "model": "image_region_v1",
        "shape": "rectangle",
        "image_size": [width, height],
        "rectangle": [18, 14, 154, 99],
    }
    block = registry.create(
        "cv.structured_edges",
        parameters={
            "model_path": str(model),
            "suppression_radius": 3,
            "boundary_radius": 4,
            "suppression_multiplier": 1.25,
            "parallel_nms": False,
            "edge_threshold": 0.2,
        },
    )
    try:
        source_packet = ValuePacket(source, {"spatial": spatial, "ticket": ticket})
        full = block.execute({"image": source_packet})
        assert full and full.passed, block.error
        detector_instance = block._detector
        selected = block.execute(
            {
                "image": source_packet,
                "mask": ValuePacket(mask, {"spatial": spatial}),
                "region": region,
            }
        )
        assert selected and selected.passed, block.error
        assert block._detector is detector_instance
        inside = np.zeros(mask.shape, bool)
        inside[14:113, 18:160] = True
        for name, expected in (
            ("response", response),
            ("orientation", orientation),
            ("thinned", thinned),
        ):
            np.testing.assert_array_equal(full.outputs[name], expected)
            np.testing.assert_array_equal(selected.outputs[name], np.where(inside, expected, 0))
            assert selected.payloads[name]["spatial"] == spatial
            assert selected.payloads[name]["ticket"] == ticket
        np.testing.assert_array_equal(selected.outputs["edges"] != 0, inside & (thinned >= 0.2))
        assert selected.outputs["model_info"]["sha256"] == MODEL_SHA256
    finally:
        block.close()


def test_saved_example_against_marked_boundaries(model, registry):
    graph = Graph.load(EXAMPLE / "structured_edges.json", registry)
    try:
        block = graph.blocks["detect"]
        block.parameters["model_path"] = str(model)
        graph.run()
        assert block_entry(block).path[:2] == ("OpenCV", "Extended Image Processing (ximgproc)")
        result = block.last_result
        assert result and result.passed, block.error
        prediction = result.outputs["edges"] > 0
        reference = graph.blocks["reference"].last_result.outputs["image"] > 0
        window = np.ones((7, 7), np.uint8)
        near_reference = cv2.dilate(reference.astype(np.uint8), window) != 0
        near_prediction = cv2.dilate(prediction.astype(np.uint8), window) != 0
        assert np.count_nonzero(prediction & near_reference) / prediction.sum() > 0.99
        assert np.count_nonzero(reference & near_prediction) / reference.sum() > 0.99
        assert np.count_nonzero(result.outputs["thinned"]) < np.count_nonzero(
            result.outputs["response"]
        )
        assert graph.blocks["view"].state == "pass"
    finally:
        graph.close()
