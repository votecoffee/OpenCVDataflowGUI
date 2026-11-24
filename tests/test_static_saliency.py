"""Compare the saliency wrapper with native maps and known foreground geometry."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.stats import rankdata

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.run_profiles import profile
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/static_saliency"


@pytest.fixture(scope="module")
def registry():
    return builtin_registry()


@pytest.mark.parametrize("method", ["spectral residual", "fine-grained"])
def test_matches_native_map_on_selected_typed_image(registry, method):
    bgr = cv2.imread(str(EXAMPLE / "assets/scene.png"))
    rgb = ImageArray(np.ascontiguousarray(bgr[..., ::-1], dtype=np.uint16) * 257, "rgb")
    spectral = method == "spectral residual"
    if spectral:
        detector = cv2.saliency.StaticSaliencySpectralResidual_create()
        detector.setImageWidth(48)
        detector.setImageHeight(80)
    else:
        detector = cv2.saliency.StaticSaliencyFineGrained_create()
    luminance = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)
    luminance = np.rint(luminance / 257).astype(np.uint8)
    ok, native = detector.computeSaliency(luminance)
    assert ok
    width, height = luminance.shape[::-1]
    spatial = frame([width, height], "camera")
    ticket = TicketClock("saliency").issue("camera", "fixed", "part")
    mask = np.zeros(luminance.shape, np.uint8)
    mask[60:172, 112:202] = 255
    region = {
        "model": "image_region_v1",
        "shape": "rectangle",
        "image_size": [width, height],
        "rectangle": [123, 74, 102, 88],
    }
    selected = np.zeros(luminance.shape, bool)
    selected[74:162, 123:202] = True
    expected = np.where(selected, native, 0)
    settings = {
        "method": method,
        "spectral_width": 48,
        "spectral_height": 80,
        "threshold_mode": "percentile",
        "percentile": 90,
    }
    block = registry.create("cv.static_saliency", parameters=settings)
    result = block.execute(
        {
            "image": ValuePacket(rgb, {"spatial": spatial, "ticket": ticket}),
            "mask": ValuePacket(mask, {"spatial": spatial}),
            "region": region,
        }
    )
    assert result and result.passed, block.error
    response = result.outputs["saliency"]
    assert response.dtype == np.float32 and response.color_format == "gray"
    np.testing.assert_array_equal(response, expected)
    threshold = float(np.percentile(expected[selected], 90))
    np.testing.assert_array_equal(
        result.outputs["mask"],
        np.where(selected & (expected > 0) & (expected >= threshold), 255, 0),
    )
    report = result.outputs["report"]
    assert report["selected_pixels"] == int(selected.sum())
    assert report["threshold"] == threshold
    assert report["foreground_pixels"] == np.count_nonzero(result.outputs["mask"])
    x, y = report["peak"]["point"]
    assert selected[y, x] and response[y, x] == report["peak"]["score"]
    for port in ("image", "saliency", "mask", "report"):
        assert result.payloads[port]["spatial"] == spatial
        assert result.payloads[port]["ticket"] == ticket
    np.testing.assert_array_equal(result.outputs["image"], rgb)
    assert result.outputs["image"].color_format == "rgb"
    assert result.outputs["image"].dtype == np.uint16


@pytest.mark.parametrize("mode", ["fixed", "otsu"])
def test_threshold_modes_use_selected_scores(registry, mode):
    image = cv2.imread(str(EXAMPLE / "assets/scene.png"), cv2.IMREAD_GRAYSCALE)
    mask = np.zeros_like(image)
    mask[30:155, 104:208] = 255
    chosen = mask != 0
    block = registry.create(
        "cv.static_saliency", parameters={"threshold_mode": mode, "threshold": 0.31}
    )
    block.run_profile = profile("lean")
    result = block.execute({"image": image, "mask": mask})
    assert result and result.passed, block.error
    scores = result.outputs["saliency"][chosen]
    if mode == "otsu":
        levels = np.rint(scores * 255).astype(np.uint8).reshape(-1, 1)
        otsu, expected = cv2.threshold(levels, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        threshold = otsu / 255
        expected = expected.ravel()
    else:
        threshold = 0.31
        expected = np.where(scores >= threshold, 255, 0)
    np.testing.assert_array_equal(result.outputs["mask"][chosen], expected)
    assert not np.any(result.outputs["mask"][~chosen])
    assert not result.drawings
    assert result.outputs["report"]["threshold"] == threshold


@pytest.mark.parametrize("method", ["spectral residual", "fine-grained"])
def test_flat_empty_and_tiny_inputs(registry, method):
    block = registry.create(
        "cv.static_saliency",
        parameters={"method": method, "threshold_mode": "fixed", "threshold": 0},
    )
    flat = np.full((48, 61), 90, np.uint8)
    varied = flat.copy()
    varied[20:25, 25:32] = 190
    for inputs, status in (
        ({"image": flat}, "uniform luminance"),
        ({"image": varied, "mask": np.zeros_like(flat)}, "empty selection"),
    ):
        result = block.execute(inputs)
        assert result and result.passed, block.error
        assert result.outputs["report"]["status"] == status
        assert result.outputs["report"]["peak"] is None
        assert not result.outputs["saliency"].any() and not result.outputs["mask"].any()
        assert not result.drawings
    assert block.execute({"image": np.zeros((6, 20), np.uint8)}) is None
    assert "8–8192 pixels per axis" in block.error


def test_saved_example_foreground_ranking_and_overlap(registry):
    graph = Graph.load(EXAMPLE / "static_saliency.json", registry)
    try:
        graph.run()
        truth = graph.blocks["truth"].last_result.outputs["image"].ravel() != 0
        positives = int(truth.sum())
        ious = {}
        for name in ("spectral", "fine"):
            block = graph.blocks[name]
            result = block.last_result
            assert result and result.passed, block.error
            assert block_entry(block).path[:2] == ("OpenCV", "Saliency (saliency)")
            saliency = result.outputs["saliency"].ravel()
            # Pixel-ranking AUC, with average ranks for tied scores.
            auc = (rankdata(saliency)[truth].sum() - positives * (positives + 1) / 2) / (
                positives * (truth.size - positives)
            )
            assert auc > 0.98
            overlap = graph.blocks[f"{name}_overlap"].last_result
            assert overlap and overlap.passed
            observed = result.outputs["mask"].ravel() != 0
            ious[name] = overlap.outputs["measurements"]["iou"]
            assert ious[name] == np.count_nonzero(observed & truth) / np.count_nonzero(
                observed | truth
            )
            assert ious[name] > 0.52
            assert result.drawings["salient areas"].objects
        assert ious["fine"] > ious["spectral"]
    finally:
        graph.close()
