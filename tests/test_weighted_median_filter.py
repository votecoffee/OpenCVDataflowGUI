"""Native modes, sample eligibility and known weighted-median edges."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

TYPE = "cv.weighted_median_filter"
EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/weighted_median"


def tool(**parameters):
    return builtin_registry().create(TYPE, parameters=parameters)


@pytest.mark.parametrize(
    ("weight", "constant"),
    [
        ("gaussian", "WMF_EXP"),
        ("inverse_linear", "WMF_IV1"),
        ("inverse_squared", "WMF_IV2"),
        ("cosine", "WMF_COS"),
        ("jaccard", "WMF_JAC"),
        ("unweighted", "WMF_OFF"),
    ],
)
def test_every_weight_mode_matches_opencv(weight, constant):
    rng = np.random.default_rng(2027)
    signal = rng.integers(1000, 62000, (26, 33), dtype=np.uint16)
    palette = np.array([[60, 30, 200], [200, 110, 40], [70, 85, 130]], np.uint8)
    guide = palette[rng.integers(0, 3, signal.shape)]
    rgb = ImageArray(guide[..., ::-1], "rgb")
    samples = np.full(signal.shape, 255, np.uint8)
    samples[8:13, 5:8] = 0
    expected = cv2.ximgproc.weightedMedianFilter(
        guide,
        signal.astype(np.float32),
        2,
        sigma=33,
        weightType=getattr(cv2.ximgproc, constant),
        mask=samples,
    ).astype(np.uint16)
    block = tool(weight=weight, radius=2, sigma_color=33)
    result = block.execute({"image": signal, "guide": rgb, "sample_mask": samples})
    assert result and result.passed, block.error
    assert result.outputs["image"].dtype == np.uint16
    np.testing.assert_array_equal(result.outputs["image"], expected)


def test_sparse_samples_and_output_region_are_independent():
    source = ImageArray(np.full((24, 32, 4), [120, 200, 315, 10], dtype=np.uint16), "rgba")
    source[10, 10, :3] = [1200, 4567, 61480]
    source[..., 3] = np.arange(24 * 32, dtype=np.uint16).reshape(24, 32)
    guide = np.full(source.shape[:2], 95, dtype=np.uint8)
    samples = np.zeros_like(guide)
    samples[10, 10] = 255
    mask = np.zeros_like(guide)
    mask[:, :13] = 255
    region = {
        "model": "image_region_v1",
        "image_size": [32, 24],
        "shape": "rectangle",
        "rectangle": [9, 7, 8, 5],
    }
    shape = [32, 24]
    context = frame(shape, "camera")
    ticket = TicketClock("median").issue("camera", "fixed", "part-2")
    block = tool(radius=2)
    packet = ValuePacket(source, {"ticket": ticket, "spatial": context})
    inputs = {
        "image": packet,
        "guide": ValuePacket(guide, {"spatial": context}),
        "sample_mask": ValuePacket(samples, {"spatial": context}),
        "mask": ValuePacket(mask, {"spatial": context}),
        "region": region,
    }
    result = block.execute(inputs)
    assert result and result.passed, block.error
    actual = result.outputs["image"]
    expected = source.copy()
    expected[8:12, 9:13, :3] = source[10, 10, :3]
    assert actual.color_format == "rgba" and actual.dtype == np.uint16
    np.testing.assert_array_equal(actual, expected)
    assert result.payloads["image"]["ticket"] == ticket
    assert result.payloads["image"]["spatial"] == context
    inputs["sample_mask"] = ValuePacket(np.zeros_like(samples), {"spatial": context})
    result = block.execute(inputs)
    assert result and result.passed
    np.testing.assert_array_equal(result.outputs["image"], source)
    inputs["sample_mask"] = ValuePacket(samples, {"spatial": frame(shape, "other")})
    assert block.execute(inputs) is None and "coordinate frame" in block.error


def test_invalid_guides_have_actionable_errors_and_single_byte_inputs_stay_exact():
    signal = np.tile(np.arange(11, dtype=np.uint8), (11, 1))
    dark = np.zeros((*signal.shape, 3), np.uint8)
    assert tool().execute({"image": signal, "guide": dark}).passed
    cosine = tool(weight="cosine")
    assert cosine.execute({"image": signal, "guide": dark}) is None
    assert "nonzero guide" in cosine.error
    jaccard = tool(weight="jaccard")
    assert jaccard.execute({"image": signal, "guide": dark[..., 0]}) is None
    assert "positive monochrome" in jaccard.error
    gaussian = tool(sigma_color=1.0)
    guide = np.where(signal < 4, 12, 180).astype(np.uint8)
    samples = np.full(signal.shape, 255, np.uint8)
    samples[:, 6:9] = 0
    assert gaussian.execute({"image": signal, "guide": guide, "sample_mask": samples}) is None
    assert "Increase color sigma" in gaussian.error
    unweighted = tool(weight="unweighted")
    hdr = (signal.astype(np.float32) - 5) * 90
    result = unweighted.execute({"image": hdr})
    assert result and result.passed, unweighted.error
    assert result.outputs["image"].dtype == np.float32


def test_native_calls_from_two_blocks_are_serialized(monkeypatch):
    first_entered, release_first, second_ran = Event(), Event(), Event()
    image = np.full((18, 20), 45, np.uint8)
    other = np.full(image.shape, 85, np.uint8)

    def fake(guide, source, *args, **kwargs):
        if source[0, 0] == 45:
            first_entered.set()
            if not release_first.wait(3):
                raise TimeoutError("Test could not release the first invocation")
        else:
            second_ran.set()
        return source.copy()

    monkeypatch.setattr(cv2.ximgproc, "weightedMedianFilter", fake)
    one, two = tool(), tool()
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(one.execute, {"image": image})
        assert first_entered.wait(3)
        second_started = Event()

        def invoke():
            second_started.set()
            return two.execute({"image": other})

        second = pool.submit(invoke)
        try:
            assert second_started.wait(3)
            assert not second_ran.wait(0.1)
        finally:
            release_first.set()
        assert first.result(timeout=3).passed
        assert second.result(timeout=3).passed and second_ran.is_set()


@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_project_qualifies_noise_edge_localization_and_selection(profile):
    graph = Graph.load(EXAMPLE / "weighted_median.json", builtin_registry())
    graph.set_run_profile(profile)
    graph.set_branch_workers(3)
    try:
        graph.run()
        block = graph.blocks["weighted"]
        assert block_entry(block).path[:2] == ("OpenCV", "Extended Image Processing (ximgproc)")
        target = cv2.imread(str(EXAMPLE / "assets/clean.png"), cv2.IMREAD_GRAYSCALE)
        keys = ("noisy", "unweighted", "weighted", "samples", "selected")
        outputs = {}
        for key in keys:
            node = graph.blocks[key]
            result = node.last_result
            assert result and result.passed, node.error
            outputs[key] = result.outputs["image"]
        edge = cv2.morphologyEx(target, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        assert np.count_nonzero(outputs["noisy"] != target) == 914
        assert np.count_nonzero(outputs["unweighted"][edge] != target[edge]) == 143
        assert np.count_nonzero(outputs["weighted"] != target) == 8
        np.testing.assert_array_equal(outputs["weighted"][edge], target[edge])
        np.testing.assert_array_equal(outputs["samples"], target)
        area = np.zeros(target.shape, bool)
        area[6:83, 19:133] = True
        np.testing.assert_array_equal(outputs["selected"][area], outputs["weighted"][area])
        np.testing.assert_array_equal(outputs["selected"][~area], outputs["noisy"][~area])
    finally:
        graph.close()
