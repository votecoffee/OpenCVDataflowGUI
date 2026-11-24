"""Bilateral texture filter parity, scoped writeback, and edge qualification."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/bilateral_texture"
TYPE = "cv.bilateral_texture"


@pytest.fixture(scope="module")
def registry():
    return builtin_registry()


@pytest.mark.parametrize(("dtype", "encoding", "channels"), (
    (np.uint8, "gray", 1),
    (np.uint8, "rgb", 3),
    (np.uint8, "bgra", 4),
    (np.uint16, "rgba", 4),
    (np.float32, "bgr", 3),
    (np.float64, "gray", 1),
))
def test_native_settings_and_source_encoding(registry, dtype, encoding, channels):
    rng = np.random.default_rng(887)
    shape = (21, 26) if channels == 1 else (21, 26, channels)
    pattern = rng.random(shape)
    source = ImageArray((pattern if np.dtype(dtype).kind == "f" else
                         np.rint(pattern * np.iinfo(dtype).max)).astype(dtype), encoding)
    before = np.asarray(source).copy()
    work = before[..., :3] if channels == 4 else before
    if dtype == np.uint16:
        work = (work.astype(float) / 65535).astype(np.float32)
    elif dtype == np.float64:
        work = work.astype(np.float32)
    expected = cv2.ximgproc.bilateralTextureFilter(
        np.ascontiguousarray(work), fr=2, numIter=2, sigmaAlpha=0.0, sigmaAvg=0.09,
    )
    if dtype == np.uint16:
        expected = np.rint(np.clip(expected.astype(float), 0, 1) * 65535).astype(dtype)
    block = registry.create(TYPE, parameters={"radius": 2, "iterations": 2, "sigma_alpha": 0, "sigma_avg": 0.09})
    result = block.execute({"image": source})
    assert result and result.passed, block.error
    output = result.outputs["image"]
    assert output.dtype == dtype and output.color_format == encoding
    assert output.shape == source.shape
    np.testing.assert_array_equal(source, before)
    if channels == 4:
        np.testing.assert_array_equal(output[..., 3], before[..., 3])
        output = output[..., :3]
    np.testing.assert_allclose(output, expected, rtol=1e-7)


@pytest.mark.parametrize("edge", (False, True))
def test_selected_writes_match_the_full_native_filter_with_iteration_halo(registry, monkeypatch, edge):
    source = np.random.default_rng(445).random((101, 123), dtype=np.float32)
    spatial = frame([123, 101], "texture-camera")
    ticket = TicketClock("texture").issue("camera", "fixed", "part-88")
    payload = {"spatial": spatial, "ticket": ticket}
    radius, iterations = 2, 2
    center = (10, 11) if edge else (75, 60)
    mask = np.zeros(source.shape, np.uint8)
    mask[:, center[0] - 3:center[0] + 5] = 255
    region = {
        "model": "image_region_v1", "shape": "circle", "image_size": [123, 101],
        "center": list(center), "radius": 9, "coordinate_context": spatial,
    }
    native = cv2.ximgproc.bilateralTextureFilter
    expected = native(source, fr=radius, numIter=iterations)
    observed = []

    def track(image, **options):
        observed.append(image.shape)
        return native(image, **options)

    monkeypatch.setattr(cv2.ximgproc, "bilateralTextureFilter", track)
    block = registry.create(TYPE, parameters={"radius": radius, "iterations": iterations})
    result = block.execute({
        "image": ValuePacket(source, payload), "mask": ValuePacket(mask, payload), "region": region,
    })
    assert result and result.passed, block.error
    y, x = np.indices(source.shape)
    keep = ((x - center[0])**2 + (y - center[1])**2 <= 9**2) & (mask != 0)
    output = result.outputs["image"]
    np.testing.assert_array_equal(output[~keep], source[~keep])
    np.testing.assert_allclose(output[keep], expected[keep], rtol=0, atol=7e-7)
    assert observed == [(39, 33) if edge else (55, 44)]
    assert result.payloads["image"]["spatial"] == spatial
    assert result.payloads["image"]["ticket"] == ticket


@pytest.mark.parametrize(("settings", "error"), (
    ({"radius": 0}, "between 1 and 63"),
    ({"iterations": 0}, "between 1 and 50"),
    ({"sigma_avg": 0.0}, "negative for automatic"),
    ({"sigma_avg": 1e-32}, "negative for automatic"),
    ({"radius": 63}, "working limit"),
))
def test_invalid_parameters_stop_before_native(registry, monkeypatch, settings, error):
    monkeypatch.setattr(cv2.ximgproc, "bilateralTextureFilter", lambda *a, **k: pytest.fail("entered native filter"))
    block = registry.create(TYPE, parameters=settings)
    assert block.execute({"image": np.zeros((80, 120), np.uint8)}) is None
    assert error in block.error


def test_empty_selection_and_invalid_color_data(registry, monkeypatch):
    monkeypatch.setattr(cv2.ximgproc, "bilateralTextureFilter", lambda *a, **k: pytest.fail("entered native filter"))
    source = ImageArray(np.full((27, 29, 4), 30, np.uint8), "rgba")
    block = registry.create(TYPE)
    result = block.execute({"image": source, "mask": np.zeros(source.shape[:2], np.uint8)})
    assert result and result.passed, block.error
    np.testing.assert_array_equal(result.outputs["image"], source)
    tagged = ValuePacket(source, {"alpha_representation": "premultiplied"})
    assert block.execute({"image": tagged}) is None
    assert "Unassociate premultiplied alpha" in block.error
    assert block.execute({"image": np.full((27, 29), 1.25, np.float32)}) is None
    assert "values must be in [0,1]" in block.error


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_saved_texture_measurement_preserves_the_known_edge(registry, profile):
    graph = Graph.load(EXAMPLE / "smooth_textured_boundary.json", registry)
    graph.set_run_profile(profile)
    graph.set_branch_workers(3)
    try:
        graph.run()
        assert block_entry(graph.blocks["smooth"]).path[:2] == (
            "OpenCV", "Extended Image Processing (ximgproc)",
        )
        rows = {key: node.last_result for key, node in graph.blocks.items() if key != "note"}
        assert all(row and row.passed for row in rows.values())
        a = rows["source_texture"].outputs["value"]
        b = rows["result_texture"].outputs["value"]
        assert a == pytest.approx(8.5435, abs=0.002)
        assert b == pytest.approx(0.8004, abs=0.025)
        assert b / a < 0.10
        edge = rows["result_edge"].outputs["measurements"]
        p0, p1 = edge["edge_endpoints"]
        midpoint = np.interp(72, [p0[1], p1[1]], [p0[0], p1[0]])
        assert abs(midpoint - 96) < 0.1
        assert edge["edge_width_10_90_pixels"] < 0.5
        assert edge["edge_width_10_90_pixels"] < rows["source_edge"].outputs["measurements"]["edge_width_10_90_pixels"]
        source = rows["source"].outputs["image"]
        smooth = rows["smooth"].outputs["image"]
        outside = rows["write_region"].outputs["mask"] == 0
        np.testing.assert_array_equal(smooth[outside], source[outside])
    finally:
        graph.close()
