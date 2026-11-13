"""Signed bands, odd-size reconstruction, source frames and native in-place safety."""

import copy
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import ImagePyramid, ValuePacket, frame, metadata
from opencv_dataflow.tickets import TicketClock


def execute(kind, inputs, **parameters):
    block = builtin_registry().create(kind, parameters=parameters)
    result = block.execute(inputs)
    assert result is not None, block.error
    return block, result


def build(image, **parameters):
    return execute("cv.laplacian_pyramid", {"image": image}, **parameters)[1].outputs["images"]


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16, np.float32, np.float64])
@pytest.mark.parametrize(
    "shape,encoding", [((35, 47), "gray"), ((35, 47, 3), "rgb"), ((35, 47, 4), "rgba")]
)
def test_floating_bands_and_exact_source_roundtrip(dtype, shape, encoding):
    rng = np.random.default_rng(124)
    values = rng.integers(0, 250, shape) if np.dtype(dtype).kind == "u" else rng.normal(0, 2, shape)
    if dtype == np.uint16:
        values *= 257
    if dtype == np.int16:
        values *= 1000
    source = ImageArray(values.astype(dtype), encoding)
    saved = source.copy()
    images = build(source)
    current = np.asarray(source, np.float64)
    for index, band in enumerate(images.levels):
        if index < len(images.levels) - 1:
            coarse = cv2.pyrDown(current)
            expected = current - cv2.pyrUp(coarse, dstsize=(current.shape[1], current.shape[0]))
        else:
            expected = current
        np.testing.assert_array_equal(band, expected)
        assert band.dtype == np.float64 and band.color_format == encoding
        if index < len(images.levels) - 1:
            assert band.min() < 0 < band.max()
            current = coarse
    _, result = execute("cv.reconstruct_laplacian_pyramid", {"images": images})
    output = result.outputs["image"]
    assert output.dtype == dtype and output.color_format == encoding
    if dtype == np.float64:
        np.testing.assert_allclose(output, source, atol=2e-15, rtol=1e-14)
    else:
        np.testing.assert_array_equal(output, source)
    np.testing.assert_array_equal(source, saved)
    assert result.outputs["info"]["clipped_samples"] == 0


@pytest.mark.parametrize("shape,count", [((35, 47, 3), 4), ((1, 17, 3), 4), ((1, 1, 3), 1)])
def test_native_detail_matches_api_and_does_not_mutate_upstream_bands(shape, count):
    source = ImageArray(np.random.default_rng(44).integers(0, 256, shape, dtype=np.uint8), "rgb")
    images = build(source, backend="native_detail", levels=count)
    native = cv2.detail.createLaplacePyr(np.asarray(source), count - 1, [])
    saved = [np.asarray(level).copy() for level in images.levels]
    for actual, expected in zip(images.levels, native, strict=True):
        assert actual.dtype == np.int16
        np.testing.assert_array_equal(actual, expected.get())
    _, result = execute("cv.reconstruct_laplacian_pyramid", {"images": images})
    np.testing.assert_array_equal(result.outputs["image"], source)
    for actual, original in zip(images.levels, saved, strict=True):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("shape,count", [((1, 1), 1), ((1, 9), 5)])
def test_single_level_and_one_pixel_axis(shape, count):
    source = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    images = build(source, levels=count)
    _, result = execute("cv.reconstruct_laplacian_pyramid", {"images": images})
    np.testing.assert_array_equal(result.outputs["image"], source)


def test_float32_working_depth_and_raw_units_are_explicit():
    source = np.random.default_rng(7).uniform(0, 1000, (33, 49)).astype(np.float64)
    images = build(source, working_depth="float32", border="replicate")
    assert all(level.dtype == np.float32 for level in images.levels)
    _, result = execute(
        "cv.reconstruct_laplacian_pyramid", {"images": images}, output_depth="float64"
    )
    np.testing.assert_allclose(result.outputs["image"], source, atol=0.00015, rtol=0)
    assert result.outputs["image"].max() > 900  # No 0..1 normalization.


def test_source_context_ticket_and_local_pixel_transforms_survive_roundtrip():
    ticket = TicketClock().issue("original")
    spatial = frame([47, 35], "camera")
    spatial.update(root_size=[400, 300], to_root=[[2, 0, 17], [0, 2, 23], [0, 0, 1]])
    source = ImageArray(
        np.random.default_rng(9).integers(0, 250, (35, 47), dtype=np.uint8),
        "gray",
        {"spatial": spatial, "ticket": ticket, "fixture": "inspection"},
    )
    gaussian = execute("cv.image_pyramid", {"image": source}, levels=2)[1].outputs["images"]
    selected = execute("cv.pyramid_level", {"images": gaussian}, level="1")[1].outputs["image"]
    original_payload = metadata(selected)
    images = build(selected, levels=3)
    for index, level in enumerate(images.levels):
        expected = np.asarray(original_payload["spatial"]["to_root"]) @ np.diag(
            [2**index, 2**index, 1]
        )
        np.testing.assert_array_equal(metadata(level)["spatial"]["to_root"], expected)
        assert metadata(level)["ticket"] == ticket
    latest_ticket = TicketClock().issue("wire")
    wrapped = ValuePacket(images, {"ticket": latest_ticket, "extra": "latest"})
    _, result = execute("cv.reconstruct_laplacian_pyramid", {"images": wrapped})
    payload = result.payloads["image"]
    assert payload["spatial"] == original_payload["spatial"]
    assert payload["pyramid"] == original_payload["pyramid"]
    assert payload["fixture"] == "inspection" and payload["extra"] == "latest"
    assert payload["ticket"] == latest_ticket
    assert "laplacian_pyramid" not in payload
    np.testing.assert_array_equal(result.outputs["image"], selected)


def test_level_selector_shows_signed_preview_and_retains_raw_band_values():
    source = np.zeros((35, 47), np.uint8)
    source[8:26, 14:31] = 255
    images = build(source)
    block, result = execute("cv.pyramid_level", {"images": images}, level="1")
    np.testing.assert_array_equal(result.outputs["image"], images.levels[1])
    assert result.outputs["image"].min() < 0
    view = result.graphics["Laplacian band preview"]
    assert view.dtype == np.uint8 and view.shape == images.levels[1].shape
    assert view.min() < 128 < view.max()
    block.set_graphics_mode("off")
    assert not block.execute({"images": images}).graphics
    ordinary = execute("cv.image_statistics", {"image": images})[1]
    assert ordinary.payloads["summary"]["pyramid"]["representation"] == "laplacian"


def test_detail_gains_remove_bands_and_report_saturation():
    source = np.random.default_rng(24).integers(0, 256, (35, 47), dtype=np.uint8)
    images = build(source)
    _, result = execute(
        "cv.reconstruct_laplacian_pyramid",
        {"images": images},
        detail_gains=[0, 0, 0],
        output_depth="float64",
    )
    expected = np.asarray(images.levels[-1]).copy()
    for band in reversed(images.levels[:-1]):
        expected = cv2.pyrUp(expected, dstsize=(band.shape[1], band.shape[0]))
    np.testing.assert_array_equal(result.outputs["image"], expected)
    _, amplified = execute(
        "cv.reconstruct_laplacian_pyramid", {"images": images}, detail_gains=[16, 16, 16]
    )
    assert amplified.outputs["info"]["clipped_samples"] > 0
    assert amplified.outputs["image"].min() == 0 and amplified.outputs["image"].max() == 255


def test_invalid_or_incompatible_sets_and_parameters_fail_safely():
    source = np.arange(35 * 47, dtype=np.uint16).reshape(35, 47)
    images = build(source)
    restore = builtin_registry().create("cv.reconstruct_laplacian_pyramid")
    gaussian = execute("cv.image_pyramid", {"image": source})[1].outputs["images"]
    assert restore.execute({"images": gaussian}) is None
    assert "not a Gaussian" in restore.error
    assert restore.execute({"images": ImagePyramid(images.levels[:-1])}) is None
    assert "incompatible geometry" in restore.error
    changed = copy.deepcopy(images)
    changed.levels[1].metadata["spatial"]["root_id"] = "other"
    assert restore.execute({"images": changed}) is None
    assert "incompatible source coordinate frames" in restore.error
    for gains in ([1], [1, True, 1], [1, float("nan"), 1], [1, 17, 1]):
        restore.parameters["detail_gains"] = gains
        assert restore.execute({"images": images}) is None
        assert "Detail gains" in restore.error
    builder = builtin_registry().create("cv.laplacian_pyramid", parameters={"levels": 5})
    assert builder.execute({"image": np.zeros((3, 5), np.uint8)}) is None
    assert "more levels" in builder.error
    builder.parameters.update(levels=4, backend="native_detail")
    assert builder.execute({"image": source}) is None
    assert "uint8 three-channel" in builder.error
    builder.parameters.update(backend="floating", working_depth="float32")
    assert builder.execute({"image": np.full((35, 47), 1e300)}) is None
    assert "working depth" in builder.error


def test_native_detail_requires_fixed_border_and_unit_gains():
    source = np.random.default_rng(2).integers(0, 256, (35, 47, 3), dtype=np.uint8)
    builder = builtin_registry().create(
        "cv.laplacian_pyramid", parameters={"backend": "native_detail", "border": "replicate"}
    )
    assert builder.execute({"image": source}) is None and "reflect_101" in builder.error
    images = build(source, backend="native_detail")
    restore = builtin_registry().create(
        "cv.reconstruct_laplacian_pyramid", parameters={"detail_gains": [0, 1, 1]}
    )
    assert restore.execute({"images": images}) is None
    assert "requires unit gains" in restore.error


def test_public_example_roundtrip_and_detail_edit():
    path = (
        Path(__file__).resolve().parents[1] / "examples/opencv tools/laplacian_pyramid/project.json"
    )
    graph = Graph.load(path, builtin_registry())
    try:
        graph.run()
        assert all(
            block.state == "pass" for block in graph.blocks.values() if not block.layout_only
        )
        source = graph.blocks["source"].last_result.outputs["image"]
        restored = graph.blocks["restore"].last_result.outputs["image"]
        np.testing.assert_array_equal(source, restored)
        assert source.shape[:2] == (129, 193) and source.dtype == np.uint16
        assert graph.blocks["verify"].last_result.outputs["value"] == 0
        edited = graph.blocks["suppress_fine"].last_result.outputs["image"]
        assert np.max(np.abs(edited.astype(float) - source)) > 100
    finally:
        graph.close()
