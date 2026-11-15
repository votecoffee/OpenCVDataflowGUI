"""Analytic, native and tiled references for registered multiband composition."""

import math

import cv2
import numpy as np
import pytest

from opencv_dataflow import multiband_mosaic as module
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.multiband_mosaic import MULTIBAND_DEFAULTS, MultibandMosaic
from opencv_dataflow.payloads import FlowGroup, ValuePacket, frame, metadata


def test_all_bands_recover_the_mean_for_full_support():
    rng = np.random.default_rng(117)
    images = rng.uniform(-5, 20, (3, 37, 53, 3))
    mask = np.ones(images.shape[1:3], bool)
    for bands in (0, 1, 3, 6):
        blender = MultibandMosaic(
            images.shape[1:], np.float64, {**MULTIBAND_DEFAULTS, "multiband_bands": bands}
        )
        for image in images:
            blender.add(image, mask)
        np.testing.assert_allclose(blender.finish(np.float64), images.mean(axis=0), atol=3e-14)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_masked_constant_preserves_depth_and_exclusive_samples(dtype, channels):
    shape = (41, 63) if channels == 1 else (41, 63, channels)
    value = 50000 if dtype == np.uint16 else 37 if dtype == np.uint8 else 0.137
    image = np.full(shape, value, dtype)
    a, b = np.zeros(shape[:2], bool), np.zeros(shape[:2], bool)
    a[:, :45], b[:, 20:] = True, True
    a[10:20, 28:32], b[10:20, 28:32] = False, False
    blender = MultibandMosaic(shape, dtype, MULTIBAND_DEFAULTS)
    for mask in (a, b):
        bad = image.copy()
        bad[~mask] = 0
        blender.add(bad, mask)
    out = blender.finish(dtype)
    assert out.dtype == dtype
    np.testing.assert_allclose(out[a | b], image[a | b], atol=1e-7, rtol=1e-6)
    np.testing.assert_array_equal(out[a ^ b], image[a ^ b])
    assert not out[~(a | b)].any()


def test_large_hdr_values_do_not_overflow_intermediate_pyramids():
    shape = (29, 39)
    mask = np.ones(shape, bool)
    b = MultibandMosaic(shape, np.float64, MULTIBAND_DEFAULTS)
    b.add(np.full(shape, 1e300), mask)
    b.add(np.full(shape, -0.5e300), mask)
    np.testing.assert_allclose(b.finish(np.float64), 0.25e300, rtol=1e-14)


@pytest.mark.parametrize("weights", ["float32", "int16"])
@pytest.mark.parametrize("channels", [1, 3])
def test_native_backend_matches_direct_cpu_reference_and_restores_opencl(weights, channels):
    rng = np.random.default_rng(117)
    shape = (39, 57) if channels == 1 else (39, 57, 3)
    a, b = [rng.integers(0, 256, shape, np.uint8) for _ in range(2)]
    ma, mb = np.zeros(shape[:2], bool), np.zeros(shape[:2], bool)
    ma[:, :43], mb[:, 15:] = True, True
    enabled = cv2.ocl.useOpenCL()
    try:
        cv2.ocl.setUseOpenCL(False)
        reference = cv2.detail.MultiBandBlender(
            0, 3, cv2.CV_32F if weights == "float32" else cv2.CV_16S
        )
        reference.prepare((0, 0, 57, 39))
        for image, mask in ((a, ma), (b, mb)):
            signal = np.repeat(image[..., None], 3, axis=2) if channels == 1 else image.copy()
            signal[~mask] = 0
            reference.feed(signal.astype(np.int16), mask.astype(np.uint8) * 255, (0, 0))
        expected, _ = reference.blend(None, None)
        if channels == 1:
            expected = expected[..., 0]
        expected = expected.clip(0, 255).astype(np.uint8)
        expected[ma & ~mb], expected[mb & ~ma] = a[ma & ~mb], b[mb & ~ma]
    finally:
        cv2.ocl.setUseOpenCL(enabled)
    blender = MultibandMosaic(
        shape,
        np.uint8,
        {
            **MULTIBAND_DEFAULTS,
            "multiband_bands": 3,
            "multiband_backend": "native_detail",
            "multiband_native_weights": weights,
        },
    )
    blender.add(a, ma)
    blender.add(b, mb)
    np.testing.assert_array_equal(blender.finish(np.uint8), expected)
    assert cv2.ocl.useOpenCL() == enabled


def _pair(a, b, shift, **parameters):
    block = builtin_registry().create(
        "cv.stitch_image_pair",
        parameters={
            "transform_source": "matrix",
            "matrix": [[1, 0, shift], [0, 1, 0], [0, 0, 1]],
            "interpolation": "nearest",
            "blend": "multiband",
            **parameters,
        },
    )
    result = block.execute({"image_a": a, "image_b": b})
    assert result is not None and result.passed, block.error
    return result


@pytest.mark.parametrize(
    "backend,dtype,encoding",
    [
        ("float64", np.uint16, "gray"),
        ("float64", np.float64, "rgb"),
        ("float64", np.float32, "rgba"),
        ("native_detail", np.uint8, "bgr"),
    ],
)
def test_phase_aligned_tiles_match_dense_pair_and_preserve_source_coordinates(
    backend, dtype, encoding
):
    rng = np.random.default_rng(117)
    shape = (103, 257) if encoding == "gray" else (103, 257, 4 if encoding == "rgba" else 3)
    a = rng.uniform(20, 100, shape).astype(dtype)
    b = rng.uniform(70, 150, shape).astype(dtype)
    a, b = ImageArray(a, encoding), ImageArray(b, encoding)
    parameters = {"multiband_backend": backend, "multiband_bands": 2}
    dense = _pair(a, b, 151, **parameters)
    layout = {
        "model": "tiled_scan_mosaic_v1",
        "image_size": [408, 103],
        "views": [
            {
                "index": i,
                "label": str(i),
                "source_size": [257, 103],
                "source_dtype": np.dtype(dtype).name,
                "source_format": encoding,
                "source_to_mosaic": [[1, 0, 151 * i], [0, 1, 0], [0, 0, 1]],
            }
            for i in range(2)
        ],
    }
    group = FlowGroup(
        tuple(
            ValuePacket(im, {"spatial": frame([257, 103], f"view-{i}")})
            for i, im in enumerate((a, b))
        ),
        ("0", "1"),
    )
    assembled = np.zeros_like(dense.outputs["image"])
    for index in range(math.ceil(408 / 64) * math.ceil(103 / 64)):
        tool = builtin_registry().create(
            "cv.render_mosaic_tile",
            parameters={
                "tile_size": 64,
                "tile_index": index,
                "blend": "multiband",
                "interpolation": "nearest",
                **parameters,
            },
        )
        result = tool.execute({"views": group, "layout": layout})
        assert result is not None, tool.error
        out, info = result.outputs["image"], result.outputs["tile"]
        assert image_format(out) == encoding and out.dtype == dtype
        x, y = info["origin"]
        w, h = info["image_size"]
        assembled[y : y + h, x : x + w] = out
        expected = np.eye(3)
        expected[:2, 2] = [x, y]
        np.testing.assert_allclose(metadata(out)["spatial"]["to_root"], expected)
        assert info["multiband"]["window_origin"][0] % 4 == 0
    if np.dtype(dtype).kind == "f":
        np.testing.assert_allclose(assembled, dense.outputs["image"], atol=3e-5, rtol=1e-6)
    else:
        assert np.max(np.abs(assembled.astype(float) - dense.outputs["image"])) <= 1


def test_work_guard_precedes_native_allocation_and_settings_validation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Workspace guard must run before native preparation")

    monkeypatch.setattr(cv2.detail, "MultiBandBlender", forbidden)
    monkeypatch.setattr(module, "MAX_WORKSPACE", 1)
    with pytest.raises(ValueError, match="workspace"):
        MultibandMosaic(
            (64, 64, 3), np.uint8, {**MULTIBAND_DEFAULTS, "multiband_backend": "native_detail"}
        )
    for key, value in [
        ("multiband_bands", 7),
        ("multiband_bands", -1),
        ("multiband_backend", "other"),
    ]:
        block = builtin_registry().create("cv.stitch_image_pair", parameters={key: value})
        assert (
            block.execute(
                {"image_a": np.ones((8, 8), np.uint8), "image_b": np.ones((8, 8), np.uint8)}
            )
            is None
        )
        assert block.error_kind == "invalid"


def test_native_rejects_high_depth_and_alpha_without_silent_conversion():
    for shape, dtype in [((32, 32), np.uint16), ((32, 32, 4), np.uint8), ((32, 32, 3), np.float32)]:
        with pytest.raises(ValueError, match="uint8"):
            MultibandMosaic(
                shape, dtype, {**MULTIBAND_DEFAULTS, "multiband_backend": "native_detail"}
            )


def test_controlled_intensity_step_has_smaller_overlap_boundary_jumps():
    shape = (67, 337)
    a, b = np.full(shape, 100, np.uint16), np.full(shape, 140, np.uint16)
    ma, mb = np.zeros(shape, bool), np.zeros(shape, bool)
    ma[:, :241], mb[:, 96:] = True, True
    blender = MultibandMosaic(shape, np.uint16, MULTIBAND_DEFAULTS)
    blender.add(a, ma)
    blender.add(b, mb)
    output = blender.finish(np.uint16)
    # Priority creates a 40-DN jump; unweighted overlap averaging creates 20 DN.
    assert np.abs(np.diff(output[33].astype(float))).max() <= 13


def test_native_exception_restores_caller_opencl_and_is_contained(monkeypatch):
    before = cv2.ocl.useOpenCL()

    def broken(*args):
        assert cv2.ocl.useOpenCL() is False
        raise cv2.error("Controlled native multiband failure")

    monkeypatch.setattr(cv2.detail, "MultiBandBlender", broken)
    block = builtin_registry().create(
        "cv.stitch_image_pair",
        parameters={
            "blend": "multiband",
            "multiband_backend": "native_detail",
            "transform_source": "matrix",
        },
    )
    image = np.full((30, 30, 3), 80, np.uint8)
    assert block.execute({"image_a": image, "image_b": image}) is None
    assert "Controlled native multiband failure" in block.error and not block.current_emissions
    assert cv2.ocl.useOpenCL() == before


def test_new_controls_are_dropdowns_and_survive_xml_reopen(app, tmp_path):
    from PySide6.QtWidgets import QComboBox

    from opencv_dataflow.app import MainWindow
    from opencv_dataflow.core import Graph

    registry = builtin_registry()
    graph = Graph()
    block = graph.add(
        registry.create(
            "cv.stitch_image_pair",
            parameters={
                "blend": "multiband",
                "multiband_bands": 2,
                "multiband_backend": "native_detail",
                "multiband_native_weights": "int16",
                "transform_source": "matrix",
            },
        )
    )
    window = MainWindow()
    try:
        window.replace_graph(graph)
        app.processEvents()
        window.inspector.select(block)
        for key in ("blend", "multiband_backend", "multiband_native_weights"):
            assert isinstance(window.inspector.fields[key], QComboBox)
        path = tmp_path / "blend.xml"
        graph.save(path)
        restored = Graph.load(path, registry)
        try:
            copy = next(iter(restored.blocks.values()))
            assert copy.parameters == block.parameters
            image = np.full((31, 43), 100, np.uint8)
            assert copy.execute({"image_a": image, "image_b": image}).passed
        finally:
            restored.close()
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
