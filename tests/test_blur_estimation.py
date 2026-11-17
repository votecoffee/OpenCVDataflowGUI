"""Controlled line moments, current-unit provenance and withheld uncertain hypotheses."""

import copy

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.timing import advance_timing, mark

TYPE = "inspect.estimate_motion_blur"
FAST = {
    "tile_size": 64,
    "minimum_length": 6.0,
    "maximum_length": 10.0,
    "length_step": 2.0,
    "angle_step": 90.0,
    "model_floors": [0.001],
}


def blurred_fixture(dtype=np.float64, encoding="gray", scale=1):
    rng = np.random.default_rng(841)
    sharp = rng.normal(0.45, 0.1, (128, 128))
    # Independent exact centered eight-pixel path integrated over the pixel basis.
    values = sum(
        weight * np.roll(sharp, shift, axis=1)
        for shift, weight in zip(range(-4, 5), [0.5 / 8, *([1 / 8] * 7), 0.5 / 8], strict=True)
    )
    if encoding != "gray":
        values = np.stack((values, values * 0.8, values * 0.6), axis=2)
    if np.dtype(dtype).kind == "u":
        values = np.rint(values * np.iinfo(dtype).max).astype(dtype)
    else:
        values = (values * scale).astype(dtype)
    if encoding in ("rgba", "bgra"):
        alpha = np.iinfo(dtype).max if np.dtype(dtype).kind == "u" else 1
        values = np.dstack((values, np.full((128, 128), alpha, dtype)))
    return ImageArray(values, encoding)


def run(image, parameters=None, payload=None, **inputs):
    block = builtin_registry().create(TYPE, parameters={**FAST, **(parameters or {})})
    result = block.execute({"image": ValuePacket(image, payload or {}), **inputs})
    assert result is not None, block.error
    return block, result


@pytest.mark.parametrize("encoding", ["gray", "rgb", "bgr", "rgba", "bgra"])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
def test_independent_centered_path_recovers_psf_without_changing_pixels_or_identity(
    dtype, encoding
):
    image = blurred_fixture(dtype, encoding)
    before = image.copy()
    clock = TicketClock()
    start = mark()
    payload = {
        "spatial": frame([128, 128], "camera"),
        "ticket": clock.issue("camera", "external", "part-84"),
        "processing_time": advance_timing(None, start, start, source=True),
        "pyramid": {
            "level": 1,
            "largest_size": [256, 256],
            "to_largest": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        },
    }
    _, result = run(image, payload=payload)
    assert result.passed, result.outputs["report"]
    assert result.outputs["length"] == 8 and result.outputs["angle"] == 0
    kernel = np.asarray(result.outputs["psf"])
    np.testing.assert_allclose(kernel, [[0.5 / 8, *([1 / 8] * 7), 0.5 / 8]])
    x = np.arange(kernel.shape[1]) - kernel.shape[1] // 2
    assert abs(np.sum(kernel * x)) < 1e-12 and abs(kernel.sum() - 1) < 1e-12
    np.testing.assert_array_equal(result.outputs["image"], before)
    np.testing.assert_array_equal(image, before)
    assert result.outputs["image"].dtype == dtype
    assert result.outputs["image"].color_format == encoding
    assert result.payloads["length"]["pyramid"] == payload["pyramid"]
    assert result.payloads["psf"]["ticket"] == payload["ticket"]
    assert "spatial" not in result.payloads["psf"] and "pyramid" not in result.payloads["psf"]
    assert result.payloads["psf"]["blur_estimate"]["source_pyramid"] == payload["pyramid"]
    assert result.payloads["image"]["processing_time"]["started_ns"] == start["ns"]
    assert result.payloads["diagnostic"]["spatial"] != payload["spatial"]
    assert not result.outputs["report"]["validation_used_for_fit"]


@pytest.mark.parametrize("scale", [1e-300, 1e300])
def test_extreme_float_color_units_do_not_change_the_blur_estimate(scale):
    _, result = run(blurred_fixture(np.float64, "rgba", scale))
    assert result.passed and result.outputs["length"] == 8 and result.outputs["angle"] == 0


def test_validation_pixels_never_select_the_model_or_trigger_a_refit():
    image = blurred_fixture()
    block, first = run(image)
    assert first.passed
    changed = image.copy()
    changed[:, 64:] = np.random.default_rng(842).normal(0.45, 0.1, (128, 64))
    second = block.execute({"image": changed})
    assert second is not None and not second.passed, block.error
    assert second.outputs["report"]["status"] == "validation_rejected"
    assert second.outputs["report"]["candidate"] == pytest.approx(
        first.outputs["report"]["candidate"]
    )
    for name in ("psf", "length", "angle"):
        assert second.outputs[name] is NO_VALUE and not block.current_emissions.get(name)
    assert block.execute({"image": image}).passed


def test_mask_region_and_alpha_only_admit_complete_disjoint_tiles():
    image = blurred_fixture(np.uint8, "rgba")
    image[8, 8, 3] = 0
    block, failed = run(image)
    assert not failed.passed and failed.outputs["report"]["status"] == "insufficient_tiles"
    image = np.tile(blurred_fixture(), (2, 2))
    payload = {"spatial": frame([256, 256], "camera")}
    shape = builtin_registry().create("region.shape", parameters={"rectangle": [0, 0, 128, 256]})
    region = shape.execute({"image": ValuePacket(image, payload)})
    assert region is not None
    mask = np.ones((256, 256), np.uint8) * 255
    mask[10, 10] = 0
    _, result = run(
        image,
        payload=payload,
        mask=ValuePacket(mask, payload),
        region=ValuePacket(region.outputs["region"], region.payloads["region"]),
    )
    assert result.passed, result.outputs["report"]
    tiles = result.outputs["report"]["fit_tiles"] + result.outputs["report"]["validation_tiles"]
    assert [0, 0] not in [list(a) for a in tiles]
    assert all(x < 128 for x, y in tiles)


@pytest.mark.parametrize(
    "parameters",
    [
        {"minimum_length": 25, "maximum_length": 24},
        {"tile_size": 64, "maximum_length": 24},
        {"minimum_frequency": 0.2, "maximum_frequency": 0.1},
        {"minimum_frequency": 0.2, "maximum_frequency": 0.200001},
        {"angle_step": True},
        {"minimum_tiles": 8, "maximum_tiles": 4},
        {"model_floors": [0]},
        {"model_floors": [True]},
        {"model_floors": [1e-3, 1e-3]},
        {"model_floors": [float("inf")]},
        {"maximum_candidates": 1},
    ],
)
def test_bad_controls_are_invalid_before_processing(parameters):
    block = builtin_registry().create(TYPE, parameters=parameters)
    with pytest.raises(ValueError):
        block.validate_parameters()
    assert block.last_result is None


@pytest.mark.parametrize("kind", ["empty", "nan", "wrong_encoding", "wrong_channels"])
def test_invalid_images_emit_no_current_kernel(kind):
    image = blurred_fixture()
    if kind == "empty":
        image = np.empty((0, 128), np.float64)
    elif kind == "nan":
        image[3, 4] = float("nan")
    elif kind == "wrong_encoding":
        image = ImageArray(np.repeat(image[..., None], 3, axis=2), "hsv")
    else:
        image = ImageArray(np.repeat(image[..., None], 3, axis=2), "rgba")
    block = builtin_registry().create(TYPE, parameters=FAST)
    assert block.execute({"image": image}) is None
    assert block.error_kind == "invalid" and not block.current_emissions.get("psf")


def test_array_and_search_budgets_precede_native_spectra_and_exceptions_recover(monkeypatch):
    from opencv_dataflow import blur_estimation

    image = blurred_fixture()
    original_input = blur_estimation.image_value
    monkeypatch.setattr(
        blur_estimation,
        "image_value",
        lambda *a: (_ for _ in ()).throw(AssertionError("converted")),
    )
    block = builtin_registry().create(TYPE, parameters={**FAST, "maximum_working_bytes": 1})
    assert block.execute({"image": image}) is None and "converted" not in block.error
    monkeypatch.setattr(blur_estimation, "image_value", original_input)
    original_dft = cv2.dft
    monkeypatch.setattr(cv2, "dft", lambda *a, **k: (_ for _ in ()).throw(cv2.error("DFT test")))
    block = builtin_registry().create(TYPE, parameters={**FAST, "maximum_search_work": 1})
    assert block.execute({"image": image}) is None and "DFT test" not in block.error
    block = builtin_registry().create(TYPE, parameters=FAST)
    assert block.execute({"image": image}) is None and "DFT test" in block.error
    assert not block.current_emissions.get("psf")
    monkeypatch.setattr(cv2, "dft", original_dft)
    assert block.execute({"image": image}).passed


def test_boundary_policy_and_graphics_off_are_explicit():
    image = blurred_fixture()
    settings = {"minimum_length": 8, "maximum_length": 10}
    block, rejected = run(image, settings)
    assert not rejected.passed and rejected.outputs["report"]["status"] == "search_boundary"
    block.parameters["boundary_policy"] = "allow"
    block.set_graphics_mode("off")
    result = block.execute({"image": image})
    assert result is not None and result.passed, block.error
    assert result.outputs["diagnostic"] is NO_VALUE and not result.drawings


def test_editor_and_xml_load_choices_and_limits_without_spectrum_work(app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QComboBox

    from opencv_dataflow.app import MainWindow
    from opencv_dataflow.project_xml import export_project_xml, read_project_xml
    from opencv_dataflow.toolbox import block_entry

    graph = Graph()
    block = graph.add(builtin_registry().create(TYPE, parameters=FAST))
    snapshot = copy.deepcopy(graph.to_dict())
    path = tmp_path / "motion.xml"
    export_project_xml(snapshot, path)
    restored = Graph.from_dict(read_project_xml(path), builtin_registry())
    assert restored.to_dict() == snapshot
    restored.close()
    assert block_entry(block).path[0] == "Custom"
    monkeypatch.setattr(
        cv2, "dft", lambda *a, **k: (_ for _ in ()).throw(AssertionError("executed"))
    )
    window = MainWindow()
    try:
        window.replace_graph(graph)
        window.inspector.select(block)
        for field in ("channel", "tile_size", "boundary_policy"):
            assert isinstance(window.inspector.fields[field], QComboBox)
        assert block.last_result is None
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
        graph.close()
