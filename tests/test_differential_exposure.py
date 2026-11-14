"""Independent photometric fits and known defects under controlled illumination."""

import copy

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.differential_exposure import DifferentialExposure
from opencv_dataflow.images import DepthArray, ImageArray
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry


def run(image, reference, **parameters):
    tool = DifferentialExposure(parameters=parameters)
    result = tool.execute({"image": image, "reference": reference})
    assert result is not None, tool.error
    return tool, result


@pytest.mark.parametrize("scope", ["global", "local"])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16, np.float32, np.float64])
@pytest.mark.parametrize("encoding", ["gray", "rgb", "bgra"])
def test_known_affine_lighting_preserves_source_depth_encoding_and_alpha(scope, dtype, encoding):
    base = np.random.default_rng(116).integers(8, 30, (19, 23, 3)) * 4
    image = base[..., 0] if encoding == "gray" else base
    gain = 1.25 if encoding == "gray" else np.array([0.5, 1.25, 2.0])
    offset = 8 if encoding == "gray" else np.array([12, -4, 8])
    reference = image * gain + offset
    if encoding == "bgra":
        image = np.dstack([image, np.full(image.shape[:2], 97)])
        reference = np.dstack([reference, np.full(image.shape[:2], 213)])
    image, reference = (
        ImageArray(image.astype(dtype), encoding),
        ImageArray(reference.astype(dtype), encoding),
    )
    original = image.copy()
    _, result = run(
        image,
        reference,
        scope=scope,
        window=9,
        minimum_fit_pixels=4,
        minimum_contrast=0.01,
        defect_threshold=1e-6,
    )
    assert result.passed
    expected = reference.copy()
    if encoding == "bgra":
        expected[..., 3] = original[..., 3]
    np.testing.assert_array_equal(result.outputs["image"], original)
    np.testing.assert_allclose(result.outputs["adjusted"], expected, atol=1e-12)
    np.testing.assert_allclose(
        result.outputs["gain"], np.broadcast_to(gain, result.outputs["gain"].shape), atol=1e-11
    )
    assert result.outputs["adjusted"].dtype == dtype
    assert result.outputs["adjusted"].color_format == encoding
    assert result.outputs["info"]["adjusted"]["maximum"] < 1e-10
    assert result.outputs["info"]["raw"]["mae"] > 1


def test_local_moments_match_independent_window_least_squares_at_borders_and_fit_holes():
    y, x = np.mgrid[:9, :11]
    source = np.random.default_rng(47).uniform(30, 130, x.shape)
    reference = (1 + x / 100) * source + y
    fit = np.full(source.shape, 255, np.uint8)
    fit[3:5, 4:6] = 0
    tool = DifferentialExposure(
        parameters={
            "window": 5,
            "robust_loss": "none",
            "minimum_fit_pixels": 2,
            "minimum_contrast": 0,
            "minimum_gain": 0.01,
            "maximum_gain": 10,
            "defect_threshold": 100,
        }
    )
    result = tool.execute({"image": source, "reference": reference, "fit_mask": fit})
    assert result is not None, tool.error
    for row in range(9):
        for col in range(11):
            window = (slice(max(0, row - 2), row + 3), slice(max(0, col - 2), col + 3))
            a, b = source[window][fit[window] != 0], reference[window][fit[window] != 0]
            expected = np.linalg.lstsq(np.c_[a, np.ones(len(a))], b, rcond=None)[0]
            assert result.outputs["gain"][row, col] == pytest.approx(expected[0], abs=1e-11)
            assert result.outputs["offset"][row, col] == pytest.approx(expected[1], abs=1e-9)


@pytest.mark.parametrize("scope", ["global", "local"])
def test_robust_fit_retains_injected_defect_instead_of_letting_it_drive_correction(scope):
    reference = np.random.default_rng(116).uniform(50, 150, (64, 96))
    source = (reference - 8) / 1.2
    truth = np.zeros(source.shape, bool)
    truth[30:34, 40:45] = True
    source[truth] += 40
    _, result = run(source, reference, scope=scope, window=31, minimum_fit_pixels=8)
    assert not result.passed and [r["code"] for r in result.fail_reasons] == ["exposure_defects"]
    np.testing.assert_array_equal(result.outputs["defect_mask"] != 0, truth)
    np.testing.assert_allclose(result.outputs["residual"][truth], 48, atol=1e-10)
    assert result.outputs["residual"][~truth].max() < 1e-9
    assert not result.outputs["fit_inliers"][truth].any()
    info = result.outputs["info"]
    expected_mae = 48 * truth.mean()
    assert info["adjusted"]["mae"] == pytest.approx(expected_mae, abs=1e-10)
    assert info["adjusted"]["rmse"] == pytest.approx(48 * np.sqrt(truth.mean()), abs=1e-10)
    _, allowed = run(
        source,
        reference,
        scope=scope,
        window=31,
        minimum_fit_pixels=8,
        maximum_defect_fraction=truth.mean(),
    )
    assert allowed.passed  # Inclusive fraction limit.


def test_stable_fit_region_can_exclude_a_broad_change_from_the_photometric_model():
    source = np.tile(np.arange(24, 104, dtype=float), (40, 1))
    reference = 1.5 * source + 7
    source[15:25, 30:50] += 20
    selection = np.zeros(source.shape, np.uint8)
    selection[15:25, 30:50] = 255
    fit = {
        "model": "image_region_v1",
        "image_size": [80, 40],
        "shape": "rectangle",
        "rectangle": [0, 0, 80, 10],
    }
    tool = DifferentialExposure(parameters={"scope": "global", "defect_threshold": 15})
    result = tool.execute(
        {"image": source, "reference": reference, "mask": selection, "fit_region": fit}
    )
    assert result is not None, tool.error
    assert result.outputs["info"]["selected_pixels"] == 200
    assert result.outputs["info"]["fit_pixels"] == 800
    assert result.outputs["info"]["defect_pixels"] == 200
    np.testing.assert_allclose(result.outputs["residual"][selection != 0], 30, atol=1e-10)
    np.testing.assert_array_equal(
        result.outputs["adjusted"][selection == 0], source[selection == 0]
    )
    assert not result.outputs["raw_residual"][selection == 0].any()


def test_low_contrast_offset_fallback_regularization_and_offset_only_mode():
    image = np.full((13, 17), 40, np.float64)
    _, flat = run(image, image + 12, scope="global")
    assert flat.passed and flat.outputs["info"]["offset_only_pixels"] == image.size
    np.testing.assert_array_equal(flat.outputs["gain"], np.ones_like(image))
    np.testing.assert_array_equal(flat.outputs["offset"], np.full_like(image, 12))
    image = np.arange(1, 222, dtype=float).reshape(13, 17)
    _, regularized = run(
        image, image * 2 + 8, scope="global", robust_loss="none", gain_regularization=20
    )
    expected_gain = (2 * image.var() + 400) / (image.var() + 400)
    assert regularized.outputs["gain"][0, 0] == pytest.approx(expected_gain)
    _, offset = run(image, image + 7, scope="global", model="offset_only")
    assert offset.passed and offset.outputs["info"]["offset_only_pixels"] == image.size


@pytest.mark.parametrize("case", ["empty_fit", "bad_gain", "too_few", "clipped"])
def test_insufficient_evidence_emits_a_failed_inspection_with_explicit_coverage(case):
    image = np.arange(1, 222, dtype=float).reshape(13, 17)
    reference = image.copy()
    parameters = {"scope": "global"}
    extra = {}
    if case == "empty_fit":
        extra["fit_mask"] = np.zeros(image.shape, np.uint8)
    elif case == "bad_gain":
        reference = 10 * image
    elif case == "too_few":
        parameters["minimum_fit_pixels"] = 1000
    else:
        image = np.zeros(image.shape, np.uint8)
        reference = image.copy()
    tool = DifferentialExposure(parameters=parameters)
    result = tool.execute({"image": image, "reference": reference, **extra})
    assert result is not None, tool.error
    assert not result.passed and [r["code"] for r in result.fail_reasons] == [
        "exposure_insufficient_support"
    ]
    assert result.outputs["info"]["valid_pixels"] == 0
    assert result.outputs["info"]["adjusted"] is None
    assert result.outputs["info"]["defect_fraction"] is None
    np.testing.assert_array_equal(result.outputs["adjusted"], image)


def test_output_rounding_and_clipping_do_not_hide_the_numeric_residual():
    source = np.full((8, 12), 250, np.uint8)
    reference = source.copy()
    source[:4] = 100
    reference[:4] = 150
    fit = np.zeros(source.shape, np.uint8)
    fit[:4] = 255
    tool = DifferentialExposure(
        parameters={"scope": "global", "model": "offset_only", "robust_loss": "none"}
    )
    result = tool.execute({"image": source, "reference": reference, "fit_mask": fit})
    assert result is not None, tool.error
    np.testing.assert_array_equal(result.outputs["adjusted"][4:], np.full((4, 12), 255))
    np.testing.assert_allclose(result.outputs["residual"][4:], 50, atol=1e-10)


def test_gain_and_residual_limits_are_inclusive_with_float64_roundoff():
    source = np.arange(1, 222, dtype=float).reshape(13, 17)
    reference = source * 1.25 + 8
    source[6, 7] += 4
    fit = np.full(source.shape, 255, np.uint8)
    fit[6, 7] = 0
    tool = DifferentialExposure(
        parameters={
            "scope": "global",
            "minimum_gain": 1.25,
            "maximum_gain": 1.25,
            "defect_threshold": 5,
        }
    )
    result = tool.execute({"image": source, "reference": reference, "fit_mask": fit})
    assert result is not None and result.passed, tool.error
    tool.parameters["defect_threshold"] = 5 - 1e-9
    result = tool.execute({"image": source, "reference": reference, "fit_mask": fit})
    assert not result.passed and result.outputs["info"]["defect_pixels"] == 1


def test_saturation_override_and_selection_failures():
    source = np.zeros((8, 12), np.uint8)
    _, result = run(source, source, scope="global", saturation="none")
    assert result.passed and result.outputs["valid_mask"].all()
    tool = DifferentialExposure(parameters={"scope": "global"})
    assert tool.execute({"image": source, "reference": source, "mask": source}) is None
    assert tool.error_kind == "invalid"
    source = np.full((8, 12, 4), 50, np.uint8)
    wrapped = ImageArray(source, "rgba", {"alpha_representation": "premultiplied"})
    assert tool.execute({"image": wrapped, "reference": wrapped}) is None
    assert "Unassociate" in tool.error


def test_candidate_context_is_retained_and_same_frame_checks_are_strict():
    source = np.arange(1, 222, dtype=float).reshape(13, 17)
    payload = {
        "spatial": frame([17, 13], "camera"),
        "ticket": TicketClock().issue("candidate"),
        "recipe_ref": {"id": "inspection", "revision": 1, "digest": "a" * 64},
    }
    ref_payload = {"spatial": frame([17, 13], "camera"), "ticket": TicketClock().issue("trained")}
    tool = DifferentialExposure(parameters={"scope": "global", "reference_alignment": "same_frame"})
    result = tool.execute(
        {"image": ValuePacket(source, payload), "reference": ValuePacket(source + 8, ref_payload)}
    )
    assert result is not None and result.passed, tool.error
    for name in result.outputs:
        assert result.payloads[name]["spatial"] == payload["spatial"]
        assert result.payloads[name]["ticket"] == payload["ticket"]
        assert result.payloads[name]["recipe_ref"] == payload["recipe_ref"]
    assert result.outputs["info"]["reference_ticket"] == ref_payload["ticket"]
    for field in ("root_id", "root_size", "to_root"):
        changed = copy.deepcopy(ref_payload)
        if field == "root_id":
            changed["spatial"][field] = "other"
        elif field == "root_size":
            changed["spatial"][field] = [18, 13]
        else:
            changed["spatial"][field][0][2] += 1
        assert (
            tool.execute(
                {"image": ValuePacket(source, payload), "reference": ValuePacket(source, changed)}
            )
            is None
        )
        assert tool.error_kind == "invalid" and not tool.current_emissions


@pytest.mark.parametrize(
    "parameters",
    [
        {"window": 4},
        {"window": 257},
        {"minimum_gain": 5, "maximum_gain": 4},
        {"minimum_fit_fraction": -0.1},
        {"defect_threshold": float("nan")},
        {"outlier_limit": 0},
        {"iterations": 11},
        {"window": 3},
        {"scope": "invalid"},
    ],
)
def test_bad_configuration_is_contained(parameters):
    image = np.ones((16, 20), np.uint8)
    tool = DifferentialExposure(parameters=parameters)
    assert tool.execute({"image": image, "reference": image}) is None
    assert tool.error_kind == "invalid" and not tool.current_emissions


@pytest.mark.parametrize(
    "source,reference",
    [
        (np.ones((8, 10)), np.ones((9, 10))),
        (np.ones((8, 10), np.uint8), np.ones((8, 10), np.uint16)),
        (np.full((8, 10), np.nan), np.ones((8, 10))),
        (np.full((8, 10), 1e20), np.ones((8, 10))),
        (DepthArray(np.ones((8, 10), np.float32), "m"), np.ones((8, 10), np.float32)),
        (
            ImageArray(np.ones((8, 10, 3), np.uint8), "hsv"),
            ImageArray(np.ones((8, 10, 3), np.uint8), "hsv"),
        ),
    ],
)
def test_unsupported_images_emit_no_result(source, reference):
    tool = DifferentialExposure()
    assert tool.execute({"image": source, "reference": reference}) is None
    assert tool.error_kind == "invalid" and not tool.current_emissions


def test_graphics_limits_native_failures_and_offscreen_controls(app, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QComboBox

    import opencv_dataflow.differential_exposure as module
    from opencv_dataflow.app import MainWindow
    from opencv_dataflow.assistant_api import block_schema

    registry = builtin_registry()
    tool = registry.create(DifferentialExposure.type_id, parameters={"window": 9})
    source = np.arange(1, 222, dtype=float).reshape(13, 17)
    assert tool.execute({"image": source, "reference": source}).drawings
    tool.graphics_mode = "off"
    assert not tool.execute({"image": source, "reference": source}).drawings
    assert block_schema(tool.type_id, registry)["opencv_functions"] == ["boxFilter"]
    assert block_entry(tool).path == (
        "OpenCV",
        "Image Processing (imgproc)",
        "Reference inspection",
    )
    graph = Graph()
    graph.add(tool)
    for suffix in ("json", "xml"):
        path = tmp_path / f"photometric.{suffix}"
        graph.save(path)
        restored = Graph.load(path, registry)
        assert restored.to_dict() == graph.to_dict()
        restored.close()
    window = MainWindow()
    try:
        window.replace_graph(graph)
        app.processEvents()
        window.inspector.select(tool)
        for name in tool.parameter_choices:
            assert isinstance(window.inspector.fields[name], QComboBox)
        assert window.inspector.can_preview()
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
    monkeypatch.setattr(module, "MAX_PIXELS", 20)
    assert tool.execute({"image": source, "reference": source}) is None
    monkeypatch.setattr(module, "MAX_PIXELS", 4_194_304)

    def broken(*args, **kwargs):
        raise cv2.error("Controlled local fit failure")

    monkeypatch.setattr(cv2, "boxFilter", broken)
    assert tool.execute({"image": source, "reference": source}) is None
    assert "Controlled local fit failure" in tool.error and not tool.current_emissions
