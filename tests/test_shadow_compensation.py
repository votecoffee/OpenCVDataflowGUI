"""Analytic attenuation, defect preservation and explicit invalid-model events."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.tickets import TicketClock


def tool(**parameters):
    return builtin_registry().create("cv.reference_shadows", parameters=parameters)


def run(block, inputs):
    result = block.execute(inputs)
    assert result is not None, block.error
    return result


@pytest.mark.parametrize(
    "dtype,scale", [(np.uint8, 1), (np.uint16, 256), (np.float32, 1 / 256), (np.float64, 1 / 256)]
)
@pytest.mark.parametrize("encoding", ["gray", "rgb", "bgr", "rgba", "bgra"])
def test_region_gain_preserves_defect_samples_depth_color_alpha_and_payload(dtype, scale, encoding):
    channels = len(encoding) if encoding != "gray" else 0
    base = (
        np.full((40, 64), 120 * scale, dtype)
        if not channels
        else np.tile(
            np.array([120, 80, 40] + ([99] if channels == 4 else []), dtype=np.float64) * scale,
            (40, 64, 1),
        ).astype(dtype)
    )
    clear = base.copy()
    clear[18:21, 26:29, ...] = (clear[18:21, 26:29, ...].astype(np.float64) / 2).astype(dtype)
    if channels == 4:
        clear[..., 3] = base[..., 3]
    shadowed = clear.copy()
    if channels:
        shadowed[8:34, 12:54, :3] = clear[8:34, 12:54, :3] / 2
    else:
        shadowed[8:34, 12:54] = clear[8:34, 12:54] / 2
    source = ImageArray(shadowed, encoding)
    reference = ImageArray(
        base[..., :3] if channels == 4 else base,
        encoding.removesuffix("a") if channels == 4 else encoding,
    )
    original, saved_reference = source.copy(), reference.copy()
    ticket = TicketClock().issue("camera", mode="external", value="part-42")
    context = {
        "spatial": frame([64, 40], "camera"),
        "ticket": ticket,
        "pyramid": {
            "level": 1,
            "largest_size": [128, 80],
            "to_largest": [[2, 0, 0.5], [0, 2, 0.5], [0, 0, 1]],
        },
    }
    block = tool()
    result = run(
        block,
        {
            "image": ValuePacket(source, context),
            "reference": ValuePacket(reference, {"spatial": frame([64, 40], "reference")}),
        },
    )
    assert result.passed and result.outputs["valid"]
    output = result.outputs["image"]
    assert image_format(output) == encoding and output.dtype == dtype
    np.testing.assert_array_equal(output, clear)
    np.testing.assert_array_equal(source, original)
    np.testing.assert_array_equal(reference, saved_reference)
    expected = np.zeros((40, 64), np.uint8)
    expected[8:34, 12:54] = 255
    np.testing.assert_array_equal(result.outputs["shadow_mask"], expected)
    np.testing.assert_array_equal(result.outputs["gain_map"], np.where(expected, 2.0, 1.0))
    assert not result.outputs["invalid_mask"].any() and not result.outputs["clipped_mask"].any()
    assert result.outputs["measurements"]["components"][0]["gain"] == 2
    assert result.payloads["image"]["ticket"] == ticket
    assert result.payloads["image"]["spatial"] == context["spatial"]
    assert result.payloads["image"]["pyramid"] == context["pyramid"]
    if channels == 4:
        np.testing.assert_array_equal(output[..., 3], original[..., 3])


def test_unchanged_reference_is_a_valid_identity_not_a_final_inspection_judgment():
    image = np.full((30, 40), 100, np.uint8)
    image[8:11, 10:13] = 50  # Isolated dark defect is below minimum shadow area.
    result = run(tool(), {"image": image, "reference": np.full_like(image, 100)})
    assert result.passed and result.final_judgment is None
    assert result.outputs["measurements"]["status"] == "no_candidates"
    np.testing.assert_array_equal(result.outputs["image"], image)
    assert not result.outputs["shadow_mask"].any()


def test_chromatic_material_change_is_not_classified_as_neutral_shadow():
    reference = ImageArray(np.full((40, 50, 3), 120, np.uint8), "rgb")
    image = reference.copy()
    image[5:35, 8:42] = [30, 60, 90]
    result = run(tool(), {"image": image, "reference": reference})
    assert result.passed and not result.outputs["shadow_mask"].any()
    np.testing.assert_array_equal(result.outputs["image"], image)


def test_bounded_hole_fill_corrects_colored_defect_without_replacing_its_samples():
    reference = ImageArray(np.full((40, 50, 3), 120, np.uint8), "rgb")
    image = reference.copy()
    image[5:35, 8:42] = 60
    image[18:21, 24:27] = [20, 30, 40]
    result = run(tool(fill_holes_area=9), {"image": image, "reference": reference})
    assert result.passed and result.outputs["measurements"]["filled_pixels"] == 9
    np.testing.assert_array_equal(
        result.outputs["image"][18:21, 24:27], np.tile([40, 60, 80], (3, 3, 1))
    )
    result = run(tool(fill_holes_area=8), {"image": image, "reference": reference})
    assert result.passed and result.outputs["measurements"]["filled_pixels"] == 0
    np.testing.assert_array_equal(result.outputs["image"][18:21, 24:27], image[18:21, 24:27])


def test_work_selection_and_fit_mask_have_different_roles():
    reference = np.full((40, 50), 120, np.uint8)
    image = reference // 2
    selection = np.zeros_like(image)
    selection[4:36, 6:44] = 255
    fit = np.zeros_like(image)
    fit[10:16, 10:16] = 255
    block = tool()
    result = run(
        block, {"image": image, "reference": reference, "mask": selection, "fit_mask": fit}
    )
    assert result.passed and result.outputs["measurements"]["components"][0]["fit_pixels"] == 36
    expected = np.where(selection, reference, image)
    np.testing.assert_array_equal(result.outputs["image"], expected)
    assert np.count_nonzero(result.outputs["shadow_mask"]) > 36
    # Excluded pixels inside a shadow are not filled through the selection boundary.
    selection[20:22, 20:22] = 0
    result = run(
        tool(fill_holes_area=100), {"image": image, "reference": reference, "mask": selection}
    )
    assert result.passed and not result.outputs["shadow_mask"][20:22, 20:22].any()


@pytest.mark.parametrize(
    "parameters,fit_mask,reason",
    [
        ({"max_gain": 1.5}, None, "gain_limit"),
        ({}, np.zeros((30, 40), np.uint8), "insufficient_fit"),
    ],
)
def test_unqualified_component_withholds_image_and_gain_but_keeps_diagnostics(
    parameters, fit_mask, reason
):
    block = tool(**parameters)
    inputs = {
        "image": np.full((30, 40), 50, np.uint8),
        "reference": np.full((30, 40), 100, np.uint8),
    }
    if fit_mask is not None:
        inputs["fit_mask"] = fit_mask
    result = run(block, inputs)
    assert not result.passed and not result.outputs["valid"]
    assert result.outputs["image"] is NO_VALUE and result.outputs["gain_map"] is NO_VALUE
    assert result.outputs["invalid_mask"].all() and result.outputs["shadow_mask"].all()
    assert result.outputs["measurements"]["components"][0]["status"] == reason
    assert result.fail_reasons[0]["code"] == "shadow_" + reason
    assert block.current_emissions["image"] == [] and block.current_emissions["gain_map"] == []


def test_nonuniform_attenuation_is_rejected_instead_of_flattened_into_reference():
    reference = np.full((40, 40), 100, np.uint8)
    image = np.full_like(reference, 40)
    image[:, 20:] = 70
    result = run(tool(), {"image": image, "reference": reference})
    assert not result.passed
    assert result.outputs["measurements"]["components"][0]["status"] == "inconsistent_attenuation"
    assert result.outputs["image"] is NO_VALUE


@pytest.mark.parametrize("policy", ["reject", "clip"])
def test_clipping_requires_explicit_policy_and_reports_exact_pixels(policy):
    reference = np.full((30, 40), 250, np.uint8)
    image = np.full_like(reference, 70)
    image[15, 20] = 200
    result = run(tool(clip_policy=policy), {"image": image, "reference": reference})
    assert result.outputs["measurements"]["clipped_pixels"] == 1
    assert result.outputs["clipped_mask"][15, 20] == 255
    assert result.passed == (policy == "clip")
    if policy == "clip":
        assert result.outputs["image"][15, 20] == 255
        assert result.outputs["image"][0, 0] == 250
    else:
        assert result.outputs["image"] is NO_VALUE


@pytest.mark.parametrize(
    "reference_value,current_value", [(0.0, 0.1), (1e-310, 1e300), (1e300, 1e-310)]
)
def test_undefined_or_unrepresentable_ratios_are_not_silent_identity(
    reference_value, current_value
):
    reference = np.full((30, 40), reference_value, np.float64)
    image = np.full_like(reference, current_value)
    result = run(tool(min_reference=0), {"image": image, "reference": reference})
    assert not result.passed and result.outputs["invalid_mask"].all()
    assert result.outputs["image"] is NO_VALUE
    assert result.fail_reasons[0]["code"] == "shadow_invalid_reference"


def test_gray_float_hdr_units_do_not_normalize_or_clip_to_one():
    reference = np.full((30, 40), 1e250, np.float64)
    source = reference / 2
    source[10:13, 10:13] *= 0.5
    expected = source * 2
    result = run(tool(min_reference=1e200), {"image": source, "reference": reference})
    assert result.passed
    np.testing.assert_array_equal(result.outputs["image"], expected)
    result = run(tool(min_reference=0), {"image": source * 1e-300, "reference": reference * 1e-300})
    assert result.passed
    np.testing.assert_array_equal(result.outputs["image"], expected * 1e-300)


def test_reference_alignment_and_selection_frames_are_checked():
    a = {"spatial": frame([40, 30], "camera")}
    b = {"spatial": frame([40, 30], "other")}
    image = ValuePacket(np.full((30, 40), 50, np.uint8), a)
    reference = ValuePacket(np.full((30, 40), 100, np.uint8), b)
    block = tool(reference_alignment="same_frame")
    assert block.execute({"image": image, "reference": reference}) is None
    assert "Reference coordinate frame" in block.error
    block = tool()
    assert run(block, {"image": image, "reference": reference}).passed
    for name in ("mask", "fit_mask"):
        assert (
            block.execute(
                {
                    "image": image,
                    "reference": reference,
                    name: ValuePacket(np.full((30, 40), 255, np.uint8), b),
                }
            )
            is None
        )
        assert "Selection coordinate frame" in block.error


@pytest.mark.parametrize(
    "parameters",
    [
        {"min_ratio": 0},
        {"min_ratio": 0.8, "max_ratio": 0.7},
        {"max_ratio": 1},
        {"fit_erosion": -1},
        {"max_gain": 65},
        {"connectivity": 6},
        {"clip_policy": "ignore"},
    ],
)
def test_parameter_limits(parameters):
    block = tool(**parameters)
    assert (
        block.execute(
            {
                "image": np.full((30, 40), 50, np.uint8),
                "reference": np.full((30, 40), 100, np.uint8),
            }
        )
        is None
    )
    assert block.error_kind == "invalid"


def test_empty_selection_and_invalid_reference_do_not_emit_a_corrected_image():
    image = np.full((30, 40), 50, np.uint8)
    ref = image * 2
    result = run(tool(), {"image": image, "reference": ref, "mask": np.zeros_like(image)})
    assert not result.passed and result.outputs["image"] is NO_VALUE
    assert result.fail_reasons[0]["code"] == "shadow_empty_selection"
    ref[12:15, 12:15] = 0
    result = run(tool(fill_holes_area=100), {"image": image, "reference": ref})
    assert not result.passed and (result.outputs["invalid_mask"][12:15, 12:15] == 255).all()
    assert not result.outputs["shadow_mask"][12:15, 12:15].any()


@pytest.mark.parametrize("kind", ["depth", "shape", "negative", "nonfinite", "premultiplied"])
def test_invalid_image_reference_contracts_are_failsafe(kind):
    image = ImageArray(np.full((30, 40, 4), 50, np.float32), "rgba")
    reference = image.copy()
    reference[..., :3] *= 2
    if kind == "depth":
        reference = reference.astype(np.float64)
    elif kind == "shape":
        reference = reference[:, :39]
    elif kind == "negative":
        reference[0, 0, 0] = -1
    elif kind == "nonfinite":
        image[0, 0, 0] = np.nan
    else:
        reference.metadata["alpha_representation"] = "premultiplied"
    block = tool()
    assert block.execute({"image": image, "reference": reference}) is None
    assert block.state == "error" and not block.current_emissions


@pytest.mark.parametrize("connectivity", [4, 8])
def test_circle_selection_and_binary_fit_mask_validation(connectivity):
    image = np.full((40, 50), 50, np.uint8)
    region = {
        "model": "image_region_v1",
        "image_size": [50, 40],
        "shape": "circle",
        "center": [25, 20],
        "radius": 12,
    }
    result = run(
        tool(connectivity=connectivity), {"image": image, "reference": image * 2, "region": region}
    )
    assert result.passed
    expected = np.zeros_like(image)
    cv2.circle(expected, (25, 20), 12, 255, -1)
    np.testing.assert_array_equal(result.outputs["shadow_mask"], expected)
    np.testing.assert_array_equal(result.outputs["image"], np.where(expected, 100, 50))
    block = tool()
    assert (
        block.execute({"image": image, "reference": image * 2, "fit_mask": image.astype(np.uint16)})
        is None
    )
    assert "Mask must be uint8" in block.error


def test_bounded_work_and_component_limits(monkeypatch):
    image = np.full((30, 40), 50, np.uint8)
    reference = image * 2
    monkeypatch.setattr("opencv_dataflow.shadow_compensation.MAX_ROI_WORK", 1)
    block = tool()
    assert block.execute({"image": image, "reference": reference}) is None
    assert "work budget" in block.error
    monkeypatch.setattr("opencv_dataflow.shadow_compensation.MAX_ROI_WORK", 10000)
    image[:, 19:21] = reference[:, 19:21]
    block = tool(maximum_components=1)
    assert block.execute({"image": image, "reference": reference}) is None
    assert "Too many shadow components" in block.error


def test_many_rejections_have_bounded_failure_codes_and_complete_component_reports():
    reference = np.full((6, 104), 100, np.uint8)
    image = reference.copy()
    for index in range(33):
        image[2:4, 1 + index * 3 : 3 + index * 3] = 50
    result = run(
        tool(minimum_area=4, minimum_fit_pixels=1, fit_erosion=0, max_gain=1.5),
        {"image": image, "reference": reference},
    )
    assert not result.passed and result.outputs["image"] is NO_VALUE
    assert len(result.outputs["measurements"]["components"]) == 33
    assert len(result.fail_reasons) == 1
    assert result.fail_reasons[0]["message"].startswith("33 shadow component(s)")


def test_xml_roundtrip_preserves_configuration_ports_and_layout(tmp_path):
    graph = Graph()
    block = graph.add(tool(connectivity=4, fill_holes_area=16, clip_policy="clip"))
    block.description = "Reference shadow compensation"
    block.position = [150, 70]
    block.node_size = "large"
    block.exposed_inputs = ["image", "reference", "fit_mask"]
    block.exposed_outputs = ["image", "valid", "shadow_mask", "measurements"]
    try:
        data = graph.to_dict()
        export_project_xml(data, tmp_path / "shadow.xml")
        restored = Graph.from_dict(read_project_xml(tmp_path / "shadow.xml"), builtin_registry())
        try:
            assert restored.to_dict() == data
        finally:
            restored.close()
    finally:
        graph.close()


def test_graphics_off_skips_drawable_creation_without_changing_pixels(monkeypatch):
    graph = Graph()
    block = graph.add(tool())
    graph.set_graphics_mode("off")
    monkeypatch.setattr(
        "opencv_dataflow.shadow_compensation.graphic_box",
        lambda *args: pytest.fail("Graphics must be skipped"),
    )
    result = run(
        block,
        {"image": np.full((30, 40), 50, np.uint8), "reference": np.full((30, 40), 100, np.uint8)},
    )
    assert result.passed and not result.drawings
    np.testing.assert_array_equal(result.outputs["image"], 100)
    graph.close()


def test_public_projects_have_analytic_pixel_and_failure_evidence():
    folder = Path("examples/custom tools/reference_shadows")
    truth = json.loads((folder / "assets/truth.json").read_text(encoding="utf-8"))
    clear = cv2.imread(str(folder / "assets/clear.png"), cv2.IMREAD_UNCHANGED)[..., ::-1]
    nominal = cv2.imread(str(folder / "assets/reference.png"), cv2.IMREAD_UNCHANGED)[..., ::-1]
    difference = nominal.astype(np.float64) - clear
    np.testing.assert_array_equal(
        difference[43:49, 76:84].mean(axis=(0, 1)), truth["reference_minus_clear_defect_mean_rgb"]
    )
    assert np.count_nonzero(np.any(difference, axis=2)) == truth["defect_pixels"]
    for profile in ("debug", "lean"):
        graph = Graph.load(folder / "project.json", builtin_registry())
        graph.set_run_profile(profile)
        try:
            for index, ticket in enumerate(("clear.png", "shadow.png")):
                graph.run()
                block = graph.blocks["shadows"]
                assert block.last_result is not None, block.error
                result = block.last_result
                assert result.passed
                np.testing.assert_array_equal(result.outputs["image"], clear)
                assert np.count_nonzero(result.outputs["shadow_mask"]) == (
                    truth["shadow_pixels"] if index else 0
                )
                assert result.payloads["image"]["ticket"]["value"] == ticket
        finally:
            graph.close()
    graph = Graph.load(folder / "invalid_fit.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["shadows"]
        result = block.last_result
        assert result is not None, block.error
        assert not result.passed and result.outputs["image"] is NO_VALUE
        assert (
            result.outputs["measurements"]["components"][0]["status"] == "inconsistent_attenuation"
        )
    finally:
        graph.close()
