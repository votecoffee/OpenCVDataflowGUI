"""Native parameter parity, independent surface truth, selections and payload ownership."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.block_xml import save_block_xml
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.project_xml import read_block_xml
from opencv_dataflow.tickets import TicketClock, ticket_key

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/masked_photo_correction"
KINDS = ("cv.illumination_change", "cv.texture_flattening")


def binding(kind):
    return getattr(cv2, "illuminationChange" if kind == KINDS[0] else "textureFlattening")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("encoding", ["bgr", "rgb", "bgra", "rgba"])
def test_native_parity_preserves_color_alpha_holes_source_and_ticket(kind, encoding):
    color = np.random.default_rng(23).integers(20, 230, (40, 48, 3), np.uint8)
    values = (
        np.dstack((color, np.arange(40 * 48, dtype=np.uint8).reshape(40, 48)))
        if encoding.endswith("a")
        else color
    )
    ticket = TicketClock().issue("camera")
    context = {"spatial": frame([48, 40], "camera"), "ticket": ticket, "custom": {"lot": "A"}}
    if encoding.endswith("a"):
        context["alpha_representation"] = "straight"
    source = ImageArray(values, encoding, context)
    mask = np.zeros((40, 48), np.uint8)
    mask[4:36, 5:43] = 255
    mask[13:22, 18:27] = 0
    original_source, original_mask = source.copy(), mask.copy()
    native_color = np.ascontiguousarray(color[..., ::-1] if encoding.startswith("rgb") else color)
    block = builtin_registry().create(kind)
    expected = binding(kind)(native_color, mask.copy(), **block.parameters)
    if encoding.startswith("rgb"):
        expected = expected[..., ::-1]
    result = block.execute({"image": source, "mask": mask})
    assert result is not None, block.error
    output = result.outputs["image"]
    selected = mask != 0
    assert image_format(output) == encoding and output.dtype == np.uint8
    np.testing.assert_array_equal(output[..., :3][selected], expected[selected])
    np.testing.assert_array_equal(output[~selected], source[~selected])
    if encoding.endswith("a"):
        np.testing.assert_array_equal(output[..., 3], source[..., 3])
    np.testing.assert_array_equal(source, original_source)
    np.testing.assert_array_equal(mask, original_mask)
    np.testing.assert_array_equal(result.outputs["selected_mask"], mask)
    np.testing.assert_array_equal(
        result.outputs["gradient_mask"], cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=3)
    )
    difference = np.abs(output[..., :3].astype(int) - source[..., :3]).max(axis=2)
    np.testing.assert_array_equal(result.outputs["change"], difference)
    assert result.outputs["report"]["changed_pixels"] == np.count_nonzero(difference)
    for name in ("image", "selected_mask", "gradient_mask", "change", "report"):
        payload = result.payloads[name]
        assert payload["spatial"] == context["spatial"] and payload["custom"] == context["custom"]
        assert ticket_key(payload["ticket"]) == ticket_key(ticket)
    assert "alpha_representation" not in result.payloads["change"]
    assert result.drawings
    json.dumps(result.outputs["report"], allow_nan=False)


@pytest.mark.parametrize("aperture", [3, 5, 7])
def test_all_exposed_canny_apertures_match_native(aperture):
    image = cv2.imread(str(EXAMPLE / "surface.png"))
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[12:116, 16:176] = 255
    block = builtin_registry().create(
        KINDS[1],
        parameters={
            "kernel_size": aperture,
            "low_threshold": 50,
            "high_threshold": 150,
        },
    )
    result = block.execute({"image": image, "mask": mask})
    assert result is not None, block.error
    expected = cv2.textureFlattening(image, mask.copy(), **block.parameters)
    np.testing.assert_equal(result.outputs["image"][mask != 0], expected[mask != 0])


@pytest.mark.parametrize("kind", KINDS)
def test_mask_and_editable_region_intersect_before_native_solve(kind):
    image = cv2.imread(str(EXAMPLE / "surface.png"))
    region = {
        "model": "image_region_v1",
        "image_size": [192, 128],
        "shape": "polygon",
        "vertices": [[20, 20], [150, 20], [20, 100]],
    }
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[30:110, 30:160] = 1
    expected_mask = np.zeros_like(mask)
    cv2.fillPoly(expected_mask, [np.asarray(region["vertices"], np.int32)], 255)
    expected_mask[mask == 0] = 0
    block = builtin_registry().create(kind)
    result = block.execute({"image": image, "mask": mask, "region": region})
    assert result is not None, block.error
    np.testing.assert_equal(result.outputs["selected_mask"], expected_mask)
    np.testing.assert_equal(result.outputs["image"][expected_mask == 0], image[expected_mask == 0])


@pytest.mark.parametrize("kind", KINDS)
def test_known_boundary_survives_reduced_variation_or_texture(kind):
    truth = json.loads((EXAMPLE / "truth.json").read_text())
    image = cv2.imread(str(EXAMPLE / "surface.png"))
    region = {
        "model": "image_region_v1",
        "image_size": truth["image_size"],
        "shape": "rectangle",
        "rectangle": truth["selected_rectangle"],
    }
    parameters = {} if kind == KINDS[0] else {"low_threshold": 50, "high_threshold": 150}
    block = builtin_registry().create(kind, parameters=parameters)
    result = block.execute({"image": image, "region": region})
    assert result is not None, block.error
    output = result.outputs["image"]
    y0, y1 = truth["profile_y_range"]
    x0, x1 = truth["profile_x_range"]
    before = image[y0:y1, :, 0].astype(float).mean(axis=0)
    after = output[y0:y1, :, 0].astype(float).mean(axis=0)
    step = truth["step_boundary_x"]
    local_derivative = np.diff(after)[step - 8 : step + 7]
    measured = int(np.argmax(local_derivative)) + step - 8 + 1
    assert abs(measured - step) <= truth["boundary_tolerance_pixels"]
    if kind == KINDS[0]:
        assert np.ptp(after[x0:x1]) < truth["illumination_max_span_ratio"] * np.ptp(before[x0:x1])
        assert local_derivative.max() >= truth["illumination_min_step_dn"]
    else:
        x, y, w, h = truth["texture_rectangle"]

        def residual(gray):
            values = gray.astype(float)
            highpass = values - cv2.GaussianBlur(values, (0, 0), 2)
            return np.std(highpass[y : y + h, x : x + w])

        assert residual(output[..., 0]) < truth["texture_max_rms_ratio"] * residual(image[..., 0])
        assert local_derivative.max() >= truth["texture_min_step_dn"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("thin", [False, True])
def test_empty_or_eroded_away_selection_is_exact_noop_without_native(monkeypatch, kind, thin):
    image = cv2.imread(str(EXAMPLE / "surface.png"))
    mask = np.zeros(image.shape[:2], np.uint8)
    if thin:
        mask[20:90, 30:34] = 255
    monkeypatch.setattr(
        cv2, binding(kind).__name__, lambda *a, **k: pytest.fail("native noop call")
    )
    block = builtin_registry().create(kind)
    result = block.execute({"image": image, "mask": mask})
    assert result is not None, block.error
    np.testing.assert_equal(result.outputs["image"], image)
    assert not result.outputs["change"].any()
    assert result.outputs["report"]["skip_reason"] == (
        "no_gradient_interior" if thin else "empty_selection"
    )


@pytest.mark.parametrize(
    "kind,parameters",
    [
        (KINDS[0], {"alpha": 2.1}),
        (KINDS[0], {"beta": float("nan")}),
        (KINDS[1], {"low_threshold": 100, "high_threshold": 50}),
        (KINDS[1], {"kernel_size": 4}),
        (KINDS[1], {"high_threshold": float("inf")}),
    ],
)
def test_invalid_controls_fail_before_native_and_emit_nothing(monkeypatch, kind, parameters):
    monkeypatch.setattr(cv2, binding(kind).__name__, lambda *a, **k: pytest.fail("invalid call"))
    block = builtin_registry().create(kind, parameters=parameters)
    assert block.execute({"image": np.zeros((32, 48, 3), np.uint8)}) is None
    assert block.error_kind == "invalid" and not block.current_emissions


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "case", ["depth", "gray", "layout", "small", "budget", "premult", "mask", "frame"]
)
def test_unsupported_image_and_selection_contracts_are_contained(monkeypatch, kind, case):
    image = ImageArray(
        np.zeros((32, 48, 3), np.uint8), "bgr", {"spatial": frame([48, 32], "camera")}
    )
    inputs = {"image": image}
    if case == "depth":
        inputs["image"] = image.astype(np.uint16)
    elif case == "gray":
        inputs["image"] = np.zeros((32, 48), np.uint8)
    elif case == "layout":
        inputs["image"] = ImageArray(image, "rgba")
    elif case == "small":
        inputs["image"] = image[:3]
    elif case == "budget":
        inputs["image"] = np.zeros((1500, 1500, 3), np.uint8)
    elif case == "premult":
        inputs["image"] = ImageArray(
            np.zeros((32, 48, 4), np.uint8), "rgba", {"alpha_representation": "premultiplied"}
        )
    elif case == "mask":
        inputs["mask"] = np.ones((3, 3), np.uint8)
    else:
        inputs["mask"] = ValuePacket(
            np.ones((32, 48), np.uint8), {"spatial": frame([48, 32], "other")}
        )
    monkeypatch.setattr(cv2, binding(kind).__name__, lambda *a, **k: pytest.fail("invalid call"))
    block = builtin_registry().create(kind)
    assert block.execute(inputs) is None
    assert block.error_kind == "invalid" and not block.current_emissions


@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_project_runs_with_profiles_and_xml_preserves_controls(profile, tmp_path):
    graph = Graph.load(EXAMPLE / "masked_photo_correction.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        assert all(b.state == "pass" for b in graph.processing_blocks())
        for ident in ("illumination", "texture"):
            block = graph.blocks[ident]
            assert block.last_result.outputs["report"]["changed_pixels"] > 10000
            assert bool(block.last_result.drawings) == (profile == "debug")
            path = tmp_path / (ident + "." + block.type_id + ".xml")
            save_block_xml(block, path)
            assert read_block_xml(path) == block.to_project_item()
    finally:
        graph.close()
