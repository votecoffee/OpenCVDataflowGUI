"""Native color edits, equal-gray separation and explicit degenerate-map handling."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.block_xml import save_block_xml
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.project_xml import read_block_xml
from opencv_dataflow.tickets import TicketClock, ticket_key

EXAMPLES = Path(__file__).resolve().parents[1] / "examples/opencv tools"
KINDS = ("cv.color_change", "cv.decolor")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("encoding", ["bgr", "rgb", "bgra", "rgba"])
def test_native_parity_and_intersected_selection_keep_order_alpha_payloads_and_inputs(
    kind, encoding
):
    values = np.random.default_rng(61).integers(10, 240, (40, 48, 3), np.uint8)
    if encoding.endswith("a"):
        values = np.dstack((values, np.arange(40 * 48, dtype=np.uint8).reshape(40, 48)))
    context = {
        "spatial": frame([48, 40], "capture"),
        "ticket": TicketClock().issue("camera"),
        "acquisition": {"encoder": 4711},
        "custom": {"lot": "A"},
        "alpha_representation": "straight",
    }
    image = ImageArray(values, encoding, context)
    mask = np.zeros((40, 48), np.uint8)
    mask[4:36, 5:43] = 255
    mask[15:23, 18:28] = 0
    original, original_mask = image.copy(), mask.copy()
    region = {
        "model": "image_region_v1",
        "image_size": [48, 40],
        "shape": "rectangle",
        "rectangle": [9, 8, 31, 26],
    }
    selected_mask = mask.copy()
    selected_mask[:8] = selected_mask[34:] = 0
    selected_mask[:, :9] = selected_mask[:, 40:] = 0
    selected = selected_mask != 0
    native = np.ascontiguousarray(
        values[..., :3][..., ::-1] if encoding.startswith("rgb") else values[..., :3]
    )
    parameters = {"red_mul": 1.4, "green_mul": 0.8, "blue_mul": 0.6} if kind == KINDS[0] else {}
    block = builtin_registry().create(kind, parameters=parameters)
    result = block.execute({"image": image, "mask": mask, "region": region})
    assert result is not None and result.passed, block.error
    if kind == KINDS[0]:
        expected = cv2.colorChange(native, selected_mask.copy(), **parameters)
        if encoding.startswith("rgb"):
            expected = expected[..., ::-1]
        actual = result.outputs["image"]
        np.testing.assert_array_equal(actual[..., :3][selected], expected[selected])
        np.testing.assert_array_equal(actual[~selected], image[~selected])
        assert image_format(actual) == encoding
        if encoding.endswith("a"):
            np.testing.assert_array_equal(actual[..., 3], image[..., 3])
    else:
        expected, boost = cv2.decolor(native)
        baseline = cv2.cvtColor(native, cv2.COLOR_BGR2GRAY)
        actual = result.outputs["image"]
        assert image_format(actual) == "gray"
        np.testing.assert_array_equal(actual[selected], expected[selected])
        np.testing.assert_array_equal(actual[~selected], baseline[~selected])
        np.testing.assert_array_equal(result.outputs["ordinary_gray"], baseline)
        enhanced = result.outputs["color_boost"]
        if encoding.startswith("rgb"):
            boost = boost[..., ::-1]
        np.testing.assert_array_equal(enhanced[..., :3][selected], boost[selected])
        np.testing.assert_array_equal(enhanced[~selected], image[~selected])
        assert image_format(enhanced) == encoding
        if encoding.endswith("a"):
            np.testing.assert_array_equal(enhanced[..., 3], image[..., 3])
        np.testing.assert_array_equal(result.outputs["change"], cv2.absdiff(actual, baseline))
        assert result.outputs["report"]["mapping_context"] == "full_image"
        assert "alpha_representation" not in result.payloads["image"]
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(mask, original_mask)
    np.testing.assert_array_equal(result.outputs["selected_mask"], selected_mask)
    for payload in result.payloads.values():
        assert payload["spatial"] == context["spatial"]
        assert payload["acquisition"] == context["acquisition"]
        assert payload["custom"] == context["custom"]
        assert ticket_key(payload["ticket"]) == ticket_key(context["ticket"])
    json.dumps(result.outputs["report"], allow_nan=False)


def test_selected_color_shift_matches_independent_step_solution_and_boundary_positions():
    folder = EXAMPLES / "masked_color_change"
    truth = json.loads((folder / "truth.json").read_text())
    graph = Graph.load(folder / "masked_color_change.json", builtin_registry())
    try:
        graph.run()
        assert all(b.state == "pass" for b in graph.processing_blocks())
        result = graph.blocks["edit"].last_result
        output = result.outputs["image"]
        x, y, w, h = truth["measurement_rectangle"]
        measured = output[y : y + h, x : x + w].mean((0, 1))
        np.testing.assert_allclose(
            measured, truth["expected_center_bgr"], atol=truth["maximum_center_error_dn"]
        )
        profile = output[y : y + h, :, 2].astype(float).mean(0)
        assert abs(np.argmax(np.diff(profile)) + 1 - truth["left_edge_x"]) <= 1
        assert abs(np.argmin(np.diff(profile)) + 1 - truth["right_edge_x"]) <= 1
        original = cv2.imread(str(folder / "surface.png"))
        np.testing.assert_array_equal(
            output[result.outputs["selected_mask"] == 0],
            original[result.outputs["selected_mask"] == 0],
        )
    finally:
        graph.close()


def test_equal_ordinary_gray_patches_are_separated_and_measured_in_graph():
    folder = EXAMPLES / "contrast_preserving_gray"
    truth = json.loads((folder / "truth.json").read_text())
    source = cv2.imread(str(folder / "patches.png"))
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    graph = Graph.load(folder / "contrast_preserving_gray.json", builtin_registry())
    try:
        graph.run()
        result = graph.blocks["decolor"].last_result.outputs["image"]
        means = []
        for name in ("patch_a_rectangle", "patch_b_rectangle"):
            x, y, w, h = truth[name]
            assert np.all(gray[y : y + h, x : x + w] == truth["ordinary_gray_dn"])
            means.append(float(result[y : y + h, x : x + w].mean()))
        delta = abs(means[1] - means[0])
        assert delta >= truth["minimum_decolor_separation_dn"]
        assert graph.blocks["before"].last_result.outputs["value"] == 0
        assert graph.blocks["after"].last_result.outputs["value"] == pytest.approx(delta)
    finally:
        graph.close()


def test_degenerate_mapping_fails_without_primary_output_or_downstream_execution():
    folder = EXAMPLES / "contrast_preserving_gray"
    graph = Graph.load(folder / "degenerate_mapping.json", builtin_registry())
    try:
        graph.run()
        block = graph.blocks["decolor"]
        assert block.state == "fail", block.error
        result = block.last_result
        assert result.outputs["report"]["status"] == "degenerate_native_mapping"
        assert result.outputs["report"]["native_gray_range"] == 0
        assert all(result.outputs[key] is NO_VALUE for key in ("image", "color_boost", "change"))
        assert not block.current_emissions.get("image")
        assert result.outputs["ordinary_gray"].shape == (128, 192)
        assert graph.blocks["after"].last_result is None
    finally:
        graph.close()


@pytest.mark.parametrize("case", ["empty_selection", "uniform_color", "achromatic_input"])
def test_noop_cases_avoid_native_and_preserve_baseline(monkeypatch, case):
    monkeypatch.setattr(cv2, "decolor", lambda *a, **k: pytest.fail("unnecessary native call"))
    source = ImageArray(np.full((24, 32, 3), [80, 110, 150], np.uint8), "bgr")
    if case == "achromatic_input":
        source[:] = np.arange(32, dtype=np.uint8)[None, :, None]
    inputs = {"image": source}
    if case == "empty_selection":
        inputs["mask"] = np.zeros((24, 32), np.uint8)
    block = builtin_registry().create("cv.decolor")
    result = block.execute(inputs)
    assert result is not None and result.passed, block.error
    np.testing.assert_equal(result.outputs["image"], cv2.cvtColor(source, cv2.COLOR_BGR2GRAY))
    np.testing.assert_equal(result.outputs["color_boost"], source)
    assert not result.outputs["change"].any()
    assert result.outputs["report"]["skip_reason"] == case


@pytest.mark.parametrize(
    "parameter,value", [("red_mul", 0.49), ("green_mul", 2.51), ("blue_mul", float("nan"))]
)
def test_invalid_color_controls_fail_before_native(monkeypatch, parameter, value):
    monkeypatch.setattr(cv2, "colorChange", lambda *a, **k: pytest.fail("invalid native call"))
    block = builtin_registry().create(KINDS[0], parameters={parameter: value})
    assert block.execute({"image": np.zeros((24, 32, 3), np.uint8)}) is None
    assert block.error_kind == "invalid" and not block.current_emissions


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("case", ["depth", "premult", "frame"])
def test_bad_depth_alpha_and_coordinate_frames_fail_before_native(monkeypatch, kind, case):
    name = "decolor" if kind == KINDS[1] else "colorChange"
    monkeypatch.setattr(cv2, name, lambda *a, **k: pytest.fail("invalid native call"))
    image = ImageArray(
        np.zeros((24, 32, 3), np.uint8), "bgr", {"spatial": frame([32, 24], "camera")}
    )
    inputs = {"image": image}
    if case == "depth":
        inputs["image"] = image.astype(np.uint16)
    elif case == "premult":
        inputs["image"] = ImageArray(
            np.zeros((24, 32, 4), np.uint8), "rgba", {"alpha_representation": "premultiplied"}
        )
    else:
        inputs["mask"] = ValuePacket(
            np.ones((24, 32), np.uint8), {"spatial": frame([32, 24], "other")}
        )
    block = builtin_registry().create(kind)
    assert block.execute(inputs) is None
    assert block.error_kind == "invalid" and not block.current_emissions


@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize(
    "folder,project,expected",
    [
        ("masked_color_change", "masked_color_change", "pass"),
        ("contrast_preserving_gray", "contrast_preserving_gray", "pass"),
        ("contrast_preserving_gray", "degenerate_mapping", "fail"),
    ],
)
def test_examples_profiles_graphics_and_xml_presets(profile, folder, project, expected, tmp_path):
    graph = Graph.load(EXAMPLES / folder / (project + ".json"), builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        block = graph.blocks["edit" if folder == "masked_color_change" else "decolor"]
        assert block.state == expected, block.error
        assert bool(block.last_result.drawings) == (profile == "debug")
        path = tmp_path / ("preset." + block.type_id + ".xml")
        save_block_xml(block, path)
        assert read_block_xml(path) == block.to_project_item()
    finally:
        graph.close()
