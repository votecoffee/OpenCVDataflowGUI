"""Native quaternion phase correlation, legal placements and color discrimination."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.color_template import MAX_DFT_PIXELS
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolbox import block_entry

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/color_template"
TYPE = "inspect.match_color_template"


@pytest.fixture(scope="module")
def registry():
    return builtin_registry()


@pytest.mark.parametrize(("encoding", "dtype", "xy"), [
    ("bgr", np.uint8, (0, 0)),
    ("rgb", np.uint16, (50, 12)),
    ("rgba", np.float64, (32, 48)),
    ("rgb", np.int16, (15, 30)),
    ("lab", np.float32, (26, 32)),
])
def test_exact_native_scores_on_padded_grid_and_typed_encodings(registry, encoding, dtype, xy):
    rng = np.random.default_rng(854)
    templ = rng.integers(10, 240, (13, 17, 3), np.uint8).astype(dtype)
    scene = rng.integers(10, 35, (61, 67, 3), np.uint8).astype(dtype)
    x, y = xy
    if encoding == "lab":
        templ = cv2.cvtColor(templ / 255, cv2.COLOR_BGR2Lab)
        scene = cv2.cvtColor(scene / 255, cv2.COLOR_BGR2Lab)
    scene[y:y + 13, x:x + 17] = templ

    def typed(image):
        if encoding in ("rgb", "rgba"):
            image = image[..., ::-1]
        if encoding == "rgba":
            alpha = np.ones(image.shape[:2]) * (0.8 if image.shape[1] > 17 else 0.4)
            image = np.dstack((image, alpha))
        return ImageArray(image, encoding)

    def native(image):
        return cv2.cvtColor(image, cv2.COLOR_Lab2BGR) if encoding == "lab" else image

    block = registry.create(TYPE)
    result = block.execute({"image": typed(scene), "template": typed(templ)})
    assert result and result.passed, block.error
    raw = cv2.ximgproc.colorMatchTemplate(native(scene), native(templ))
    expected = np.array([[raw[-j % raw.shape[0], -i % raw.shape[1]]
                          for i in range(67 - 17 + 1)] for j in range(61 - 13 + 1)])
    assert result.outputs["response"].dtype == np.float64
    np.testing.assert_array_equal(result.outputs["response"], expected)
    assert result.outputs["location"]["top_left"] == [x, y]
    assert result.outputs["center"] == [[x + 8, y + 6]]
    assert result.outputs["score"] == raw[-y % raw.shape[0], -x % raw.shape[1]]
    assert image_format(result.outputs["image"]) == encoding
    np.testing.assert_array_equal(result.outputs["image"], typed(scene))


def test_region_and_mask_intersect_on_template_center_with_source_pose_metadata(registry):
    scene = cv2.imread(str(EXAMPLE / "assets/scene.png"))
    templ = cv2.imread(str(EXAMPLE / "assets/template.png"))
    source_grid = frame([239, 139], "inspection-camera")
    source_grid["to_root"] = [[1, 0, 12], [0, 1, 18], [0, 0, 1]]
    payload = {
        "spatial": source_grid,
        "ticket": TicketClock("color").issue("camera", "fixed", "part-85"),
    }
    crop_mask = np.zeros(scene.shape[:2], np.uint8)
    crop_mask[53, 54] = crop_mask[88, 175] = 255
    region = {
        "model": "image_region_v1", "image_size": [239, 139], "shape": "rectangle",
        "rectangle": [140, 55, 71, 66], "coordinate_context": source_grid,
    }
    match = registry.create(TYPE, parameters={
        "pose_offset_x": 6, "pose_offset_y": -3, "pose_offset_angle": 12, "min_score": 0.35,
    })
    inputs = {
        "image": ValuePacket(scene, payload),
        "template": templ,
        "search_mask": ValuePacket(crop_mask, payload),
        "region": region,
    }
    result = match.execute(inputs)
    assert result and result.passed, match.error
    response_payload = result.payloads["response"]
    assert response_payload["spatial"]["image_size"] == [187, 97]
    assert response_payload["spatial"]["to_root"] == source_grid["to_root"]
    assert response_payload["ticket"] == payload["ticket"]
    assert result.outputs["location"]["bounds"] == [149, 67, 53, 43]
    pose = result.outputs["pose"]
    assert pose["origin"] == [181, 85] and pose["angle_degrees"] == pytest.approx(12)
    assert pose["target_payload"]["ticket"] == payload["ticket"]
    assert result.drawings["match"].source == "image"
    target_lost = {**region, "rectangle": [25, 26, 71, 64]}
    rejected = match.execute({**inputs, "region": target_lost})
    assert rejected and not rejected.passed and rejected.outputs["pose"] is NO_VALUE
    assert rejected.outputs["location"]["top_left"] == [28, 32]
    assert rejected.fail_reasons[0]["code"] == "color_template_score_below_minimum"
    wrong = ValuePacket(crop_mask, {"spatial": frame([239, 139], "other-camera")})
    assert match.execute({**inputs, "search_mask": wrong}) is None
    assert "coordinate frame" in match.error


def test_subpixel_refinement_on_eligible_neighborhood(registry, monkeypatch):
    source = np.full((18, 27, 3), 20, np.uint8)
    templ = np.full((3, 4, 3), 29, np.uint8)
    y, x = np.indices((18, 27))
    raw = np.maximum(0, 0.9 - 0.01 * (((-x % 27) - 8.3)**2 + ((-y % 18) - 7.7)**2))
    monkeypatch.setattr(cv2.ximgproc, "colorMatchTemplate", lambda *args: raw)
    block = registry.create(TYPE, parameters={"subpixel": True})
    result = block.execute({"image": source, "template": templ})
    assert result and result.passed, block.error
    assert result.outputs["location"]["subpixel_top_left"] == pytest.approx([8.3, 7.7])
    assert result.outputs["center"][0] == pytest.approx([9.8, 8.7])
    mask = np.zeros(source.shape[:2], np.uint8)
    mask[9, 10] = 255
    result = block.execute({"image": source, "template": templ, "search_mask": mask})
    assert result and result.passed, block.error
    assert "subpixel_top_left" not in result.outputs["location"]


def test_invalid_dimensions_selection_and_degenerate_spectrum_report_exceptions(registry, monkeypatch):
    block = registry.create(TYPE)
    src = np.zeros((23, 31, 3), np.uint8)
    patch = np.random.default_rng(31).integers(0, 255, (9, 11, 3), np.uint8)
    assert block.execute({"image": src, "template": patch}) is None
    assert "undefined" in block.error
    monkeypatch.setattr(cv2.ximgproc, "colorMatchTemplate", lambda *args: pytest.fail("entered native matcher"))
    for image, template, mask, message in (
        (src, np.zeros((24, 31, 3), np.uint8), None, "fit"),
        (src[..., 0], patch, None, "color channels"),
        (src, patch, np.zeros(src.shape[:2], np.uint8), "no legal template centers"),
        (np.zeros((32, MAX_DFT_PIXELS // 32 + 1, 3), np.uint8), patch, None, "transform limit"),
    ):
        inputs = {"image": image, "template": template}
        if mask is not None:
            inputs["search_mask"] = mask
        assert block.execute(inputs) is None
        assert message in block.error


@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_saved_examples_discriminate_identical_grayscale_patches(registry, profile):
    for filename, expected, xy, native_score in (
        ("compare_color_and_gray", True, [149, 67], 0.39462),
        ("reject_wrong_color", False, [28, 32], 0.30168),
    ):
        graph = Graph.load(EXAMPLE / f"{filename}.json", registry)
        graph.set_run_profile(profile)
        graph.set_branch_workers(3)
        try:
            graph.run()
            color, gray = (graph.blocks[name].last_result for name in ("color", "gray"))
            assert block_entry(graph.blocks["color"]).path[:2] == (
                "OpenCV", "Extended Image Processing (ximgproc)",
            )
            assert color and color.passed == expected
            assert gray and gray.passed
            assert color.outputs["location"]["top_left"] == xy
            assert color.outputs["score"] == pytest.approx(native_score, abs=0.00005)
            assert bool(color.drawings) == bool(color.graphics) == (profile == "debug")
            if expected:
                src = graph.blocks["scene"].last_result.outputs["image"]
                gray_source = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
                a = gray_source[32:75, 28:81]
                b = gray_source[67:110, 149:202]
                np.testing.assert_array_equal(a, b)
                values = gray.outputs["response"]
                assert values[32, 28] == pytest.approx(values[67, 149], abs=1e-6)
                assert color.outputs["response"][32, 28] < 0.26
            else:
                assert color.outputs["pose"] is NO_VALUE
        finally:
            graph.close()
