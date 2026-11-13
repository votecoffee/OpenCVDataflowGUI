"""Source-over arithmetic, alpha association, transparency and payload contracts."""

import json
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.alpha_composition import AlphaOver
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.tickets import TicketClock

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/alpha_composition"


def run(foreground, background, extra=None, **parameters):
    block = builtin_registry().create("cv.alpha_over", parameters=parameters)
    result = block.execute({"foreground": foreground, "background": background, **(extra or {})})
    assert result is not None, block.error
    return block, result


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
@pytest.mark.parametrize(
    "fg_repr,bg_repr",
    [
        ("straight", "straight"),
        ("premultiplied", "straight"),
        ("straight", "premultiplied"),
        ("premultiplied", "premultiplied"),
    ],
)
def test_source_over_matches_an_independent_rational_pixel(dtype, fg_repr, bg_repr):
    maximum = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1
    # Thirds are exact on both uint8/16 scales; the color products are integral too.
    fg = np.array([[[Fraction(3, 5), Fraction(1, 5), Fraction(4, 5), Fraction(1, 3)]]], object)
    bg = np.array([[[Fraction(1, 5), Fraction(2, 5), Fraction(3, 5), Fraction(2, 3)]]], object)
    expected_alpha = fg[0, 0, 3] + bg[0, 0, 3] * (1 - fg[0, 0, 3])
    expected = [
        (fg[0, 0, c] * fg[0, 0, 3] + bg[0, 0, c] * bg[0, 0, 3] * (1 - fg[0, 0, 3])) / expected_alpha
        for c in range(3)
    ] + [expected_alpha]
    if fg_repr == "premultiplied":
        fg[..., :3] *= fg[..., 3:]
    if bg_repr == "premultiplied":
        bg[..., :3] *= bg[..., 3:]
    if maximum != 1:
        foreground = np.asarray([round(v * int(maximum)) for v in fg.flat], dtype).reshape(1, 1, 4)
        background = np.asarray([round(v * int(maximum)) for v in bg.flat], dtype).reshape(1, 1, 4)
        truth = np.array([round(v * int(maximum)) for v in expected], dtype)
    else:
        foreground, background = fg.astype(dtype), bg.astype(dtype)
        truth = np.array(expected, dtype=dtype)
    _, result = run(
        ImageArray(foreground, "rgba"),
        ImageArray(background, "rgba"),
        foreground_representation=fg_repr,
        background_representation=bg_repr,
    )
    np.testing.assert_allclose(
        result.outputs["image"][0, 0], truth, atol=1 if maximum != 1 else 2e-7, rtol=0
    )
    assert result.outputs["alpha"][0, 0] == pytest.approx(float(expected_alpha), abs=1e-7)


def test_zero_alpha_is_finite_and_selection_preserves_hidden_background_colors():
    foreground = ImageArray(np.full((2, 3, 4), [0.8, 0.2, 0.4, 0], np.float64), "rgba")
    background = ImageArray(np.full((2, 3, 4), [0.1, 0.3, 0.9, 0], np.float64), "rgba")
    original = background.copy()
    mask = np.zeros((2, 3), np.uint8)
    mask[0, :2] = 255
    _, result = run(foreground, background, {"mask": mask})
    assert np.all(result.outputs["image"][mask != 0] == 0)
    np.testing.assert_array_equal(result.outputs["image"][mask == 0], original[mask == 0])
    np.testing.assert_array_equal(background, original)
    assert np.isfinite(result.outputs["image"]).all()


def test_external_alpha_and_opacity_multiply_embedded_alpha_and_associated_color():
    fg = ImageArray(
        np.full((2, 4, 4), [0.4, 0.1, 0.0, 0.5], np.float64),
        "rgba",
        {"alpha_representation": "premultiplied"},
    )
    bg = ImageArray(np.full((2, 4, 4), [0.0, 0.0, 1.0, 1.0], np.float64), "rgba")
    matte = np.full((2, 4), 0.5, np.float64)
    _, result = run(fg, bg, {"alpha": matte}, opacity=0.5)
    np.testing.assert_allclose(result.outputs["image"], np.tile([0.1, 0.025, 0.875, 1], (2, 4, 1)))
    np.testing.assert_allclose(result.outputs["foreground_alpha"], 0.125)


def test_association_metadata_prevents_double_multiplication_when_chaining():
    fg = ImageArray(np.array([[[1.0, 0, 0, 0.5]]]), "rgba")
    blank = ImageArray(np.zeros((1, 1, 4), np.float64), "rgba")
    _, first = run(fg, blank, output_representation="premultiplied")
    assert first.payloads["image"]["alpha_representation"] == "premultiplied"
    blue = ImageArray(np.array([[[0.0, 0, 1, 1]]]), "rgba")
    _, second = run(first.outputs["image"], blue)
    np.testing.assert_allclose(second.outputs["image"], [[[0.5, 0, 0.5, 1]]])
    assert second.outputs["info"]["foreground_representation"] == "premultiplied"


def test_mixed_depth_channel_order_gray_and_integer_clipping():
    fg = ImageArray(np.array([[[0, 0, 255, 128]]], np.uint8), "bgra")
    bg = ImageArray(np.array([[[0, 0, 65535]]], np.uint16), "rgb")
    _, result = run(fg, bg)
    np.testing.assert_array_equal(result.outputs["image"], [[[32896, 0, 32639]]])
    assert result.outputs["image"].dtype == np.uint16
    assert result.outputs["image"].color_format == "rgb"
    _, gray = run(
        ImageArray(np.array([[[255, 0, 0, 255]]], np.uint8), "rgba"), np.zeros((1, 1), np.uint8)
    )
    assert gray.outputs["image"][0, 0] == 76
    _, clipped = run(
        ImageArray(np.array([[[2.0, -1, 0.5, 1]]]), "rgba"),
        np.zeros((1, 1, 3), np.uint8),
        output_format="rgb",
    )
    np.testing.assert_array_equal(clipped.outputs["image"], [[[255, 0, 128]]])
    assert clipped.outputs["info"]["clipped_samples"] == 2


def test_linear_srgb_has_the_expected_half_white_value():
    fg = np.full((2, 3, 3), 255, np.uint8)
    bg = np.zeros_like(fg)
    _, stored = run(fg, bg, opacity=0.5)
    _, linear = run(fg, bg, opacity=0.5, blend_space="linear_srgb")
    assert np.all(stored.outputs["image"] == 128)
    assert np.all(linear.outputs["image"] == 188)


def test_selected_frames_tickets_pyramids_and_graphics_optimization():
    spatial = frame([32, 24], "camera")
    spatial.update(root_size=[64, 48], to_root=[[2, 0, 0], [0, 2, 0], [0, 0, 1]])
    ticket = TicketClock().issue("part")
    context = {
        "spatial": spatial,
        "ticket": ticket,
        "encoder": 700,
        "pyramid": {
            "level": 1,
            "largest_size": [64, 48],
            "to_largest": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        },
    }
    fg = ImageArray(np.full((24, 32, 4), [240, 50, 10, 200], np.uint8), "rgba")
    bg = ImageArray(np.full((24, 32, 4), [10, 20, 30, 180], np.uint8), "rgba")
    region = {
        "model": "image_region_v1",
        "image_size": [32, 24],
        "shape": "circle",
        "center": [16, 12],
        "radius": 7,
    }
    mask = np.zeros((24, 32), np.uint8)
    mask[:, :18] = 255
    expected = np.zeros_like(mask)
    cv2.circle(expected, (16, 12), 7, 255, -1)
    selected = (expected != 0) & (mask != 0)
    block, result = run(
        ValuePacket(fg, context), ValuePacket(bg, context), {"region": region, "mask": mask}
    )
    np.testing.assert_array_equal(result.outputs["image"][~selected], bg[~selected])
    assert np.all(result.outputs["foreground_alpha"][~selected] == 0)
    for port in ("image", "alpha", "foreground_alpha", "info"):
        assert result.payloads[port]["ticket"] == ticket
        assert result.payloads[port]["spatial"] == spatial
        assert result.payloads[port]["pyramid"] == context["pyramid"]
        assert result.payloads[port]["encoder"] == 700
    assert "alpha_representation" not in result.payloads["alpha"]
    assert result.drawings["Alpha composition"].from_output
    block.set_graphics_mode("off")
    lean = block.execute(
        {
            "foreground": ValuePacket(fg, context),
            "background": ValuePacket(bg, context),
            "region": region,
            "mask": mask,
        }
    )
    assert lean is not None, block.error
    assert not lean.graphics and not lean.drawings and lean.outputs["preview"] is NO_VALUE
    np.testing.assert_array_equal(lean.outputs["image"], result.outputs["image"])


def test_cross_source_ownership_is_explicit_and_background_wins():
    clock = TicketClock()
    a, b = clock.issue("overlay"), clock.issue("camera")
    fg = ValuePacket(
        np.full((4, 5, 3), 200, np.uint8), {"ticket": a, "spatial": frame([5, 4], "a")}
    )
    bg = ValuePacket(np.full((4, 5, 3), 40, np.uint8), {"ticket": b, "spatial": frame([5, 4], "b")})
    block = AlphaOver(parameters={"alignment": "pixel_grid"})
    assert block.execute({"foreground": fg, "background": bg}) is None
    assert "different tickets" in block.error
    _, result = run(fg, bg, alignment="pixel_grid", ticket_policy="background")
    assert result.payloads["image"]["ticket"] == b
    assert result.payloads["image"]["spatial"]["root_id"] == "b"
    assert result.outputs["info"]["sources"][0]["ticket"] == a


def test_premultiplied_integer_quantization_keeps_zero_alpha_colors_valid_for_chaining():
    fg = ImageArray(np.array([[[0.8, 0.2, 0.3, 1e-4]]], np.float64), "rgba")
    blank = ImageArray(np.zeros((1, 1, 4), np.uint8), "rgba")
    _, result = run(fg, blank, output_representation="premultiplied")
    assert np.all(result.outputs["image"] == 0)
    _, chained = run(result.outputs["image"], blank)
    assert np.all(chained.outputs["image"] == 0)


def test_extreme_associated_hdr_does_not_require_unassociation_for_preview():
    fg = ImageArray(
        np.array([[[1e300, 0, 0, 1e-300]]]), "rgba", {"alpha_representation": "premultiplied"}
    )
    blank = ImageArray(np.zeros((1, 1, 4), np.float64), "rgba")
    _, result = run(fg, blank, output_representation="premultiplied")
    assert result.outputs["image"][0, 0, 0] == 1e300
    assert np.isfinite(result.outputs["preview"]).all()
    block = AlphaOver()
    assert block.execute({"foreground": fg, "background": blank}) is None
    assert "exceed float64" in block.error


@pytest.mark.parametrize(
    "parameters,fg,message",
    [
        ({"opacity": 1.1}, [1, 0, 0, 1], "between"),
        ({}, [1, 0, 0, 2], "[0,1]"),
        ({"foreground_representation": "premultiplied"}, [1, 0, 0, 0], "must be zero"),
        ({"blend_space": "linear_srgb"}, [2, 0, 0, 1], "[0,1]"),
        (
            {"output_format": "rgb", "output_representation": "premultiplied"},
            [1, 0, 0, 1],
            "requires",
        ),
        ({"output_depth": "float32"}, [1e300, 0, 0, 1], "output depth"),
    ],
)
def test_invalid_parameters_and_alpha_are_contained(parameters, fg, message):
    block = AlphaOver(parameters=parameters)
    assert (
        block.execute(
            {
                "foreground": ImageArray(np.array([[fg]], np.float64), "rgba"),
                "background": ImageArray(np.zeros((1, 1, 4), np.float64), "rgba"),
            }
        )
        is None
    )
    assert message in block.error and not block.current_emissions


def test_shape_frame_alpha_map_and_pixel_budget_validation():
    fg, bg = np.ones((8, 10, 3), np.uint8), np.zeros((8, 10, 3), np.uint8)
    for extra, message in (
        ({"foreground": ImageArray(fg, "rgba")}, "channel layout"),
        ({"foreground": fg[:, :-1]}, "matching pixel grids"),
        ({"alpha": np.full((8, 10), 2.0)}, "[0,1]"),
        ({"alpha": np.ones((8, 9), np.uint8)}, "matching dimensions"),
    ):
        block = AlphaOver()
        assert block.execute({"foreground": fg, "background": bg, **extra}) is None
        assert message in block.error
    block = AlphaOver()
    assert (
        block.execute(
            {
                "foreground": ValuePacket(fg, {"spatial": frame([10, 8], "a")}),
                "background": ValuePacket(bg, {"spatial": frame([10, 8], "b")}),
            }
        )
        is None
    )
    assert "different coordinate frames" in block.error
    large = np.zeros((1600, 1600, 3), np.uint8)
    assert block.execute({"foreground": large, "background": large}) is None
    assert "2.1 MP" in block.error


def test_documentation_and_block_fields_roundtrip_through_xml(tmp_path):
    graph = Graph()
    try:
        block = graph.add(AlphaOver(block_id="alpha", parameters={"opacity": 0.7}))
        assert "W3C" in block.documentation
        block.documentation = "<p>Operator documentation.</p>"
        expected = graph.to_dict()
        path = tmp_path / "program.xml"
        export_project_xml(expected, path)
        restored = Graph.from_dict(read_project_xml(path), builtin_registry())
        try:
            assert restored.to_dict() == expected
        finally:
            restored.close()
    finally:
        graph.close()


@pytest.mark.parametrize("graphics", ["on", "off"])
def test_original_example_matches_scalar_rational_reference(graphics):
    graph = Graph.load(EXAMPLE / "project.json", builtin_registry())
    truth = cv2.imread(str(EXAMPLE / "assets/reference.png"), cv2.IMREAD_UNCHANGED)
    try:
        for block in graph.blocks.values():
            block.set_graphics_mode(graphics)
        graph.run()
        assert all(b.state == "pass" for b in graph.processing_blocks())
        for name in ("straight", "premultiplied", "external"):
            result = graph.blocks[name].last_result
            np.testing.assert_allclose(result.outputs["image"], truth, atol=1, rtol=0)
            assert result.outputs["image"].dtype == np.uint16
        json.dumps(graph.to_dict(), allow_nan=False)
    finally:
        graph.close()
