"""Real glyphs, original sample depth, frame ownership and headless replay."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.filmstrip import annotate_filmstrip_image
from opencv_dataflow.graphics_layers import (
    RESULT_GRAPHICS,
    GraphicLayer,
    GraphicsBundle,
    draw_object,
    graphic_object,
)
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import ValuePacket, frame, moved_payload
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.unicode_text import DEFAULT_FONT, TextStyle, text_masks


@pytest.mark.parametrize("text", ["Qualité ±5 µm", "Έλεγχος Δ", "Контроль №2"])
def test_fixture_characters_have_real_distinct_glyphs(text):
    from PIL import ImageFont

    font = ImageFont.truetype(str(DEFAULT_FONT), 24)
    missing = bytes(font.getmask(chr(0x10FFFF)))
    for char in text.replace(" ", ""):
        assert bytes(font.getmask(char)) != missing, char
    mask = text_masks(text, TextStyle(font_size=24))[0]
    assert mask.any()
    escaped = text_masks(text.encode("unicode_escape").decode("ascii"), TextStyle(font_size=24))[0]
    assert mask.shape != escaped.shape or not np.array_equal(mask, escaped)


@pytest.mark.parametrize(
    "dtype,scale", [(np.uint8, 1), (np.uint16, 257), (np.float32, 1 / 255), (np.float64, 1 / 255)]
)
@pytest.mark.parametrize("encoding", ["bgr", "rgb", "bgra", "rgba", "gray"])
def test_unicode_overlay_preserves_samples_alpha_encoding_and_context(dtype, scale, encoding):
    registry = builtin_registry()
    shape = (90, 280) if encoding == "gray" else (90, 280, len(encoding))
    pixels = np.full(shape, 37 * scale, dtype)
    if encoding in ("rgba", "bgra"):
        pixels[..., 3] = 113 * scale
    image = ImageArray(pixels, encoding)
    original = image.copy()
    ticket = TicketClock().issue("source")
    context = {"spatial": frame([280, 90], "source"), "ticket": ticket}
    annotation = registry.create(
        "cv.annotation_layer",
        parameters={
            "shape": "label",
            "label": "Qualité Δ №2",
            "origin": [20, 50],
            "color_rgb": [239, 119, 31],
            "font_size": 24,
            "text_outline": 0,
        },
    )
    result = annotation.execute({"image": ValuePacket(image, context)})
    assert result is not None, annotation.error
    graphic = result.outputs[RESULT_GRAPHICS].layers[0].objects[0]
    assert graphic.text == "Qualité Δ №2" and graphic.text_style.font_size == 24
    overlay = registry.create("cv.overlay_result_graphics")
    rendered = overlay.execute(
        {
            "image": ValuePacket(image, context),
            "graphics_1": result.outputs[RESULT_GRAPHICS],
        }
    )
    assert rendered is not None, overlay.error
    output = rendered.outputs["image"]
    assert output.dtype == dtype
    assert image_format(output) == ("bgr" if encoding == "gray" else encoding)
    assert rendered.payloads["image"]["ticket"] == ticket
    assert rendered.payloads["image"]["spatial"] == context["spatial"]
    np.testing.assert_array_equal(image, original)
    base = np.repeat(original[..., None], 3, axis=2) if encoding == "gray" else original
    mask = text_masks(graphic.text, graphic.text_style)[0]
    support = np.zeros((90, 280), bool)
    top = 50 - mask.shape[0] - 4
    support[top : top + mask.shape[0], 20 : 20 + mask.shape[1]] = mask != 0
    np.testing.assert_array_equal(output[~support], base[~support])
    assert np.any(output[support] != base[support])
    color = [31, 119, 239] if encoding in ("bgr", "bgra", "gray") else [239, 119, 31]
    opaque = np.zeros_like(support)
    opaque[top : top + mask.shape[0], 20 : 20 + mask.shape[1]] = mask == 255
    np.testing.assert_allclose(
        output[opaque, :3],
        np.tile(np.array(color) * scale, (np.count_nonzero(opaque), 1)),
        rtol=1e-7,
    )
    if encoding in ("rgba", "bgra"):
        np.testing.assert_array_equal(output[..., 3], original[..., 3])


@pytest.mark.parametrize(
    "to_root,expected_anchor",
    [
        ([[1, 0, 100], [0, 1, 60], [0, 0, 1]], (40, 70)),
        ([[2, 0, 0.5], [0, 2, 0.5], [0, 0, 1]], (70, 65)),
    ],
)
def test_anchor_reprojects_only_once_and_wrong_ticket_or_frame_is_skipped(to_root, expected_anchor):
    registry = builtin_registry()
    context = {"spatial": frame([600, 240], "camera"), "ticket": TicketClock().issue("camera")}
    crop = moved_payload(context, np.float64(to_root), [300, 120])
    graphic = graphic_object(
        "label", [(140, 130)], (30, 190, 240), text="Δ Контроль", text_style=TextStyle(font_size=20)
    )
    layers = GraphicsBundle((GraphicLayer("test", (graphic,), context),))
    image = np.zeros((120, 300, 3), np.uint16)
    overlay = registry.create("cv.overlay_result_graphics")
    result = overlay.execute({"image": ValuePacket(image, crop), "graphics_1": layers})
    assert result is not None, overlay.error
    expected = image.copy()
    draw_object(expected, replace(graphic, points=(expected_anchor,)), np.eye(3))
    np.testing.assert_array_equal(result.outputs["image"], expected)
    for foreign in (
        {**context, "spatial": frame([600, 240], "other")},
        {**context, "ticket": TicketClock().issue("other")},
    ):
        layers = GraphicsBundle((GraphicLayer("foreign", (graphic,), foreign),))
        result = overlay.execute({"image": ValuePacket(image, crop), "graphics_1": layers})
        assert result is not None and "skipped 1" in result.message
        np.testing.assert_array_equal(result.outputs["image"], image)


@pytest.mark.parametrize("kind", ["label", "text"])
def test_legacy_unicode_graphics_and_filmstrip_use_real_glyphs(kind):
    graphic = graphic_object(kind, [(15, 60)], (0, 255, 0), text="Δ №2")
    expected = np.zeros((90, 220, 3), np.uint8)
    draw_object(expected, graphic, np.eye(3))
    assert expected.any()
    explicit = np.zeros_like(expected)
    draw_object(explicit, replace(graphic, text_style=TextStyle()), np.eye(3))
    np.testing.assert_array_equal(expected, explicit)
    context = {"spatial": frame([220, 90], "source")}
    actual = annotate_filmstrip_image(
        np.zeros_like(expected), context["spatial"], [GraphicLayer("text", (graphic,), context)]
    )
    np.testing.assert_array_equal(actual, expected)


def test_legacy_ascii_labels_keep_their_existing_hershey_pixels():
    from opencv_dataflow.graphics_layers import _draw_label

    graphic = graphic_object("label", [(10, 40)], (0, 255, 0), text="Pass 42")
    image = np.zeros((90, 220, 3), np.uint8)
    expected = image.copy()
    _draw_label(expected, graphic.text, (10, 40), graphic.color)
    draw_object(image, graphic, np.eye(3))
    np.testing.assert_array_equal(image, expected)


def test_project_relative_font_roundtrip_and_fail_safe_bad_font(tmp_path):
    registry = builtin_registry()
    font = tmp_path / "inspection.ttf"
    font.write_bytes(DEFAULT_FONT.read_bytes())
    graph = Graph()
    block = graph.add(
        registry.create(
            "cv.annotation_layer",
            block_id="label",
            parameters={
                "shape": "label",
                "label": "Δ №2",
                "font_file": "inspection.ttf",
            },
        )
    )
    graph.save(tmp_path / "project.json")
    graph = Graph.load(tmp_path / "project.json", registry)
    block = graph.blocks["label"]
    image = np.zeros((90, 220), np.uint8)
    result = block.execute({"image": image})
    assert result is not None, block.error
    assert result.outputs[RESULT_GRAPHICS].layers[0].objects[0].text_style.font_file == str(font)
    font.write_bytes(b"not a font")
    assert block.execute({"image": image}) is None
    assert "Cannot render annotation" in block.error and block.state == "error"
    assert not block.current_emissions
    graph.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("font_size", 7),
        ("font_size", 129),
        ("font_index", -1),
        ("text_outline", 9),
        ("text_direction", "sideways"),
        ("font_file", 1),
    ],
)
def test_font_parameter_validation(field, value):
    block = builtin_registry().create("cv.annotation_layer", parameters={field: value})
    assert block.execute({"image": np.zeros((30, 60), np.uint8)}) is None
    assert block.state == "error"


def test_explicit_direction_is_not_silently_ignored_without_shaping(monkeypatch):
    monkeypatch.setattr("opencv_dataflow.unicode_text.features.check_feature", lambda _: False)
    block = builtin_registry().create(
        "cv.annotation_layer",
        parameters={
            "shape": "label",
            "text_direction": "rtl",
        },
    )
    assert block.execute({"image": np.zeros((90, 220), np.uint8)}) is None
    assert "requires a Pillow build with Raqm" in block.error


def test_parallel_rasterization_and_tiny_clipped_canvases():
    def render(index):
        image = np.zeros((90, 220, 3), np.uint8)
        obj = graphic_object(
            "label",
            [(10, 40)],
            (0, 255, 0),
            text="Qualité Δ №2",
            text_style=TextStyle(font_size=24),
        )
        draw_object(image, obj, np.eye(3))
        return image

    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(render, range(12)))
    assert outputs[0].any()
    assert all(np.array_equal(output, outputs[0]) for output in outputs)
    for shape in ((1, 1), (7, 5), (0, 0)):
        image = np.zeros((*shape, 3), np.uint8)
        draw_object(image, graphic_object("text", [(-8, 3)], (0, 255, 0), text="Δ №2"), np.eye(3))


def test_unreadable_fonts_and_oversized_glyph_masks_fail_before_emission(tmp_path):
    registry = builtin_registry()
    block = registry.create(
        "cv.annotation_layer",
        parameters={
            "shape": "label",
            "font_file": str(tmp_path / "missing.ttf"),
        },
    )
    assert block.execute({"image": np.zeros((20, 20), np.uint8)}) is None
    assert "Cannot load text font" in block.error and not block.current_emissions
    block.parameters.update(font_file="", font_size=128, label="W" * 160)
    assert block.execute({"image": np.zeros((20, 20), np.uint8)}) is None
    assert "coverage budget" in block.error and not block.current_emissions


def test_float_hdr_and_nan_pixels_outside_glyphs_are_not_quantized():
    image = np.full((80, 200, 3), 123456.789012345, np.float64)
    image[0, 0] = [np.nan, np.inf, -np.inf]
    original = image.copy()
    style = TextStyle(font_size=24, outline=0)
    graphic = graphic_object("text", [(10, 50)], (0, 128, 255), text="Δ №2", text_style=style)
    draw_object(image, graphic, np.eye(3))
    fill, _, (left, top, _, _) = text_masks(graphic.text, style)
    support = np.zeros(image.shape[:2], bool)
    support[50 + top : 50 + top + fill.shape[0], 10 + left : 10 + left + fill.shape[1]] = fill != 0
    np.testing.assert_array_equal(image[~support], original[~support])
    assert np.any(image[support] != original[support])


def test_bad_runtime_font_is_contained_in_filmstrip_rendering(tmp_path):
    graphic = graphic_object(
        "label",
        [(15, 60)],
        (0, 255, 0),
        text="Δ №2",
        text_style=TextStyle(font_file=str(tmp_path / "gone.ttf")),
    )
    context = {"spatial": frame([220, 90], "source")}
    image = np.zeros((90, 220, 3), np.uint8)
    actual = annotate_filmstrip_image(
        image.copy(), context["spatial"], [GraphicLayer("text", (graphic,), context)]
    )
    np.testing.assert_array_equal(actual, image)


def test_public_project_replays_headlessly_and_graphics_modes_are_explicit():
    graph = Graph.load(
        Path("examples/opencv tools/unicode_annotations/project.json"), builtin_registry()
    )
    try:
        graph.run()
        assert all(block.state == "pass" for block in graph.processing_blocks())
        source = graph.blocks["source"].last_result.outputs["image"]
        output = graph.blocks["overlay"].last_result.outputs["image"]
        assert source.dtype == output.dtype == np.uint16
        assert np.any(source != output)
        expected = output.copy()
        graph.set_run_profile("lean")
        graph.run()
        np.testing.assert_array_equal(
            graph.blocks["overlay"].last_result.outputs["image"], expected
        )
        graph.set_graphics_mode("off")
        graph.run()
        np.testing.assert_array_equal(graph.blocks["overlay"].last_result.outputs["image"], source)
    finally:
        graph.close()
