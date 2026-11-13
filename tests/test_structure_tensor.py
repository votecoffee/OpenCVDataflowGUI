"""Analytic directions, ambiguous corners, native apertures and graph/payload contracts."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import DepthArray, ImageArray
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.structure_tensor import StructureTensor, orientation_summary, tensor_maps
from opencv_dataflow.tickets import TicketClock

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/structure_tensor"


def ramp(degrees, shape=(80, 96)):
    y, x = np.mgrid[: shape[0], : shape[1]]
    radians = np.deg2rad(degrees)
    return (
        0.5 + (np.cos(radians) * (x - shape[1] / 2) + np.sin(radians) * (y - shape[0] / 2)) / 200
    ).astype(np.float32)


def run(source, extra=None, **parameters):
    block = builtin_registry().create("inspect.structure_tensor", parameters=parameters)
    result = block.execute({"image": source, **(extra or {})})
    assert result is not None, block.error
    return block, result


def axial_error(actual, expected):
    return np.abs((np.asarray(actual) - expected + 90) % 180 - 90)


@pytest.mark.parametrize("normal", [0, 30, 45, 90, 135])
@pytest.mark.parametrize("basis", ["normal", "tangent"])
def test_analytic_ramp_angles_vectors_and_axial_summary(normal, basis):
    source = ramp(normal)
    mask = np.zeros(source.shape, np.uint8)
    mask[10:-10, 10:-10] = 255
    _, result = run(source, {"mask": mask}, orientation_basis=basis)
    expected = (normal + (90 if basis == "tangent" else 0)) % 180
    valid = result.outputs["valid"] != 0
    assert np.array_equal(valid, mask != 0)
    assert np.max(axial_error(result.outputs["orientation"][valid], expected)) < 0.001
    np.testing.assert_allclose(result.outputs["coherence"][valid], 1, atol=1e-6)
    x, y = result.outputs["vector_x"][valid], result.outputs["vector_y"][valid]
    np.testing.assert_allclose(x * x + y * y, 1, atol=1e-6)
    assert np.max(axial_error(np.degrees(np.arctan2(y, x)), expected)) < 0.001
    assert axial_error(result.outputs["dominant_angle"], expected) < 0.001
    assert result.outputs["measurements"]["orientation_concentration"] > 0.999999


@pytest.mark.parametrize(
    "aperture,border,block_size",
    [
        ("1", "replicate", 4),
        ("3", "constant", 9),
        ("3", "reflect_101", 9),
        ("3", "reflect_edge", 9),
        ("5", "replicate", 7),
        ("7", "reflect_101", 11),
        ("scharr", "replicate", 5),
    ],
)
def test_eigenvalues_match_native_apertures_and_image_borders(aperture, border, block_size):
    source = np.random.default_rng(129).integers(0, 256, (28, 34), dtype=np.uint8)
    _, result = run(source, gradient_size=aperture, border=border, block_size=block_size)
    borders = {
        "constant": cv2.BORDER_CONSTANT,
        "replicate": cv2.BORDER_REPLICATE,
        "reflect_101": cv2.BORDER_REFLECT_101,
        "reflect_edge": cv2.BORDER_REFLECT,
    }
    native = cv2.cornerEigenValsAndVecs(
        source.astype(np.float32) / 255,
        block_size,
        -1 if aperture == "scharr" else int(aperture),
        borderType=borders[border],
    )
    high = np.maximum(np.maximum(native[..., 0], native[..., 1]), 0)
    low = np.maximum(np.minimum(native[..., 0], native[..., 1]), 0)
    np.testing.assert_array_equal(result.outputs["eigenvalue_max"], high)
    np.testing.assert_array_equal(result.outputs["eigenvalue_min"], low)
    np.testing.assert_allclose(result.outputs["energy"], high.astype(float) + low, rtol=1e-7)


def test_corners_flat_areas_and_zero_energy_are_not_reliable_directions():
    y, x = np.mgrid[:96, :96]
    _, result = run(((x >= 48) & (y >= 48)).astype(np.float32))
    assert result.outputs["eigenvalue_min"][48, 48] > 0.1
    assert result.outputs["coherence"][48, 48] < 0.2
    assert result.outputs["valid"][48, 48] == 0
    assert result.outputs["valid"][48, 65] == 255
    assert result.outputs["energy"][20, 20] == 0
    for source, extra in (
        (np.full((32, 40), 0.6, np.float32), {}),
        (ramp(30), {"mask": np.zeros((80, 96), np.uint8)}),
    ):
        _, blank = run(source, extra, min_energy=0, min_coherence=0)
        assert blank.passed
        assert blank.outputs["dominant_angle"] is NO_VALUE
        assert blank.outputs["measurements"]["dominant_angle_degrees"] is None
        assert np.count_nonzero(blank.outputs["valid"]) == 0
        assert np.isfinite(blank.outputs["orientation"]).all()


def test_unsorted_eigenpairs_and_equal_eigenvalues():
    native = np.array([[[1, 4, 1, 0, 0, 1], [2, 2, 1, 0, 0, 1]]], np.float32)
    p = {**StructureTensor.defaults, "min_coherence": 0, "orientation_basis": "normal"}
    maps = tensor_maps(native, np.ones((1, 2), bool), p)
    assert maps["eigenvalue_max"].tolist() == [[4, 2]]
    assert maps["orientation"].tolist() == [[90, 0]]
    assert maps["valid"].tolist() == [[255, 0]]
    assert maps["coherence"][0, 0] == pytest.approx(0.6)
    p["min_coherence"] = 0.6
    assert tensor_maps(native, np.ones((1, 2), bool), p)["valid"][0, 0] == 255
    p["min_coherence"] = 0.60001
    assert tensor_maps(native, np.ones((1, 2), bool), p)["valid"][0, 0] == 0
    p.update(min_coherence=0, min_energy=5)
    assert tensor_maps(native, np.ones((1, 2), bool), p)["valid"][0, 0] == 0


def test_axial_mean_wrap_and_balanced_perpendicular_directions():
    maps = {
        "orientation": np.array([[1, 179]], np.float32),
        "valid": np.full((1, 2), 255, np.uint8),
        "energy": np.ones((1, 2), np.float32),
        "coherence": np.ones((1, 2), np.float32),
    }
    selected = np.ones((1, 2), bool)
    report = orientation_summary(maps, selected, StructureTensor.defaults)
    assert axial_error(report["dominant_angle_degrees"], 0) < 1e-9
    assert 0 <= report["dominant_angle_degrees"] < 180
    maps["orientation"] = np.array([[0, 90]], np.float32)
    report = orientation_summary(maps, selected, StructureTensor.defaults)
    assert report["dominant_angle_degrees"] is None
    assert report["orientation_concentration"] < 1e-6


@pytest.mark.parametrize("shape", ["circle", "polygon"])
def test_selection_intersection_payload_and_original_high_bit_color(shape):
    source = (np.rint(ramp(30) * 65535)).astype(np.uint16)
    rgb = ImageArray(np.dstack((source, source, source)), "rgb")
    original = rgb.copy()
    ticket = TicketClock().issue("part-129")
    spatial = frame([96, 80], "camera")
    spatial.update(root_size=[192, 160], to_root=[[2, 0, 0], [0, 2, 0], [0, 0, 1]])
    context = {
        "ticket": ticket,
        "spatial": spatial,
        "encoder": 512,
        "pyramid": {
            "level": 1,
            "largest_size": [192, 160],
            "to_largest": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        },
    }
    mask = np.zeros((80, 96), np.uint8)
    mask[:, 35:] = 255
    expected = np.zeros_like(mask)
    region = {"model": "image_region_v1", "image_size": [96, 80], "shape": shape}
    if shape == "circle":
        region.update(center=[48, 40], radius=24)
        cv2.circle(expected, (48, 40), 24, 255, -1)
    else:
        vertices = [[20, 20], [74, 24], [65, 60], [28, 58]]
        region["vertices"] = vertices
        cv2.fillPoly(expected, [np.asarray(vertices, np.int32)], 255)
    selected = (expected != 0) & (mask != 0)
    block, result = run(ValuePacket(rgb, context), {"mask": mask, "region": region})
    assert result.outputs["image"].dtype == np.uint16
    assert result.outputs["image"].color_format == "rgb"
    np.testing.assert_array_equal(result.outputs["image"], original)
    np.testing.assert_array_equal(rgb, original)
    assert result.outputs["measurements"]["selected_pixels"] == int(selected.sum())
    assert np.array_equal(result.outputs["valid"] != 0, selected)
    for name in (
        "orientation",
        "coherence",
        "energy",
        "vector_x",
        "vector_y",
        "eigenvalue_max",
        "eigenvalue_min",
    ):
        assert np.all(result.outputs[name][~selected] == 0)
        assert result.payloads[name]["spatial"] == spatial
        assert result.payloads[name]["ticket"] == ticket
        assert result.payloads[name]["pyramid"] == context["pyramid"]
        assert result.payloads[name]["encoder"] == 512
        assert result.outputs[name].color_format == "gray"
        assert result.payloads[name]["structure_tensor_map"]["quantity"] == name
    block.set_graphics_mode("off")
    lean = block.execute({"image": ValuePacket(rgb, context), "mask": mask, "region": region})
    assert lean is not None, block.error
    assert not lean.drawings and not lean.graphics
    assert lean.outputs["overlay"] is NO_VALUE and lean.outputs["orientation_preview"] is NO_VALUE
    np.testing.assert_array_equal(lean.outputs["orientation"], result.outputs["orientation"])
    assert lean.outputs["measurements"] == result.outputs["measurements"]


def test_selection_does_not_create_an_artificial_image_border():
    source = np.ones((40, 48), np.float32)
    mask = np.zeros(source.shape, np.uint8)
    mask[8:24, 12:30] = 255
    _, result = run(source, {"mask": mask})
    assert np.all(result.outputs["energy"] == 0)
    assert np.all(result.outputs["valid"] == 0)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
def test_intensity_scale_is_consistent_and_source_is_unchanged(dtype):
    base = np.rint(ramp(45) * 255).astype(np.uint8)
    source = (
        base
        if dtype == np.uint8
        else base.astype(np.uint16) * 257
        if dtype == np.uint16
        else base.astype(dtype) / 255
    )
    _, result = run(source)
    _, reference = run(base)
    np.testing.assert_allclose(result.outputs["energy"], reference.outputs["energy"], atol=1e-10)
    np.testing.assert_array_equal(result.outputs["image"], source)


def test_graphics_are_bounded_and_invalid_pixels_are_black():
    block, result = run(ramp(30), glyph_step=1, max_glyphs=12)
    assert len(result.drawings["Local orientation"].objects) == 12
    assert result.outputs["orientation_preview"].shape == (80, 96, 3)
    assert result.outputs["overlay"].dtype == np.uint8
    block.parameters["min_energy"] = 10
    empty = block.execute({"image": ramp(30)})
    assert empty is not None, block.error
    assert np.all(empty.outputs["orientation_preview"] == 0)
    assert len(empty.drawings["Local orientation"].objects) == 0


@pytest.mark.parametrize(
    "parameters,message",
    [
        ({"block_size": 0}, "between"),
        ({"gradient_size": "9"}, "one of"),
        ({"border": "wrap"}, "one of"),
        ({"orientation_basis": "signed"}, "one of"),
        ({"min_energy": -1}, "at least"),
        ({"min_coherence": 1.01}, "between"),
        ({"min_energy": float("nan")}, "finite"),
        ({"glyph_step": 0}, "between"),
    ],
)
def test_invalid_configuration_is_contained(parameters, message):
    block = StructureTensor(parameters=parameters)
    assert block.execute({"image": ramp(0)}) is None
    assert block.state == "error" and message in block.error


def test_invalid_images_and_selection_frames_are_rejected():
    for source, message in (
        (DepthArray(np.ones((20, 20))), "depth tools"),
        (np.full((20, 20), float("nan"), np.float32), "finite"),
        (np.full((20, 20), 1e40, np.float64), "exceeds float32"),
    ):
        block = StructureTensor()
        assert block.execute({"image": source}) is None
        assert message.lower() in block.error.lower()
    source = ValuePacket(ramp(0), {"spatial": frame([96, 80], "a")})
    mask = ValuePacket(np.ones((80, 96), np.uint8), {"spatial": frame([96, 80], "b")})
    block = StructureTensor()
    assert block.execute({"image": source, "mask": mask}) is None
    assert "frame differs" in block.error


def test_pixel_budget_and_native_exceptions_are_contained(monkeypatch):
    def broken(*args, **kwargs):
        raise cv2.error("native tensor failed")

    monkeypatch.setattr(cv2, "cornerEigenValsAndVecs", broken)
    block = StructureTensor()
    assert block.execute({"image": np.zeros((1600, 1600), np.uint8)}) is None
    assert "2.1 million" in block.error
    assert block.execute({"image": ramp(0)}) is None
    assert "native tensor failed" in block.error
    assert not block.current_emissions


def test_block_metadata_and_xml_persistence(tmp_path):
    registry = builtin_registry()
    block = registry.create(
        "inspect.structure_tensor",
        block_id="tensor",
        parameters={
            "gradient_size": "scharr",
            "orientation_basis": "normal",
            "min_coherence": 0.7,
        },
    )
    assert "energy" not in block.exposed_outputs
    block.exposed_outputs += ["energy", "eigenvalue_max"]
    block.description, block.documentation = "Selected coating direction", "<p>Inspect texture.</p>"
    graph = Graph()
    try:
        graph.add(block)
        expected = graph.to_dict()
        path = tmp_path / "tensor.xml"
        export_project_xml(expected, path)
        restored = Graph.from_dict(read_project_xml(path), registry)
        try:
            assert restored.to_dict() == expected
        finally:
            restored.close()
    finally:
        graph.close()


@pytest.mark.parametrize("graphics", ["on", "off"])
def test_public_example_angles_and_ambiguous_regions(graphics):
    graph = Graph.load(EXAMPLE / "project.json", builtin_registry())
    truth = json.loads((EXAMPLE / "assets/truth.json").read_text(encoding="utf-8"))
    try:
        for block in graph.blocks.values():
            block.set_graphics_mode(graphics)
        graph.run()
        assert all(b.state == "pass" for b in graph.processing_blocks())
        for name, expected in truth["probes"].items():
            result = graph.blocks[name].last_result
            report = result.outputs["measurements"]
            if expected["tangent_degrees"] is None:
                assert report["valid_pixels"] == 0
                assert result.outputs["dominant_angle"] is NO_VALUE
            else:
                assert report["valid_fraction"] > 0.99
                assert (
                    axial_error(result.outputs["dominant_angle"], expected["tangent_degrees"]) < 0.1
                )
                assert (
                    np.max(
                        axial_error(
                            result.outputs["orientation"][result.outputs["valid"] != 0],
                            expected["tangent_degrees"],
                        )
                    )
                    < 0.1
                )
    finally:
        graph.close()
