"""Independent channel-shift truth, polynomial convention and profile ownership."""

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.chromatic_aberration import donor_coordinates, make_model
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock, ticket_key
from opencv_dataflow.timing import advance_timing, mark

WIDTH, HEIGHT = 36, 22
SHIFTS = [(-2, 1), (0, 0), (3, -2)]  # B, G, R displacements in native convention.


def model():
    return make_model([WIDTH, HEIGHT], 1, [[-2, 0, 0], [1, 0, 0], [3, 0, 0], [-2, 0, 0]])


def scene(dtype, encoding):
    y, x = np.indices((HEIGHT, WIDTH))
    intensity = 3000 + 17 * x + 39 * y + (((x - 18) ** 2 + (y - 11) ** 2) < 36) * 8000
    ideal = np.stack([intensity + 100, intensity + 300, intensity + 700], axis=-1).astype(float)
    if dtype == np.uint8:
        ideal /= 128
    elif np.dtype(dtype).kind == "f":
        ideal = ideal / 65535 + 0.00123
    ideal = ideal.astype(dtype)
    observed = np.empty_like(ideal)
    for channel, (dx, dy) in enumerate(SHIFTS):
        observed[..., channel] = ideal[
            np.clip(y + dy, 0, HEIGHT - 1), np.clip(x + dx, 0, WIDTH - 1), channel
        ]
    if encoding.startswith("rgb"):
        observed, ideal = observed[..., ::-1], ideal[..., ::-1]
    if encoding in ("bgra", "rgba"):
        alpha = ((x + y) % 13).astype(dtype)
        ideal = np.dstack((ideal, alpha))
        observed = np.dstack((observed, alpha))
    return ImageArray(observed, encoding), ideal


def execute(image, calibration=None, parameters=None, **inputs):
    block = builtin_registry().create(
        "cv.correct_chromatic_aberration", parameters=parameters or {}
    )
    result = block.execute({"image": image, "calibration": calibration or model(), **inputs})
    assert result is not None, block.error
    return block, result


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16, np.float32, np.float64])
@pytest.mark.parametrize("encoding", ["rgb", "bgr", "rgba", "bgra"])
def test_integer_fringe_correction_restores_independent_pixels_and_depth_order_alpha(
    dtype, encoding
):
    source, truth = scene(dtype, encoding)
    original = source.copy()
    _, result = execute(source)
    output, valid = result.outputs["image"], result.outputs["valid_mask"] != 0
    assert output.dtype == source.dtype and image_format(output) == encoding
    expected_scope = np.zeros((HEIGHT, WIDTH), bool)
    expected_scope[1 : HEIGHT - 2, 3 : WIDTH - 2] = True
    np.testing.assert_equal(valid, expected_scope)
    np.testing.assert_equal(output[valid], truth[valid])
    np.testing.assert_equal(source, original)
    green = 1
    np.testing.assert_equal(output[..., green], source[..., green])
    if source.shape[-1] == 4:
        np.testing.assert_equal(output[..., 3], source[..., 3])
    assert result.outputs["valid_fraction"] == pytest.approx(31 * 19 / (WIDTH * HEIGHT))
    json.dumps(result.outputs["report"], allow_nan=False)


def test_quadratic_native_basis_and_float32_maps_against_analytic_ramps():
    width, height = 40, 32
    y, x = np.indices((height, width), dtype=np.float64)
    coefficients = [
        [0.5, 0.25, -0.125, 0.125, 0.25, -0.125],
        [-0.25, 0.125, 0.25, 0.125, -0.125, 0.25],
        [-0.75, -0.125, 0.25, -0.125, 0.125, 0.25],
        [0.5, 0.25, -0.125, 0.25, 0.125, -0.125],
    ]
    calibration = make_model([width, height], 2, coefficients)
    u, v = (x - width / 2) / (width / 2), (y - height / 2) / (height / 2)
    independent = []
    for axis, c in enumerate(coefficients):
        delta = c[0] + c[1] * v + c[2] * u + c[3] * v * v + c[4] * u * v + c[5] * u * u
        independent.append(
            (x if axis % 2 == 0 else y).astype(np.float32) - delta.astype(np.float32)
        )
    np.testing.assert_equal(donor_coordinates(x, y, calibration), independent)
    source = np.stack((100 + 2 * x + 3 * y, 1000 + x + y, 200 + 4 * x + 7 * y), axis=-1)
    _, result = execute(source, calibration)
    output = result.outputs["image"]
    valid = result.outputs["valid_mask"] != 0
    for channel, index, a, b, intercept in ((0, 0, 2, 3, 100), (2, 2, 4, 7, 200)):
        # OpenCV linear interpolation quantizes subpixel positions to 1/32 pixel.
        xx, yy = (np.rint(independent[j] * 32) / 32 for j in (index, index + 1))
        expected = intercept + a * xx + b * yy
        np.testing.assert_allclose(output[..., channel][valid], expected[valid], atol=1e-12)


def test_scoped_correction_retains_surroundings_and_zero_border_policy_preserves_alpha():
    source, truth = scene(np.uint16, "rgba")
    scope = np.zeros((HEIGHT, WIDTH), np.uint8)
    scope[4:18, 5:31] = 255
    _, result = execute(source, mask=scope)
    np.testing.assert_equal(result.outputs["image"][scope == 0], source[scope == 0])
    np.testing.assert_equal(result.outputs["image"][scope != 0], truth[scope != 0])
    np.testing.assert_equal(result.outputs["valid_mask"], scope)
    _, zero = execute(source, parameters={"invalid_border": "zero"})
    invalid = zero.outputs["valid_mask"] == 0
    assert not zero.outputs["image"][..., :3][invalid].any()
    np.testing.assert_equal(zero.outputs["image"][..., 3], source[..., 3])


@pytest.mark.parametrize(
    "region",
    [
        {"shape": "rectangle", "rectangle": [6, 5, 20, 12]},
        {"shape": "circle", "center": [18, 11], "radius": 6},
        {"shape": "polygon", "vertices": [[6, 5], [25, 5], [25, 16], [6, 16]]},
    ],
)
def test_region_selection_composes_with_mask(region):
    from opencv_dataflow.selection import rasterize

    source, truth = scene(np.uint16, "bgr")
    scope = {"model": "image_region_v1", "image_size": [WIDTH, HEIGHT], **region}
    mask = np.full((HEIGHT, WIDTH), 255, np.uint8)
    mask[:, 20:] = 0
    _, result = execute(source, mask=mask, region=scope)
    active = (rasterize(scope, source.shape) != 0) & (mask != 0)
    np.testing.assert_equal(result.outputs["image"][active], truth[active])
    np.testing.assert_equal(result.outputs["image"][~active], source[~active])


def test_green_frame_ticket_start_and_companions_survive_with_explicit_grid_policy():
    source, _ = scene(np.uint16, "rgb")
    spatial = frame([WIDTH, HEIGHT], "camera")
    spatial["to_root"][0][2] = 40
    ticket = TicketClock().issue("camera")
    timing = advance_timing(None, mark(), mark(), source=True)
    payload = {
        "spatial": spatial,
        "ticket": ticket,
        "recipe": "parts-v3",
        "pyramid": {
            "level": 1,
            "largest_size": [72, 44],
            "to_largest": [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
        },
        "processing_time": timing,
    }
    packet = ValuePacket(source, payload)
    block = builtin_registry().create("cv.correct_chromatic_aberration")
    assert block.execute({"image": packet, "calibration": model()}) is None
    assert "original calibration grid" in block.error
    block.parameters["calibration_grid"] = "current"
    result = block.execute({"image": packet, "calibration": model()})
    assert result is not None, block.error
    for name in ("image", "valid_mask", "report"):
        value = result.payloads[name]
        assert ticket_key(value["ticket"]) == ticket_key(ticket)
        assert value["processing_time"]["started_at"] == timing["started_at"]
        assert value["spatial"] == spatial and value["pyramid"] == payload["pyramid"]
        assert value["recipe"] == "parts-v3"


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
def test_explicit_bayer_path_matches_native_demosaicing_and_excludes_synthesized_border(
    dtype, pattern
):
    y, x = np.indices((HEIGHT, WIDTH))
    mosaic = ((x * 3 + y * 7) % 200).astype(dtype)
    identity = make_model([WIDTH, HEIGHT], 0, [[0.0]] * 4)
    _, result = execute(mosaic, identity, {"bayer_pattern": pattern})
    expected = cv2.demosaicing(mosaic, getattr(cv2, f"COLOR_Bayer{pattern}2BGR"))
    np.testing.assert_equal(result.outputs["image"], expected)
    valid = np.full((HEIGHT, WIDTH), 255, np.uint8)
    valid[[0, -1]] = 0
    valid[:, [0, -1]] = 0
    np.testing.assert_equal(result.outputs["valid_mask"], valid)


def write_native(path, nested=False):
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    if nested:
        storage.startWriteStruct("lens", cv2.FileNode_MAP)
    storage.write("image_width", WIDTH)
    storage.write("image_height", HEIGHT)
    for name, index in (("blue_channel", 0), ("red_channel", 2)):
        storage.startWriteStruct(name, cv2.FileNode_MAP)
        for axis, row in (("x", index), ("y", index + 1)):
            storage.startWriteStruct("coeffs_" + axis, cv2.FileNode_SEQ | cv2.FileNode_FLOW)
            for value in model()["coefficients"][row]:
                storage.write("", value)
            storage.endWriteStruct()
        storage.endWriteStruct()
    if nested:
        storage.endWriteStruct()
    storage.release()


@pytest.mark.parametrize("suffix", [".yml", ".xml", ".json"])
@pytest.mark.parametrize("nested", [False, True])
def test_loads_native_calibration_formats_and_project_relative_paths(tmp_path, suffix, nested):
    path = tmp_path / ("lens" + suffix)
    write_native(path, nested)
    block = builtin_registry().create(
        "input.chromatic_calibration",
        parameters={"path": path.name, "path_base": "project", "node": "lens" if nested else ""},
    )
    block.project_dir = tmp_path
    result = block.execute({})
    assert result is not None, block.error
    assert result.outputs["calibration"] == model()
    assert result.outputs["info"]["degree"] == 1
    json.dumps(result.outputs, allow_nan=False)


@pytest.mark.parametrize(
    "case", ["dimension", "fingerprint", "row_order", "missing", "nonfinite", "large", "degree"]
)
def test_invalid_models_do_not_reach_native_or_emit(monkeypatch, case):
    source, _ = scene(np.uint16, "bgr")
    calibration = copy.deepcopy(model())
    if case == "dimension":
        calibration = make_model([WIDTH + 1, HEIGHT], 1, calibration["coefficients"])
    elif case == "fingerprint":
        calibration["id"] = "obsolete"
    elif case == "row_order":
        calibration["row_order"] = list(reversed(calibration["row_order"]))
    elif case == "missing":
        calibration = {}
    elif case == "nonfinite":
        calibration["coefficients"][0][0] = float("nan")
    elif case == "large":
        calibration["coefficients"][0][0] = 1e8
    elif case == "degree":
        calibration["degree"] = 12

    def unexpected(*args, **kwargs):
        pytest.fail("invalid model reached native correction")

    monkeypatch.setattr(cv2, "correctChromaticAberration", unexpected)
    block = builtin_registry().create("cv.correct_chromatic_aberration")
    assert block.execute({"image": source, "calibration": calibration}) is None
    assert not block.current_emissions


def test_low_support_is_a_failed_result_with_masks_and_no_primary_image():
    source, _ = scene(np.uint16, "bgr")
    block, result = execute(source, parameters={"minimum_valid_fraction": 0.99})
    assert block.state == "fail"
    assert result.outputs["image"] is NO_VALUE
    assert not block.current_emissions.get("image")
    assert result.outputs["valid_mask"].any()
    assert not result.outputs["report"]["passed"]


@pytest.mark.parametrize("parameters", [{"maximum_displacement": 1}, {"maximum_work": 1}])
def test_work_and_displacement_guards_fail_before_native(monkeypatch, parameters):
    source, _ = scene(np.uint16, "bgr")

    def unexpected(*args, **kwargs):
        pytest.fail("budget violation reached native correction")

    monkeypatch.setattr(cv2, "correctChromaticAberration", unexpected)
    block = builtin_registry().create("cv.correct_chromatic_aberration", parameters=parameters)
    assert block.execute({"image": source, "calibration": model()}) is None
    assert "maximum" in block.error and not block.current_emissions


def test_graphics_off_and_xml_configuration(tmp_path):
    from opencv_dataflow.block_xml import save_block_xml
    from opencv_dataflow.project_xml import read_block_xml

    source, _ = scene(np.uint16, "bgr")
    block = builtin_registry().create("cv.correct_chromatic_aberration")
    block.graphics_mode = "off"
    result = block.execute({"image": source, "calibration": model()})
    assert result is not None and not result.drawings
    for kind in (
        "cv.correct_chromatic_aberration",
        "calibration.chromatic_model",
        "input.chromatic_calibration",
    ):
        block = builtin_registry().create(kind)
        path = tmp_path / ("preset." + kind + ".xml")
        save_block_xml(block, path)
        assert read_block_xml(path) == block.to_project_item()


@pytest.mark.parametrize("dtype", [np.uint16, np.float32, np.float64])
@pytest.mark.parametrize("kind", ["text", "label"])
def test_ascii_result_annotations_retain_non8bit_samples_and_alpha(dtype, kind):
    from opencv_dataflow.graphics_layers import draw_object, graphic_object

    image = np.full((80, 240, 4), 0.123 if np.dtype(dtype).kind == "f" else 17000, dtype)
    before = image.copy()
    obj = graphic_object(kind, [(20, 40)], (80, 200, 120), text="CA support 96.1%")
    draw_object(image, obj, np.eye(3))
    assert image.dtype == before.dtype
    assert np.isfinite(image).all()
    np.testing.assert_equal(image[..., 3], before[..., 3])
    assert np.any(image[..., :3] != before[..., :3])
    np.testing.assert_equal(image[:10], before[:10])


@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize(
    "project", ["chromatic_aberration", "defined_model", "insufficient_support"]
)
def test_public_graphs_restore_independent_reference_or_withhold_unqualified_image(
    profile, project
):
    from opencv_dataflow.core import Graph

    folder = Path(__file__).resolve().parents[1] / "examples/opencv tools/chromatic_aberration"
    truth = json.loads((folder / "expected.json").read_text(encoding="utf8"))
    graph = Graph.load(folder / (project + ".json"), builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        correction = graph.blocks["correct"]
        result = correction.last_result
        assert result is not None, correction.error
        assert np.count_nonzero(result.outputs["valid_mask"]) == truth["valid_pixels"]
        assert result.outputs["valid_fraction"] == pytest.approx(
            truth["valid_pixels"] / truth["total_pixels"]
        )
        if project == "insufficient_support":
            assert correction.state == "fail" and not correction.current_emissions.get("image")
            assert graph.blocks["after"].last_result is None
            assert graph.blocks["filmstrip"].last_result is None
        else:
            assert all(b.state == "pass" for b in graph.processing_blocks()), [
                (b.id, b.state, b.error) for b in graph.processing_blocks()
            ]
            assert (
                graph.blocks["before"].last_result.outputs["value"]
                == truth["before_max_absolute_error"]
            )
            assert graph.blocks["after"].last_result.outputs["value"] == 0
            valid = result.outputs["valid_mask"] != 0
            reference = graph.blocks["reference"].last_result.outputs["image"]
            np.testing.assert_equal(result.outputs["image"][valid], reference[valid])
            assert image_format(result.outputs["image"]) == "rgb"
            assert result.outputs["image"].dtype == np.uint16
    finally:
        graph.close()


@pytest.mark.parametrize(
    "content",
    [
        "",
        "x" * 262145,
        '{"image_width":36,"image_height":22}',
        '{"image_width":36,"image_height":22,"blue_channel":{"coeffs_x":[0,1],"coeffs_y":[0,1]},"red_channel":{"coeffs_x":[0,1],"coeffs_y":[0,1]}}',
    ],
)
def test_malformed_or_oversized_calibration_files_are_contained(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf8")
    block = builtin_registry().create("input.chromatic_calibration", parameters={"path": str(path)})
    assert block.execute({}) is None
    assert block.error and not block.current_emissions


def test_full_degree_11_model_is_supported_on_a_bounded_grid():
    coefficients = np.zeros((4, 78), np.float32)
    coefficients[0, 66] = 0.125  # y^11, first term of the degree-11 group.
    calibration = make_model([WIDTH, HEIGHT], 11, coefficients)
    source, _ = scene(np.float32, "bgr")
    _, result = execute(source, calibration)
    assert result.outputs["image"].shape == source.shape
    x, y = np.array([18.0]), np.array([5.0])
    expected = np.float32(18) - np.float32(0.125 * ((5 - HEIGHT / 2) / (HEIGHT / 2)) ** 11)
    assert donor_coordinates(x, y, calibration)[0, 0] == expected
