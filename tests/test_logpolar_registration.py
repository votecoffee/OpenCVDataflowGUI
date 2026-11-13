"""Independent similarity geometry, symmetric-pose rejection and source/target context."""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.timing import advance_timing, mark

TYPE = "inspect.logpolar_registration"


def part(shape=(192, 192)):
    h, w = shape
    image = np.zeros(shape, np.float64)
    cv2.rectangle(image, (w // 4, h // 4), (3 * w // 5, 7 * h // 10), 0.7, -1)
    cv2.circle(image, (2 * w // 3, 3 * h // 8), min(h, w) // 10, 1, -1)
    cv2.circle(image, (3 * w // 8, 7 * h // 16), min(h, w) // 17, 0, -1)
    cv2.rectangle(image, (9 * w // 20, 11 * h // 20), (11 * w // 20, 13 * h // 20), 0.1, -1)
    noise = np.random.default_rng(137).random(shape)
    image += noise * 0.12 * (image > 0)
    return cv2.GaussianBlur(image, (3, 3), 0.5)


def transform(shape, angle, scale, offset):
    h, w = shape
    x, y = (w - 1) / 2, (h - 1) / 2
    a, b = scale * math.cos(math.radians(angle)), scale * math.sin(math.radians(angle))
    return np.array(
        [[a, b, x - a * x - b * y + offset[0]], [-b, a, y + b * x - a * y + offset[1]], [0, 0, 1]],
        float,
    )


def register(reference, moving, parameters=None, **extra):
    block = builtin_registry().create(TYPE, parameters=parameters or {})
    result = block.execute({"image": moving, "reference": reference, **extra})
    assert result is not None, block.error
    return block, result


@pytest.mark.parametrize(
    "angle,scale,offset",
    (
        (23, 1.15, (7, -5)),
        (-34, 0.82, (-4, 6)),
        (145, 1.07, (4, 3)),
        (0, 1, (5, 2)),
        (63, 1.3, (-6, 3)),
    ),
)
def test_similarity_recovers_independent_forward_and_inverse_geometry(angle, scale, offset):
    reference = part()
    generated = transform(reference.shape, angle, scale, offset)
    moving = cv2.warpAffine(reference, generated[:2], reference.shape[::-1])
    block, result = register(reference, moving)
    assert result.passed, result.outputs["info"]
    found = np.asarray(result.outputs["transform"]["source_to_target"])
    points = np.array([[50, 60, 1], [120, 60, 1], [120, 120, 1], [50, 120, 1]]).T
    errors = np.linalg.norm((found @ generated @ points - points)[:2], axis=0)
    assert max(errors) < 0.3
    assert result.outputs["angle"] == pytest.approx(-angle, abs=0.1)
    assert result.outputs["scale"] == pytest.approx(1 / scale, abs=0.002)
    assert result.outputs["similarity"] > 0.98
    np.testing.assert_allclose(result.outputs["geometry"]["output_to_input"], np.linalg.inv(found))
    assert result.outputs["info"]["distinct_poses"] >= 1
    assert result.outputs["info"]["margin"] is None or result.outputs["info"]["margin"] >= 0.03
    assert len(result.outputs["candidates"]) <= result.outputs["info"]["candidate_budget"]
    assert block.parallel_safe and block.process_inputs_readonly


@pytest.mark.parametrize("shape,work_size", (((127, 192), 128), ((300, 420), 192)))
def test_isotropic_work_canvas_maps_rectangular_odd_and_reduced_images_back_to_original(
    shape, work_size
):
    reference = part(shape)
    generated = transform(shape, -19, 0.92, (6, -4))
    moving = cv2.warpAffine(reference, generated[:2], shape[::-1])
    _, result = register(reference, moving, {"work_size": work_size})
    assert result.passed, result.outputs["info"]
    found = np.asarray(result.outputs["transform"]["source_to_target"])
    points = np.array([[shape[1] * 0.3, shape[0] * 0.3, 1], [shape[1] * 0.7, shape[0] * 0.7, 1]]).T
    assert np.max(np.linalg.norm((found @ generated @ points - points)[:2], axis=0)) < 1.0


def test_symmetric_parts_withhold_primary_outputs_until_an_orientation_prior_resolves_them():
    half = part()
    reference = np.maximum(half, half[::-1, ::-1])
    _, result = register(reference, reference)
    assert not result.passed and result.outputs["info"]["status"] == "registration_ambiguous"
    assert result.outputs["info"]["distinct_poses"] >= 2
    for name in ("image", "transform", "geometry", "angle", "scale", "valid_mask"):
        assert result.outputs[name] is NO_VALUE
    assert result.outputs["info"]["margin"] < 0.03
    _, restricted = register(reference, reference, {"minimum_angle": -45, "maximum_angle": 45})
    assert restricted.passed
    assert abs(restricted.outputs["angle"]) < 0.1


def test_rotationally_repeated_parts_or_flat_images_never_publish_an_arbitrary_pose():
    a = part()
    square = np.maximum.reduce([np.rot90(a, k) for k in range(4)])
    for image in (square, np.zeros((128, 128)), np.ones((127, 192))):
        _, result = register(image, image)
        assert not result.passed and result.outputs["transform"] is NO_VALUE
        assert result.outputs["image"] is NO_VALUE
        json.dumps(result.outputs["info"], allow_nan=False)
        json.dumps(result.outputs["candidates"], allow_nan=False)


@pytest.mark.parametrize("depth", (np.uint8, np.uint16, np.float32, np.float64))
def test_source_encoding_depth_alpha_and_distinct_ticket_frames_survive_registration(depth):
    reference = part()
    maximum = np.iinfo(depth).max if np.dtype(depth).kind == "u" else 1
    color = np.dstack([reference, reference * 0.75, reference * 0.5, np.ones_like(reference)])
    reference_image = ImageArray((color * maximum).astype(depth), "rgba")
    generated = transform(reference.shape, 27, 1.12, (3, -4))
    moving = ImageArray(
        cv2.warpAffine(reference_image, generated[:2], reference.shape[::-1]), "rgba"
    )
    clock = TicketClock()
    source_context = {
        "ticket": clock.issue("part-8"),
        "spatial": frame([192, 192], "moving"),
        "fixture": {"lot": "L"},
        "alpha_representation": "straight",
        "processing_time": advance_timing(None, mark(), mark(), source=True),
    }
    ref_context = {
        "ticket": clock.issue("master"),
        "spatial": frame([192, 192], "master"),
        "pyramid": {
            "level": 1,
            "largest_size": [384, 384],
            "to_largest": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        },
    }
    _, result = register(
        ValuePacket(reference_image, ref_context), ValuePacket(moving, source_context)
    )
    assert result.passed, result.outputs["info"]
    assert result.outputs["image"].dtype == depth and result.outputs["image"].color_format == "rgba"
    for name in ("image", "transform", "valid_mask", "reference"):
        assert result.payloads[name]["ticket"] == source_context["ticket"]
        assert result.payloads[name]["spatial"] == ref_context["spatial"]
        assert result.payloads[name]["pyramid"] == ref_context["pyramid"]
    assert result.payloads["image"]["fixture"] == source_context["fixture"]
    assert result.payloads["image"]["registration_source"]["spatial"] == source_context["spatial"]
    assert "alpha_representation" not in result.payloads["valid_mask"]
    mapper = builtin_registry().create(
        "cv.transform_points", parameters={"pixel_points": [[96, 96]]}
    )
    for name, invert, expected in (
        ("transform", False, ref_context["spatial"]),
        ("transform", True, source_context["spatial"]),
        ("geometry", False, source_context["spatial"]),
        ("geometry", True, ref_context["spatial"]),
    ):
        mapper.parameters["invert"] = invert
        mapped = mapper.execute({"transform": result.outputs[name]})
        assert mapped is not None, mapper.error
        assert mapped.payloads["points"]["spatial"] == expected
        assert (
            mapped.payloads["points"]["processing_time"]["started_ns"]
            == source_context["processing_time"]["started_ns"]
        )


def test_reference_selection_and_moving_support_use_their_own_grids_and_do_not_change_the_seed():
    reference = part()
    generated = transform(reference.shape, 17, 0.9, (3, 2))
    moving = cv2.warpAffine(reference, generated[:2], (192, 192))
    mask = np.zeros(reference.shape, np.uint8)
    mask[35:150, 35:155] = 255
    region = {
        "model": "image_region_v1",
        "image_size": [192, 192],
        "shape": "rectangle",
        "rectangle": [45, 40, 100, 100],
    }
    support = np.full(reference.shape, 255, np.uint8)
    support[:8] = 0
    _, result = register(reference, moving, mask=mask, region=region, moving_mask=support)
    assert result.passed, result.outputs["info"]
    assert not result.outputs["valid_mask"][:40].any()
    wrong = ImageArray(mask, "gray", {"spatial": frame([192, 192], "wrong")})
    ref = ImageArray(reference, "gray", {"spatial": frame([192, 192], "reference")})
    block = builtin_registry().create(TYPE)
    assert block.execute({"image": moving, "reference": ref, "mask": wrong}) is None
    assert block.error_kind == "invalid"


@pytest.mark.parametrize("amplitude", (1e300, 1e-300))
def test_hdr_and_tiny_samples_are_scaled_before_native_float32_refinement(amplitude):
    reference = part()
    generated = transform(reference.shape, -21, 1.1, (2, 3))
    moving = cv2.warpAffine(reference, generated[:2], reference.shape[::-1])
    _, result = register(reference * amplitude, moving * amplitude)
    assert result.passed, result.outputs["info"]
    assert result.outputs["scale"] == pytest.approx(1 / 1.1, abs=0.002)
    assert np.isfinite(result.outputs["image"]).all()


def test_explicit_coarse_only_and_refinement_failure_policy(monkeypatch):
    import opencv_dataflow.logpolar_registration as implementation

    reference = part()
    _, coarse = register(reference, reference, {"ecc_iterations": 0})
    assert coarse.passed

    def fail(*args, **kwargs):
        raise cv2.error("intentional native refinement failure")

    monkeypatch.setattr(implementation.cv2, "findTransformECCWithMask", fail)
    _, rejected = register(reference, reference)
    assert not rejected.passed and rejected.outputs["transform"] is NO_VALUE
    _, allowed = register(reference, reference, {"refinement_failure": "coarse"})
    assert allowed.passed


@pytest.mark.parametrize(
    "shift,response", (((float("nan"), 0), 0.9), ((1e12, 0), 0.9), ((0, 0), float("inf")))
)
def test_unusable_native_proposals_never_escape_as_primary_geometry(monkeypatch, shift, response):
    monkeypatch.setattr(cv2, "phaseCorrelate", lambda *args: (shift, response))
    image = part()
    _, result = register(image, image)
    assert not result.passed and result.outputs["transform"] is NO_VALUE
    json.dumps(result.outputs["info"], allow_nan=False)


@pytest.mark.parametrize(
    "parameters",
    (
        {"minimum_scale": 2, "maximum_scale": 1},
        {"minimum_angle": 30, "maximum_angle": -30},
        {"ecc_gaussian": 4},
        {"radius_fraction": 0.9},
        {"work_size": 32},
    ),
)
def test_invalid_search_parameters_use_common_validation(parameters):
    image = part()
    block = builtin_registry().create(TYPE, parameters=parameters)
    assert block.execute({"image": image, "reference": image}) is None
    assert block.error_kind == "invalid"


def test_graphics_off_avoids_object_creation_and_xml_preserves_every_control(tmp_path, monkeypatch):
    import opencv_dataflow.logpolar_registration as implementation

    def unexpected(*args, **kwargs):
        raise AssertionError("Unexpected graphics")

    monkeypatch.setattr(implementation, "graphic_object", unexpected)
    image = part()
    block = builtin_registry().create(TYPE)
    block.set_graphics_mode("off")
    result = block.execute({"image": image, "reference": image})
    assert result is not None and result.passed, block.error
    assert not result.drawings
    graph = Graph()
    graph.add(block)
    try:
        expected = graph.to_dict()
        path = tmp_path / "registration.xml"
        export_project_xml(expected, path)
        restored = Graph.from_dict(read_project_xml(path), builtin_registry())
        try:
            assert restored.to_dict() == expected
        finally:
            restored.close()
    finally:
        graph.close()


@pytest.mark.local_examples
@pytest.mark.parametrize(
    "filename,passed",
    (("project.json", True), ("symmetric.json", False), ("orientation_prior.json", True)),
)
def test_public_examples_validate_analytic_truth_and_deliberate_ambiguity(filename, passed):
    folder = Path(__file__).parents[1] / "examples/opencv tools/logpolar_registration"
    truth = json.loads((folder / "assets/truth.json").read_text(encoding="utf-8"))
    graph = Graph.load(folder / filename, builtin_registry())
    try:
        graph.run()
        result = graph.blocks["register"].last_result
        assert result is not None and result.passed == passed
        if filename == "project.json":
            found = np.asarray(result.outputs["transform"]["source_to_target"])
            forward = np.asarray(truth["reference_to_moving"])
            controls = np.c_[truth["controls_reference"], np.ones(4)].T
            error = np.linalg.norm((found @ forward @ controls - controls)[:2], axis=0)
            assert max(error) < truth["maximum_control_error_pixels"]
            assert graph.blocks["verify"].last_result.passed
            assert (
                graph.blocks["before"].last_result.outputs["value"]
                > 5 * graph.blocks["verify"].last_result.outputs["value"]
            )
        elif not passed:
            assert result.outputs["info"]["status"] == "registration_ambiguous"
            assert result.outputs["transform"] is result.outputs["image"] is NO_VALUE
        else:
            assert abs(result.outputs["angle"]) < 0.1
    finally:
        graph.close()
