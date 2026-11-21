"""Qualify perceptual descriptor encoding, settings and score direction."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, ValuePacket
from opencv_dataflow.tickets import TicketClock

EXAMPLE = Path(__file__).parents[1] / "examples/opencv tools/perceptual_descriptors"
COMPARE = "inspect.compare_perceptual_descriptors"


@pytest.fixture(scope="module")
def color_pair():
    rng = np.random.default_rng(6419)
    ref = rng.integers(0, 256, (57, 79, 4), dtype=np.uint8)
    return ref, np.roll(ref, (11, 7), axis=(0, 1))


def test_color_moments_preserve_float64_native_values_and_scaled_distance(color_pair):
    a, b = color_pair
    registry = builtin_registry()
    compute = registry.create("cv.image_hash", parameters={"method": "color_moment"})
    sample_ticket = TicketClock("test").issue("camera", "fixed", "sample")
    sample = compute.execute(
        {"image": ValuePacket(ImageArray(a[..., [2, 1, 0, 3]], "rgba"), {"ticket": sample_ticket})}
    )
    ref = compute.execute({"image": b})
    assert sample and ref, compute.error
    records = [
        json.loads(json.dumps(item.outputs["hash"], allow_nan=False)) for item in (sample, ref)
    ]
    native = cv2.img_hash.ColorMomentHash_create()
    lhs, rhs = native.compute(a), native.compute(b)
    assert lhs.dtype == np.float64 and lhs.shape == (1, 42)
    np.testing.assert_array_equal(records[0]["values"], lhs[0])
    np.testing.assert_array_equal(records[1]["values"], rhs[0])
    measured = float(np.linalg.norm(lhs - rhs) * 10000)
    assert measured > 0
    compare = registry.create(
        COMPARE,
        parameters={
            "check_threshold": True,
            "maximum_color_distance": native.compare(lhs, rhs),
        },
    )
    inputs = {"sample": ValuePacket(records[0], sample.payloads["hash"]), "reference": records[1]}
    checked = compare.execute(inputs)
    assert checked and checked.passed, compare.error
    assert checked.outputs["distance"] == pytest.approx(measured)
    assert checked.outputs["similarity"] is NO_VALUE
    assert checked.outputs["measurements"]["direction"] == "lower"
    assert checked.payloads["distance"]["ticket"] == sample_ticket

    compare.parameters["maximum_color_distance"] = measured * 0.9
    rejected = compare.execute(inputs)
    assert rejected and not rejected.passed
    assert rejected.fail_reasons[0]["code"] == "perceptual_hash_distance_outside_limit"


def test_radial_preserves_byte_coefficients_and_cyclic_correlation(color_pair):
    a, b = color_pair
    radius = {"method": "radial_variance", "radial_sigma": 1.4, "radial_angle_lines": 127}
    hasher = builtin_registry().create("cv.image_hash", parameters=radius)
    packets = [hasher.execute({"image": image}) for image in (a, b)]
    assert all(packets), hasher.error
    first, second = [item.outputs["hash"] for item in packets]
    assert first["options"] == {"radial_sigma": 1.4, "radial_angle_lines": 127}
    assert not first["uniform_input"]
    factory = cv2.img_hash.RadialVarianceHash_create(1.4, 127)
    ra, rb = factory.compute(a), factory.compute(b)
    assert ra.dtype == np.uint8 and ra.shape == (1, 40)
    np.testing.assert_array_equal(first["values"], ra[0])
    np.testing.assert_array_equal(second["values"], rb[0])
    x, y = ra.flatten().astype(float), rb.flatten().astype(float)
    x, y = x - x.mean(), y - y.mean()
    expected = max(x @ np.roll(y, i) for i in range(40)) / np.linalg.norm(x) / np.linalg.norm(y)
    native = factory.compare(ra, rb)
    assert native == pytest.approx(expected, abs=1e-7)
    compare = builtin_registry().create(
        COMPARE,
        parameters={
            "check_threshold": True,
            "minimum_radial_similarity": native,
        },
    )
    result = compare.execute({"sample": first, "reference": second})
    assert result and result.passed, compare.error
    assert result.outputs["distance"] is NO_VALUE
    assert result.outputs["similarity"] == pytest.approx(expected, abs=1e-7)
    assert result.outputs["measurements"]["direction"] == "higher"
    assert result.outputs["measurements"]["native_value"] == native

    compare.parameters["minimum_radial_similarity"] = 1.0
    distinct = compare.execute({"sample": first, "reference": second})
    assert distinct and not distinct.passed
    assert distinct.fail_reasons[0]["code"] == "perceptual_hash_similarity_outside_limit"
    identical = compare.execute({"sample": first, "reference": first})
    assert identical and identical.passed and identical.outputs["similarity"] == 1.0


def test_radial_rejects_uniform_images_and_constant_descriptors(color_pair):
    registry = builtin_registry()
    compute = registry.create("cv.image_hash", parameters={"method": "radial_variance"})
    regular = compute.execute({"image": color_pair[0]}).outputs["hash"]
    compare = registry.create(COMPARE)
    for level in (0, 128):
        flat = compute.execute({"image": np.full((48, 48), level, np.uint8)}).outputs["hash"]
        assert flat["uniform_input"]
        result = compare.execute({"sample": flat, "reference": regular})
        assert result and not result.passed
        assert result.outputs["similarity"] is NO_VALUE
        assert result.fail_reasons[0]["code"] == "radial_hash_degenerate"
    zero_descriptor = {**regular, "values": [0] * 40}
    result = compare.execute({"sample": zero_descriptor, "reference": regular})
    assert result and not result.passed
    assert result.outputs["measurements"]["status"] == "insufficient_radial_variation"


def test_descriptor_validation_rejects_mixed_records_and_bad_coefficients(color_pair):
    registry = builtin_registry()
    image = color_pair[0]
    compute = registry.create("cv.image_hash", parameters={"method": "radial_variance"})
    radial = compute.execute({"image": image}).outputs["hash"]
    compute.parameters["method"] = "color_moment"
    color = compute.execute({"image": image}).outputs["hash"]
    compare = registry.create(COMPARE)
    for sample, reference, reason in (
        (color, radial, "same method and settings"),
        (
            radial,
            {**radial, "options": {**radial["options"], "radial_angle_lines": 90}},
            "same method and settings",
        ),
        (radial, {**radial, "values": [False] * 40}, "byte coefficients"),
        (radial, {**radial, "values": radial["values"][:-1]}, "coefficient count"),
        (color, {**color, "values": [1e308] * 42}, "exceeds the native range"),
    ):
        assert compare.execute({"sample": sample, "reference": reference}) is None
        assert reason in compare.error
    legacy = registry.create("cv.image_hash").execute({"image": image}).outputs["hash"]
    assert compare.execute({"sample": legacy, "reference": color}) is None
    hamming = registry.create("inspect.compare_image_hashes")
    assert hamming.execute({"sample": color, "reference": color}) is None
    compute.parameters.update(method="radial_variance", radial_angle_lines=1440)
    assert compute.execute({"image": np.zeros((3, 8192), np.uint8)}) is None
    assert "projection angle count" in compute.error


@pytest.mark.local_examples
@pytest.mark.parametrize(
    ("method", "profile"),
    [
        ("color_moment", "debug"),
        ("radial_variance", "lean"),
    ],
)
def test_saved_perceptual_hash_projects_distinguish_changed_part(method, profile):
    graph = Graph.load(EXAMPLE / f"{method}.json", builtin_registry())
    graph.set_branch_workers(2)
    graph.set_run_profile(profile)
    try:
        graph.run()
        values = {}
        for key in ("reference", "rotated", "resampled", "changed"):
            tool = graph.blocks[f"{key}_compare"]
            result = tool.last_result
            assert result and result.passed, tool.error
            values[key] = result.outputs["measurements"]["value"]
        if method == "color_moment":
            assert values["reference"] == 0
            assert max(values["rotated"], values["resampled"]) < 0.1 * values["changed"]
        else:
            assert values["reference"] == 1
            assert values["resampled"] > 0.98
            assert values["rotated"] > values["changed"] + 0.1
    finally:
        graph.close()
