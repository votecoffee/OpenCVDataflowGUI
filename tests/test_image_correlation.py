"""Aligned comparison against independent Pearson and native masked references."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock


def run(image, reference, **parameters):
    block = builtin_registry().create("inspect.image_correlation", parameters=parameters)
    result = block.execute({"image": image, "reference": reference})
    assert result is not None, block.error
    return block, result, result.outputs["measurements"]


@pytest.mark.parametrize("method", ["ecc", "normalized_correlation"])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
def test_matches_pearson_and_native_masked_reference(method, dtype):
    rng = np.random.default_rng(119)
    reference = rng.integers(0, 180, (24, 30)).astype(dtype)
    candidate = reference.copy()
    candidate[8:16, 10:22] = 20
    mask = np.zeros(reference.shape, np.uint8)
    mask[4:21, 6:27] = 255
    region = {
        "model": "image_region_v1",
        "image_size": [30, 24],
        "shape": "rectangle",
        "rectangle": [8, 6, 14, 12],
    }
    selected = np.zeros(reference.shape, bool)
    selected[6:18, 8:22] = True
    block = builtin_registry().create(
        "inspect.image_correlation",
        parameters={
            "method": method,
            "emit_residual": True,
            "minimum_similarity": -1,
        },
    )
    result = block.execute(
        {"image": candidate, "reference": reference, "mask": mask, "region": region}
    )
    assert result is not None, block.error
    a, b = candidate[selected].astype(float), reference[selected].astype(float)
    expected = np.corrcoef(a, b)[0, 1]
    assert result.outputs["similarity"] == pytest.approx(expected, abs=1e-12)
    native = cv2.computeECC(
        reference.astype(np.float64), candidate.astype(np.float64), selected.astype(np.uint8)
    )
    assert result.outputs["similarity"] == pytest.approx(native, abs=1e-12)
    residual = result.outputs["residual"]
    np.testing.assert_allclose(
        residual[selected],
        np.abs((a - a.mean()) / a.std() - (b - b.mean()) / b.std()),
        rtol=1e-6,
        atol=1e-6,
    )
    assert not residual[~selected].any()
    assert result.outputs["measurements"]["selected_pixels"] == int(selected.sum())
    assert result.outputs["image"].dtype == dtype
    np.testing.assert_array_equal(result.outputs["image"], candidate)


def test_per_channel_aggregation_and_grayscale_mode():
    gray = np.arange(600, dtype=np.float64).reshape(20, 30)
    reference = np.dstack((gray, gray, gray))
    candidate = reference.copy()
    candidate[..., 2] = 599 - gray
    block, result, report = run(candidate, reference, minimum_similarity=-1)
    assert result.outputs["similarity"] == pytest.approx(-1)
    assert [c["channel"] for c in report["channels"]] == ["B", "G", "R"]
    assert [c["coefficient"] for c in report["channels"]] == pytest.approx([1, 1, -1])
    block.parameters["aggregation"] = "mean"
    assert block.execute({"image": candidate, "reference": reference}).outputs[
        "similarity"
    ] == pytest.approx(1 / 3)
    _, gray_result, gray_report = run(gray, reference, channels="gray")
    assert gray_result.passed and gray_report["channels"][0]["channel"] == "Gray"
    _, gray_result, _ = run(candidate, reference, channels="gray")
    assert gray_result.outputs["similarity"] == pytest.approx(1)


def test_canonical_color_ticket_frame_and_graphics_optimization():
    rng = np.random.default_rng(88)
    reference = rng.integers(200, 3000, (18, 24, 3), dtype=np.uint16)
    rgba = ImageArray(cv2.cvtColor(reference, cv2.COLOR_BGR2RGBA), "rgba")
    ticket = TicketClock().issue("part")
    spatial = frame([24, 18], "candidate")
    source = ValuePacket(rgba, {"ticket": ticket, "spatial": spatial})
    target = ValuePacket(reference, {"spatial": frame([24, 18], "reference")})
    block, result, report = run(source, target, emit_residual=True)
    assert result.passed and report["similarity"] == 1
    assert result.outputs["image"].color_format == "rgba"
    assert result.payloads["image"]["ticket"] == ticket
    assert result.payloads["similarity"]["ticket"] == ticket
    assert result.payloads["residual"]["spatial"] == spatial
    assert "normalized residual" in result.graphics
    assert result.drawings["aligned image comparison"].objects[0].kind == "label"
    block.set_graphics_mode("off")
    block.parameters["emit_residual"] = False
    lean = block.execute({"image": source, "reference": target})
    assert lean.passed and not lean.graphics and not lean.drawings
    assert lean.outputs["residual"] is NO_VALUE
    block.parameters["reference_alignment"] = "same_frame"
    assert block.execute({"image": source, "reference": target}) is None
    assert "matching spatial payloads" in block.error
    target = ValuePacket(reference, {"spatial": spatial})
    assert block.execute({"image": source, "reference": target}).passed


@pytest.mark.parametrize("method", ["ecc", "normalized_correlation"])
def test_constant_policy_is_explicit_and_never_silently_matches_undefined_channels(method):
    image = np.full((8, 9), 20, np.uint8)
    block, result, report = run(image, image, method=method, check_threshold=False)
    assert not result.passed and not report["valid"]
    assert result.outputs["similarity"] is NO_VALUE
    assert result.fail_reasons[0]["code"] == "correlation_undefined"
    block.parameters.update(
        constant_policy="match_equal", check_threshold=True, minimum_similarity=1
    )
    matched = block.execute({"image": image, "reference": image})
    assert matched.passed and matched.outputs["similarity"] == 1
    channel = matched.outputs["measurements"]["channels"][0]
    assert channel["coefficient"] is None and channel["state"] == "constant_equal"
    mismatch = block.execute({"image": image + 1, "reference": image})
    assert not mismatch.passed
    assert mismatch.fail_reasons[0]["code"] == "correlation_constant_mismatch"
    block.parameters["constant_tolerance"] = 1
    assert block.execute({"image": image + 1, "reference": image}).passed
    textured = image.copy()
    textured[2:4] += 1
    mixed = block.execute({"image": image, "reference": textured})
    assert not mixed.passed and mixed.outputs["similarity"] is NO_VALUE


def test_opaque_alpha_is_ignored_unless_requested():
    gray = np.arange(100, dtype=np.uint8).reshape(10, 10)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGRA)
    block, result, report = run(image, image)
    assert result.passed and len(report["channels"]) == 3
    block.parameters["include_alpha"] = True
    assert not block.execute({"image": image, "reference": image}).passed
    block.parameters["constant_policy"] = "match_equal"
    assert block.execute({"image": image, "reference": image}).passed


@pytest.mark.parametrize(
    "encoding,depth,forward,backward",
    [
        ("rgb", np.uint8, cv2.COLOR_BGR2RGB, cv2.COLOR_RGB2BGR),
        ("hsv", np.float32, cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2BGR),
        ("lab", np.uint8, cv2.COLOR_BGR2Lab, cv2.COLOR_Lab2BGR),
    ],
)
def test_tagged_encodings_compare_in_canonical_color_order(encoding, depth, forward, backward):
    rng = np.random.default_rng(129)
    bgr = rng.integers(20, 230, (12, 18, 3), dtype=np.uint8).astype(depth)
    if depth == np.float32:
        bgr /= 255
    encoded = ImageArray(cv2.cvtColor(bgr, forward), encoding)
    reference = cv2.cvtColor(np.asarray(encoded), backward)
    _, result, report = run(encoded, reference)
    assert result.passed and report["similarity"] == 1
    assert result.outputs["image"].color_format == encoding


@pytest.mark.parametrize("scale", [1e-300, 1e300])
def test_extreme_floating_ranges_do_not_overflow_or_lose_contrast(scale):
    reference = np.linspace(-1, 1, 150).reshape(10, 15) * scale
    _, result, report = run(reference * -0.5, reference, minimum_stddev=0, minimum_similarity=-1)
    assert result.passed and result.outputs["similarity"] == pytest.approx(-1)
    assert np.isfinite(report["channels"][0]["image_stddev"])


def test_brightness_gain_offset_and_inclusive_threshold():
    x = np.arange(300, dtype=np.float64).reshape(15, 20)
    candidate = x * 1.2 + 30
    block, result, _ = run(candidate, x, minimum_similarity=1)
    assert result.passed  # Correlation intentionally ignores positive gain and offset.
    candidate[4:8, 5:15] += 200
    block.parameters["minimum_similarity"] = -1
    score = block.execute({"image": candidate, "reference": x}).outputs["similarity"]
    block.parameters["minimum_similarity"] = score
    assert block.execute({"image": candidate, "reference": x}).passed
    block.parameters["minimum_similarity"] = score + 0.001
    failed = block.execute({"image": candidate, "reference": x})
    assert not failed.passed
    assert failed.fail_reasons[0]["code"] == "image_correlation_below_minimum"


def test_validation_of_selection_dimensions_encoding_frames_and_parameter_limits():
    image = np.arange(100, dtype=np.uint8).reshape(10, 10)
    block = builtin_registry().create("inspect.image_correlation")
    assert block.execute({"image": image, "reference": image[:5]}) is None
    assert "matching image dimensions" in block.error
    for count in (0, 1):
        mask = np.zeros(image.shape, np.uint8)
        mask.ravel()[:count] = 255
        assert block.execute({"image": image, "reference": image, "mask": mask}) is None
        assert "two selected pixels" in block.error
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    assert block.execute({"image": image, "reference": color}) is None
    assert "matching channels" in block.error
    block.parameters["minimum_similarity"] = 2
    assert block.execute({"image": image, "reference": image}) is None
    assert "Minimum similarity" in block.error
    block.parameters["minimum_similarity"] = 0.95
    source = ValuePacket(image, {"spatial": frame([10, 10], "source")})
    mask = ValuePacket(np.ones(image.shape, np.uint8), {"spatial": frame([10, 10], "other")})
    assert block.execute({"image": source, "reference": image, "mask": mask}) is None
    assert "coordinate frame differs" in block.error
    nonfinite = image.astype(np.float64)
    nonfinite[2, 3] = np.nan
    assert block.execute({"image": nonfinite, "reference": image}) is None
    assert "finite" in block.error
    block.parameters["reference_alignment"] = "same_frame"
    assert block.execute({"image": image, "reference": image}) is None
    assert "matching spatial payloads" in block.error


def test_public_example_has_known_pass_and_defect_fail():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples/opencv tools/aligned_image_comparison/project.json"
    )
    graph = Graph.load(path, builtin_registry())
    try:
        for expected in (True, False):
            graph.run()
            result = graph.blocks["compare"].last_result
            assert result is not None and result.passed == expected
            assert result.outputs["image"].dtype == np.uint8
    finally:
        graph.close()
