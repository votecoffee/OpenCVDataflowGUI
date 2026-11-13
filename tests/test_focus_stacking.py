"""Independent sharp-image truth, analytic derivatives and fusion support contracts."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket, frame
from opencv_dataflow.project_xml import export_project_xml, read_project_xml
from opencv_dataflow.tickets import TicketClock

TYPE = "cv.focus_stack"


def batch(images, payloads=None):
    payloads = payloads or [{} for _ in images]
    return FlowGroup(
        tuple(ValuePacket(a, p) for a, p in zip(images, payloads, strict=True)),
        tuple(f"level-{i}" for i in range(len(images))),
    )


def fuse(images, parameters=None, **extra):
    block = builtin_registry().create(TYPE, parameters=parameters or {})
    result = block.execute(
        {
            "images": images if isinstance(images, (FlowGroup, ValuePacket)) else batch(images),
            **extra,
        }
    )
    assert result is not None, block.error
    return block, result


def focal_pair(depth=np.uint16):
    y, x = np.mgrid[:64, :128]
    truth = (((x // 2 + y // 2) % 2) * 24000 + 8000).astype(np.uint16)
    blurred = cv2.GaussianBlur(truth, (7, 7), 1.5)
    a, b = blurred.copy(), blurred.copy()
    a[:, :80], b[:, 48:] = truth[:, :80], truth[:, 48:]
    if np.dtype(depth).kind == "u":
        divisor = 257 if depth == np.uint8 else 1
    else:
        divisor = 65535
    return tuple((image / divisor).astype(depth) for image in (truth, a, b))


@pytest.mark.parametrize("depth", (np.uint8, np.uint16, np.float32, np.float64))
@pytest.mark.parametrize("method", ("tenengrad", "laplacian_energy"))
def test_hard_fusion_matches_independent_sharp_truth_without_changing_source_samples(depth, method):
    truth, a, b = focal_pair(depth)
    mask = np.zeros(a.shape, np.uint8)
    mask[10:-10, 10:-10] = 255
    _, result = fuse([a, b], {"method": method}, mask=mask)
    assert result.passed
    output, valid = result.outputs["image"], result.outputs["valid_mask"] != 0
    np.testing.assert_array_equal(output[mask != 0], truth[mask != 0])
    np.testing.assert_array_equal(output[~valid], a[~valid])
    assert output.dtype == depth
    assert np.any(result.outputs["focus_index"] == 0) and np.any(result.outputs["focus_index"] == 1)
    assert np.all(result.outputs["focus_index"][~valid] == -1)
    assert result.outputs["focus_index"].dtype == np.int32


@pytest.mark.parametrize("method", ("tenengrad", "laplacian_energy"))
def test_local_energy_has_an_independent_analytic_derivative_reference(method):
    x = np.arange(32)
    image = np.broadcast_to(x * 1000 if method == "tenengrad" else x * x * 10, (24, 32)).astype(
        np.uint16
    )
    aperture = "3" if method == "tenengrad" else "1"
    _, result = fuse(
        [np.full_like(image, 5000), image],
        {"method": method, "window": 1, "aperture": aperture, "minimum_energy": 0},
    )
    expected = (8000 / 65535) ** 2 if method == "tenengrad" else (20 / 65535) ** 2
    assert result.outputs["response"][12, 16] == pytest.approx(expected, rel=1e-11)
    assert result.outputs["focus_index"][12, 16] == 1


@pytest.mark.parametrize("power,radius", ((1.0, 0), (2.0, 3)))
@pytest.mark.parametrize("scale", (1.0, 1e300, 1e-300))
def test_weighted_fusion_is_a_bounded_convex_mix_including_hdr_and_tiny_values(
    power, radius, scale
):
    image = np.broadcast_to(np.linspace(0.1, 0.9, 64), (40, 64)) * scale
    _, result = fuse(
        [image, image * 0.5], {"blend": "weighted", "blend_power": power, "weight_radius": radius}
    )
    valid = result.outputs["valid_mask"] != 0
    ratio = (1 + (0.25**power) * 0.5) / (1 + 0.25**power)
    np.testing.assert_allclose(
        result.outputs["image"][valid], image[valid] * ratio, rtol=1e-12, atol=0
    )
    assert np.isfinite(result.outputs["image"]).all()
    assert np.all(result.outputs["focus_index"][valid] == 0)


def test_flat_or_tied_stacks_do_not_invent_a_focus_level_and_ties_can_prefer_anchor():
    image = np.full((30, 40), 100, np.uint8)
    _, flat = fuse([image, image])
    assert not flat.passed and np.all(flat.outputs["focus_index"] == -1)
    np.testing.assert_array_equal(flat.outputs["image"], image)
    _, a, _ = focal_pair()
    _, tied = fuse([a, a])
    assert not tied.passed
    _, allowed = fuse([a, a], {"minimum_confidence": 0, "anchor_index": 1})
    assert allowed.passed
    assert np.all(allowed.outputs["focus_index"][allowed.outputs["valid_mask"] != 0] == 1)


def test_coverage_holes_exclude_entire_scoring_neighborhoods_and_single_contributor_is_explicit():
    _, a, b = focal_pair()
    coverage = [np.full(a.shape, 255, np.uint8) for _ in range(2)]
    coverage[1][20:30, 20:30] = 0
    _, result = fuse([a, b], coverage=batch(coverage))
    # Halo 5 includes derivative support and the 9x9 energy window.
    assert np.all(result.outputs["contributor_count"][15:35, 15:35] == 1)
    assert not np.any(result.outputs["valid_mask"][15:35, 15:35])
    _, one = fuse([a, b], {"minimum_contributors": 1}, coverage=batch(coverage))
    assert one.outputs["valid_mask"][24, 24] == 255
    assert one.outputs["confidence"][24, 24] == 1


@pytest.mark.parametrize("support,count", (("strict", 0), ("extrapolate", 2)))
def test_image_edge_extrapolation_is_an_explicit_support_policy(support, count):
    _, a, b = focal_pair()
    _, result = fuse([a, b], {"support": support})
    assert result.outputs["contributor_count"][0, 0] == count


def test_mask_and_region_select_centers_without_cutting_the_focus_window():
    _, a, b = focal_pair()
    _, full = fuse([a, b])
    mask = np.zeros(a.shape, np.uint8)
    mask[15:45, 15:40] = 255
    region = {
        "model": "image_region_v1",
        "image_size": [128, 64],
        "shape": "rectangle",
        "rectangle": [25, 20, 30, 20],
    }
    _, result = fuse([a, b], mask=mask, region=region)
    selected = np.zeros_like(mask, bool)
    selected[20:40, 25:40] = True
    np.testing.assert_array_equal(
        result.outputs["response"][selected], full.outputs["response"][selected]
    )
    np.testing.assert_array_equal(result.outputs["image"][~selected], a[~selected])
    assert result.outputs["info"]["selected_pixels"] == 300


def test_encoding_alpha_and_batch_ticket_context_are_preserved_without_scoring_alpha():
    _, a, b = focal_pair()
    clock = TicketClock()
    context = {
        "spatial": frame([128, 64], "camera"),
        "pyramid": {
            "level": 1,
            "largest_size": [256, 128],
            "to_largest": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        },
        "fixture": {"lot": "L"},
        "alpha_representation": "straight",
    }
    alpha = np.full(a.shape, 12345, np.uint16)
    images = [ImageArray(np.dstack([im, im // 2, im // 4, alpha]), "rgba") for im in (a, b)]
    contexts = [{**context, "ticket": clock.issue(str(i))} for i in range(2)]
    cohort = clock.issue("focus-batch")
    group = ValuePacket(batch(images, contexts), {"ticket": cohort})
    _, result = fuse(group)
    np.testing.assert_array_equal(result.outputs["image"][..., 3], alpha)
    assert result.outputs["image"].color_format == "rgba"
    for name in ("image", "focus_index", "valid_mask"):
        assert result.payloads[name]["ticket"] == cohort
        assert result.payloads[name]["pyramid"] == context["pyramid"]
        assert result.payloads[name]["fixture"] == context["fixture"]
    assert "alpha_representation" not in result.payloads["focus_index"]
    assert result.payloads["source"]["ticket"] == contexts[0]["ticket"]
    assert result.drawings["focus coverage"].from_output


def test_rgb_and_bgr_luminance_have_the_same_focus_decisions():
    _, a, b = focal_pair()
    rgb = [ImageArray(np.dstack([im, im // 2, im // 4]), "rgb") for im in (a, b)]
    bgr = [ImageArray(np.asarray(im)[..., ::-1], "bgr") for im in rgb]
    _, first = fuse(rgb)
    _, second = fuse(bgr)
    np.testing.assert_array_equal(first.outputs["focus_index"], second.outputs["focus_index"])
    np.testing.assert_array_equal(first.outputs["image"], second.outputs["image"][..., ::-1])


def test_known_frame_mismatch_requires_pixel_grid_and_selection_still_checks_its_frame():
    _, a, b = focal_pair()
    group = batch([a, b], [{"spatial": frame([128, 64], name)} for name in ("a", "b")])
    block = builtin_registry().create(TYPE)
    assert block.execute({"images": group}) is None and block.error_kind == "invalid"
    _, result = fuse(group, {"alignment": "pixel_grid"})
    assert result.passed and result.payloads["image"]["spatial"]["root_id"] == "a"
    mask = ImageArray(
        np.full_like(a, 255, np.uint8), "gray", {"spatial": frame([128, 64], "wrong")}
    )
    block.parameters["alignment"] = "pixel_grid"
    assert block.execute({"images": group, "mask": mask}) is None


@pytest.mark.parametrize(
    "parameters",
    (
        {"window": 4},
        {"window": 3.5},
        {"anchor_index": 2},
        {"minimum_contributors": 3},
        {"minimum_energy": -1},
    ),
)
def test_invalid_parameters_are_reported_by_the_common_validation_contract(parameters):
    _, a, b = focal_pair()
    block = builtin_registry().create(TYPE, parameters=parameters)
    assert block.execute({"images": batch([a, b])}) is None and block.error_kind == "invalid"


def test_invalid_capture_and_coverage_layouts_are_rejected():
    _, a, b = focal_pair()
    block = builtin_registry().create(TYPE)
    for bad in (
        b.astype(np.float32),
        b[:, :-1],
        np.full_like(b, 1, dtype=np.int16),
        np.full(b.shape, np.nan),
    ):
        assert block.execute({"images": batch([a, bad])}) is None
        assert block.error_kind == "invalid"
    assert (
        block.execute(
            {
                "images": batch([a, b]),
                "coverage": batch([np.zeros_like(a, np.float32), np.zeros_like(b, np.uint8)]),
            }
        )
        is None
    )
    assert block.execute({"images": batch([a, b]), "mask": np.zeros_like(a, np.uint8)}) is None


def test_graphics_off_skips_colorization_and_drawing(monkeypatch):
    import opencv_dataflow.focus_stacking as implementation

    def unexpected(*args, **kwargs):
        raise AssertionError("Unexpected graphics work")

    monkeypatch.setattr(implementation.cv2, "applyColorMap", unexpected)
    monkeypatch.setattr(implementation, "graphic_object", unexpected)
    _, a, b = focal_pair()
    block = builtin_registry().create(TYPE)
    block.set_graphics_mode("off")
    result = block.execute({"images": batch([a, b])})
    assert result is not None and result.passed, block.error
    assert result.outputs["preview"] is NO_VALUE and not result.graphics and not result.drawings


def test_xml_roundtrip_preserves_all_controls_and_documentation(tmp_path):
    registry, graph = builtin_registry(), Graph()
    try:
        graph.add(
            registry.create(
                TYPE, parameters={"blend": "weighted", "weight_radius": 3, "prefilter": "5"}
            )
        )
        expected = graph.to_dict()
        path = tmp_path / "stack.xml"
        export_project_xml(expected, path)
        restored = Graph.from_dict(read_project_xml(path), registry)
        try:
            assert restored.to_dict() == expected
        finally:
            restored.close()
    finally:
        graph.close()


@pytest.mark.parametrize("sigma", (0.0, 1.5))
def test_prefiltered_ramp_keeps_analytic_energy_and_expands_the_support_halo(sigma):
    image = np.broadcast_to(np.arange(64) * 500, (40, 64)).astype(np.uint16)
    _, result = fuse(
        [np.zeros_like(image), image],
        {
            "prefilter": "5",
            "prefilter_sigma": sigma,
            "window": 5,
        },
    )
    assert result.outputs["info"]["support_halo"] == 5
    assert result.outputs["response"][20, 30] == pytest.approx((4000 / 65535) ** 2, rel=1e-12)
    assert not np.any(result.outputs["valid_mask"][:5])


def test_resource_budget_is_checked_before_native_response_calculation(monkeypatch):
    import opencv_dataflow.focus_stacking as implementation

    def unexpected(*args, **kwargs):
        raise AssertionError("Oversized stack reached filtering")

    monkeypatch.setattr(implementation, "focus_energy", unexpected)
    image = np.zeros((2048, 2048), np.uint8)
    block = builtin_registry().create(TYPE)
    assert block.execute({"images": batch([image] * 5)}) is None
    assert block.error_kind == "invalid" and "16,777,216" in block.error


def test_alpha_and_encoding_validation_do_not_admit_false_focus_data():
    block = builtin_registry().create(TYPE)
    bad = ImageArray(np.ones((20, 30, 3), np.uint8), "hsv")
    assert block.execute({"images": batch([bad, bad])}) is None
    rgba = ImageArray(np.ones((20, 30, 4)), "rgba")
    assert (
        block.execute(
            {"images": batch([rgba, rgba], [{"alpha_representation": "premultiplied"}] * 2)}
        )
        is None
    )
    rgba[..., 3] = 1.1
    assert block.execute({"images": batch([rgba, rgba])}) is None


@pytest.mark.local_examples
@pytest.mark.parametrize("filename", ("project.json", "registered.json"))
def test_public_projects_reconstruct_independent_sharp_truth_and_known_alignment(filename):
    folder = Path(__file__).parents[1] / "examples/opencv tools/focus_stack"
    graph = Graph.load(folder / filename, builtin_registry())
    try:
        graph.run()
        assert all(block.state == "pass" for block in graph.processing_blocks())
        assert graph.blocks["verify"].last_result.outputs["value"] == 0
        assert graph.blocks["before"].last_result.outputs["value"] > 1000
        result = graph.blocks["focus"].last_result
        assert 0.3 < result.outputs["info"]["valid_fraction"] < 1
        assert result.outputs["image"].dtype == np.uint16
        assert result.outputs["image"].color_format == "rgb"
        assert np.any(result.outputs["focus_index"] == -1)
        if filename == "registered.json":
            transform = graph.blocks["align"].last_result.outputs["transform"]["source_to_target"]
            np.testing.assert_allclose(np.asarray(transform)[:2, 2], [-2, 1], atol=0.2)
    finally:
        graph.close()
