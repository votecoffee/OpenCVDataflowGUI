"""Exact radiance/support controls, independent weighted means and native alignment."""

import copy

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket, frame, packet_for
from opencv_dataflow.tickets import TicketClock


def tool(**parameters):
    return builtin_registry().create(
        "cv.merge_hdr_motion",
        parameters={"reference_index": 1, "exposure_times": [0.25, 0.5, 1.0], **parameters},
    )


def run(block, inputs):
    result = block.execute(inputs)
    assert result is not None, block.error
    return result


def fixture(moving=False, independent_roots=False):
    yy, xx = np.indices((48, 80))
    codes = (4 * (4 + (xx // 4 + yy // 3) % 53)).astype(np.uint16)
    radiance = codes.astype(np.float64) / 64
    clock = TicketClock("bracket")
    packets = []
    moving_mask = np.zeros(codes.shape, bool)
    for i, seconds in enumerate([0.25, 0.5, 1.0]):
        actual = radiance.copy()
        if moving:
            x = 8 + 20 * i
            actual[18:24, x : x + 8] = 3.75
            moving_mask[18:24, x : x + 8] = True
        pixels = np.rint(np.minimum(actual * 128 * seconds, 255)).astype(np.uint8)
        packets.append(
            ValuePacket(
                pixels,
                {
                    "spatial": frame([80, 48], f"camera-{i}" if independent_roots else "camera"),
                    "ticket": clock.issue("camera"),
                },
            )
        )
    group = FlowGroup(tuple(packets), ("short", "reference", "long"))
    return ValuePacket(group, {"ticket": clock.issue("batch")}), radiance, moving_mask


@pytest.mark.parametrize("weighting", ["uniform", "triangle"])
def test_exact_linear_radiance_survives_clipping_without_inventing_missing_measurements(weighting):
    batch, radiance, _ = fixture()
    block = tool(weighting=weighting, absolute_tolerance=0, relative_tolerance=0)
    result = run(block, {"exposures": batch})
    assert result.passed and result.score == 1
    np.testing.assert_array_equal(result.outputs["image"], radiance.astype(np.float32))
    assert np.all(result.outputs["valid_mask"] == 255) and not result.outputs["motion_mask"].any()
    assert not result.outputs["fallback_mask"].any()
    assert set(np.unique(result.outputs["contributors"])) == {2, 3}
    assert result.outputs["support_masks"] is NO_VALUE
    assert not block.current_emissions.get("support_masks")


def test_moving_part_union_is_rejected_but_reference_display_and_static_detail_are_kept_separate():
    batch, radiance, changed = fixture(moving=True)
    block = tool(absolute_tolerance=0, relative_tolerance=0, emit_support_masks=True)
    result = run(block, {"exposures": batch})
    report = result.outputs["report"]
    assert report["inconsistent_pixels"] == 144
    np.testing.assert_array_equal(result.outputs["motion_mask"] != 0, changed)
    np.testing.assert_array_equal(result.outputs["valid_mask"] != 0, ~changed)
    np.testing.assert_array_equal(result.outputs["image"][~changed], radiance[~changed])
    assert not result.outputs["image"][changed].any()
    reference = batch.value.items[1].value.astype(np.float32) / 64
    np.testing.assert_array_equal(result.outputs["display"][changed], reference[changed])
    np.testing.assert_array_equal(result.outputs["fallback_mask"] != 0, changed)
    assert not result.outputs["contributors"][changed].any()
    masks = result.outputs["support_masks"]
    assert masks.labels == batch.value.labels
    for mask, source in zip(masks.items, batch.value.items, strict=True):
        assert not mask.value[changed].any()
        assert mask.payload["ticket"] == source.payload["ticket"]
    for name in (
        "image",
        "display",
        "valid_mask",
        "motion_mask",
        "fallback_mask",
        "contributors",
        "report",
    ):
        assert result.payloads[name]["ticket"] == batch.payload["ticket"]
        assert result.payloads[name]["spatial"] == batch.value.items[1].payload["spatial"]
    assert result.payloads["display"]["hdr_support"]["review_only"] is True
    assert result.payloads["image"]["hdr_support"]["requires_valid_mask"] is True
    assert report["reconstructed_pixels_are_measured"] is False
    assert result.drawings["HDR rejected support"].source == "display"


def test_weighted_radiance_matches_independent_scalar_reference_and_response_channel_order():
    values = np.array([[20, 24, 28], [32, 36, 40]], np.uint8)
    frames = []
    times = [0.25, 0.5, 1.0]
    for i, time in enumerate(times):
        pixels = np.stack([values, values * 2, values * 3], axis=2).astype(float) * 2 * time
        pixels[0, 0, 0] += i
        frames.append(ImageArray(pixels.astype(np.uint8), "rgb"))
    group = FlowGroup(tuple(ValuePacket(p, {}) for p in frames), ("a", "b", "c"))
    table = {
        "model": "opencv_robertson_response_v1",
        "encoding": "bgr",
        "values": [[z / 128 * 3, z / 128 * 2, z / 128] for z in range(256)],
    }
    result = run(
        tool(absolute_tolerance=0.2, relative_tolerance=0), {"exposures": group, "response": table}
    )
    expected = np.zeros(frames[0].shape)
    for y in range(2):
        for x in range(3):
            weights = [min(int(p[y, x].min()) - 4, 251 - int(p[y, x].max())) for p in frames]
            for c in range(3):
                expected[y, x, c] = sum(
                    w * float(p[y, x, c]) / 128 * (c + 1) / t
                    for p, t, w in zip(frames, times, weights, strict=True)
                ) / sum(weights)
    np.testing.assert_allclose(result.outputs["image"], expected, rtol=1e-6)
    assert result.outputs["image"].color_format == "rgb"


def test_known_nonlinear_response_recovers_exact_radiance():
    q = np.tile(np.arange(16, 112, 4, dtype=np.uint8), (8, 1))
    frames = [q // 2, q, (q.astype(np.uint16) * 2).astype(np.uint8)]
    group = FlowGroup(tuple(ValuePacket(p, {}) for p in frames), ("a", "b", "c"))
    curve = {
        "model": "opencv_robertson_response_v1",
        "encoding": "gray",
        "values": [[(z / 128) ** 2] for z in range(256)],
    }
    result = run(
        tool(exposure_times=[0.25, 1.0, 4.0], absolute_tolerance=0, relative_tolerance=0),
        {"exposures": group, "response": curve},
    )
    assert result.passed
    np.testing.assert_array_equal(result.outputs["image"], (q.astype(np.float32) / 128) ** 2)


def test_masks_and_reference_region_intersect_and_output_masks_can_be_reused():
    batch, _, _ = fixture()
    group = batch.value
    masks = []
    for i, source in enumerate(group.items):
        mask = np.full(source.value.shape, 255, np.uint8)
        if i != 1:
            mask[8:12, 10:16] = 0
        masks.append(ValuePacket(mask, copy.deepcopy(source.payload)))
    region = {
        "model": "image_region_v1",
        "image_size": [80, 48],
        "shape": "rectangle",
        "rectangle": [4, 4, 50, 30],
    }
    block = tool(emit_support_masks=True)
    result = run(
        block,
        {"exposures": batch, "masks": FlowGroup(tuple(masks), group.labels), "region": region},
    )
    assert result.outputs["report"]["selected_pixels"] == 1500
    assert not result.outputs["valid_mask"][8:12, 10:16].any()
    assert not result.outputs["valid_mask"][:4].any()
    replay = run(
        block, {"exposures": batch, "masks": packet_for(result, "support_masks"), "region": region}
    )
    np.testing.assert_array_equal(replay.outputs["image"], result.outputs["image"])
    np.testing.assert_array_equal(replay.outputs["valid_mask"], result.outputs["valid_mask"])


def test_independent_roots_need_explicit_alignment_assertion_and_bad_mask_frames_do_not_pass():
    batch, _, _ = fixture(independent_roots=True)
    block = tool()
    assert block.execute({"exposures": batch}) is None
    assert "Align exposure grids" in block.error
    block.parameters["grid_policy"] = "assert_pixel_alignment"
    assert run(block, {"exposures": batch}).passed
    masks = FlowGroup(
        tuple(
            ValuePacket(np.ones((48, 80), np.uint8), {"spatial": frame([80, 48], "wrong")})
            for _ in range(3)
        ),
        batch.value.labels,
    )
    assert block.execute({"exposures": batch, "masks": masks}) is None
    assert "mask coordinate" in block.error


@pytest.mark.parametrize(
    "parameters",
    [
        {"clip_low": 250, "clip_high": 251},
        {"clip_low": True},
        {"relative_tolerance": 1.1},
        {"absolute_tolerance": float("nan")},
        {"minimum_contributors": 4},
        {"reference_index": 10},
        {"exposure_times": [0.1, 0.1, 0.1]},
        {"emit_support_masks": "yes"},
        {"maximum_pixels": 5},
        {"maximum_working_bytes": 1},
    ],
)
def test_bad_parameters_and_work_limits_are_contained_before_native_lut(parameters, monkeypatch):
    monkeypatch.setattr(cv2, "LUT", lambda *a, **k: (_ for _ in ()).throw(AssertionError("native")))
    batch, _, _ = fixture()
    block = tool(**parameters)
    assert block.execute({"exposures": batch}) is None
    assert "AssertionError" not in block.error and not any(block.current_emissions.values())


def test_empty_or_unobserved_domains_and_zero_camera_response_never_claim_measurements():
    batch, _, _ = fixture()
    block = tool(minimum_coverage=0)
    for extra in [
        {"mask": np.zeros((48, 80), np.uint8)},
        {
            "response": {
                "model": "opencv_robertson_response_v1",
                "encoding": "gray",
                "values": [[0]] * 256,
            }
        },
    ]:
        result = run(block, {"exposures": batch, **extra})
        assert not result.passed and result.outputs["report"]["measured_pixels"] == 0
        assert not result.outputs["image"].any() and not result.outputs["contributors"].any()


def test_strict_coverage_and_zero_fallback_are_valid_inspection_failures_with_graphics_off(
    monkeypatch,
):
    from opencv_dataflow import hdr_motion

    batch, _, changed = fixture(moving=True)
    block = tool(minimum_coverage=1, fallback="zero")
    block.set_graphics_mode("off")
    monkeypatch.setattr(
        hdr_motion,
        "graphic_object",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("graphics")),
    )
    result = run(block, {"exposures": batch})
    assert not result.passed and not result.drawings
    assert not result.outputs["fallback_mask"].any()
    assert not result.outputs["display"][changed].any()


def test_native_failure_emits_nothing_and_a_later_good_bracket_recovers(monkeypatch):
    batch, _, _ = fixture()
    block = tool()
    original = cv2.LUT
    monkeypatch.setattr(
        cv2, "LUT", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("LUT failed"))
    )
    assert block.execute({"exposures": batch}) is None
    assert "LUT failed" in block.error and not any(block.current_emissions.values())
    monkeypatch.setattr(cv2, "LUT", original)
    assert run(block, {"exposures": batch}).passed


@pytest.mark.parametrize("border", ["crop", "zero_pad"])
def test_native_mtb_alignment_feeds_matching_reference_grids_and_preserves_static_radiance(border):
    raw = cv2.resize(np.random.default_rng(14).integers(16, 112, (16, 24), np.uint8), (96, 80))
    codes = (raw // 4 * 4).astype(np.uint8)
    shifts = [(2, 1), (0, 0), (-2, -1)]
    frames = []
    for i, (seconds, (dx, dy)) in enumerate(zip([0.25, 0.5, 1.0], shifts, strict=True)):
        pixels = (codes.astype(float) * 2 * seconds).astype(np.uint8)
        shifted = cv2.warpAffine(
            pixels, np.float32([[1, 0, dx], [0, 1, dy]]), (96, 80), flags=cv2.INTER_NEAREST
        )
        frames.append(ValuePacket(shifted, {"spatial": frame([96, 80], f"capture-{i}")}))
    group = FlowGroup(tuple(frames), ("a", "b", "c"))
    align = run(
        builtin_registry().create(
            "cv.align_exposures",
            parameters={
                "reference_index": 1,
                "max_bits": 3,
                "exclude_range": 1,
                "border_mode": border,
            },
        ),
        {"exposures": group},
    )
    result = run(
        tool(absolute_tolerance=0, relative_tolerance=0),
        {"exposures": packet_for(align, "aligned")},
    )
    assert result.passed
    assert not result.outputs["motion_mask"].any()
    x, y, w, h = align.outputs["info"]["output_xywh"]
    valid = result.outputs["valid_mask"] != 0
    np.testing.assert_array_equal(
        result.outputs["image"][valid], codes[y : y + h, x : x + w][valid].astype(np.float32) / 64
    )
    assert result.payloads["image"]["spatial"]["root_id"] == "capture-1"


def test_xml_and_offscreen_inspector_offer_modes_without_running(app, tmp_path):
    from PySide6.QtWidgets import QComboBox

    from opencv_dataflow.app import MainWindow
    from opencv_dataflow.project_xml import export_project_xml, read_project_xml
    from opencv_dataflow.toolbox import block_entry

    graph = Graph()
    block = graph.add(tool(emit_support_masks=True))
    snapshot = copy.deepcopy(graph.to_dict())
    path = tmp_path / "hdr.xml"
    export_project_xml(snapshot, path)
    assert Graph.from_dict(read_project_xml(path), builtin_registry()).to_dict() == snapshot
    assert "photo" in block_entry(block).path[1]
    window = MainWindow()
    try:
        window.replace_graph(graph)
        window.inspector.select(block)
        for name in ("weighting", "fallback", "grid_policy", "clipping_evidence"):
            assert isinstance(window.inspector.fields[name], QComboBox)
        assert block.last_result is None
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
        graph.close()


@pytest.mark.parametrize(
    "support",
    [
        {"model": "bad"},
        {
            "model": "opencv_mtb_support_v1",
            "source_size": [80, 48],
            "shift": [True, 0],
            "output_xywh": [0, 0, 80, 48],
        },
        {
            "model": "opencv_mtb_support_v1",
            "source_size": [80, 48],
            "shift": [0, 0],
            "output_xywh": [0, 0, 80, 48],
            "spatial": frame([80, 48], "different"),
        },
    ],
)
def test_stale_or_malformed_alignment_support_is_rejected(support):
    batch, _, _ = fixture()
    items = list(batch.value.items)
    items[0] = ValuePacket(items[0].value, {**items[0].payload, "exposure_support": support})
    block = tool()
    assert block.execute({"exposures": FlowGroup(tuple(items), batch.value.labels)}) is None
    assert "support" in block.error and not any(block.current_emissions.values())
