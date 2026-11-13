"""Independent donor references, sampling direction, motion support and grid ownership."""

import json
import struct
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.block_xml import save_block_xml
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import NO_VALUE, frame
from opencv_dataflow.project_xml import read_block_xml
from opencv_dataflow.sequence_tools import DenseFlow
from opencv_dataflow.tickets import TicketClock, ticket_key

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/dense_flow_warp"


def vectors(shape, dx=0, dy=0):
    return np.full((*shape, 2), [dx, dy], np.float32)


def run(image, flow, parameters=None, **extra):
    block = builtin_registry().create("cv.warp_flow", parameters=parameters or {})
    result = block.execute({"image": image, "flow": flow, **extra})
    assert result is not None, block.error
    return block, result


def partition(result):
    names = ("coverage", "invalid", "uncovered", "occluded")
    masks = [result.outputs[n] for n in names if result.outputs[n] is not NO_VALUE]
    total = sum(m.astype(int) // 255 for m in masks)
    np.testing.assert_array_equal(total, 1)


@pytest.mark.parametrize("direction", ["source_to_target", "target_to_source"])
@pytest.mark.parametrize(
    "dtype,encoding",
    [
        (np.uint8, "gray"),
        (np.uint16, "rgb"),
        (np.int16, "gray"),
        (np.float32, "rgba"),
        (np.float64, "bgr"),
    ],
)
def test_integer_motion_matches_array_slicing_and_target_grid_keeps_source_ticket(
    direction, dtype, encoding
):
    h, w, dx, dy = 24, 32, 3, 2
    values = np.arange(h * w, dtype=float).reshape(h, w)
    if encoding != "gray":
        values = np.stack([values + c * 100 for c in range(4 if encoding == "rgba" else 3)], axis=2)
    values = values.astype(dtype)
    clock = TicketClock()
    source_context = {
        "spatial": frame([w, h], "source"),
        "ticket": clock.issue("camera", "fixed", "source"),
        "acquisition": {"encoder": 39},
        "custom": {"lot": "A"},
    }
    target_context = {
        "spatial": frame([w, h], "target"),
        "ticket": clock.issue("camera", "fixed", "target"),
    }
    source = ImageArray(values, encoding, source_context)
    target = ImageArray(np.zeros((h, w), np.uint8), "gray", target_context)
    sign = 1 if direction == "source_to_target" else -1
    flow = DenseFlow(
        vectors((h, w), sign * dx, sign * dy), source_context if sign == 1 else target_context
    )
    original = source.copy()
    original_flow = flow.vectors.copy()
    _, result = run(
        source, flow, {"direction": direction, "interpolation": "nearest"}, target=target
    )
    expected = np.zeros_like(source)
    expected[dy:, dx:] = source[:-dy, :-dx]
    np.testing.assert_array_equal(result.outputs["image"], expected)
    assert image_format(result.outputs["image"]) == encoding
    assert result.outputs["image"].dtype == np.dtype(dtype)
    valid = np.zeros((h, w), np.uint8)
    valid[dy:, dx:] = 255
    np.testing.assert_array_equal(result.outputs["coverage"], valid)
    assert result.outputs["report"]["valid_pixels"] == (h - dy) * (w - dx)
    assert result.outputs["report"]["flow_grid_association"] == "known_frame"
    assert result.outputs["occluded"] is NO_VALUE
    context = result.payloads["image"]
    assert context["spatial"] == target_context["spatial"]
    assert context["custom"] == source_context["custom"]
    assert context["acquisition"] == source_context["acquisition"]
    assert ticket_key(context["ticket"]) == ticket_key(source_context["ticket"])
    assert ticket_key(context["ticket"]) != ticket_key(target_context["ticket"])
    np.testing.assert_array_equal(source, original)
    np.testing.assert_array_equal(flow.vectors, original_flow)
    partition(result)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fractional_pull_matches_independent_linear_ramp(dtype):
    y, x = np.indices((20, 29))
    source = (7 * x - 3 * y + 5).astype(dtype)
    _, result = run(source, vectors(source.shape, 0.5, 0.25))
    expected = 7 * (x + 0.5) - 3 * (y + 0.25) + 5
    selected = result.outputs["coverage"] != 0
    np.testing.assert_allclose(result.outputs["image"][selected], expected[selected], atol=1e-5)
    assert not selected[-1].any() and not selected[:, -1].any()


def test_nonuniform_forward_field_inverts_known_analytic_scale_and_reports_residual():
    y, x = np.indices((21, 33))
    flow = vectors(x.shape)
    flow[..., 0] = x * 0.2
    _, result = run(x.astype(np.float32), flow, {"direction": "source_to_target"})
    valid = result.outputs["coverage"] != 0
    np.testing.assert_allclose(result.outputs["map_x"][valid], (x / 1.2)[valid], atol=0.04)
    np.testing.assert_allclose(result.outputs["image"][valid], (x / 1.2)[valid], atol=0.04)
    assert np.nanmax(result.outputs["inverse_error"][valid]) <= 0.1


@pytest.mark.parametrize("method,halo", [("cubic", 2), ("lanczos4", 4)])
def test_high_order_sampling_has_conservative_support_halo(method, halo):
    source = np.arange(24 * 32, dtype=np.uint16).reshape(24, 32)
    _, result = run(source, vectors(source.shape), {"interpolation": method})
    expected = np.zeros(source.shape, np.uint8)
    expected[halo:-halo, halo:-halo] = 255
    np.testing.assert_array_equal(result.outputs["coverage"], expected)
    np.testing.assert_array_equal(result.outputs["image"][expected != 0], source[expected != 0])


@pytest.mark.parametrize("border", ["replicate", "wrap", "reflect"])
def test_extrapolated_border_display_never_counts_as_coverage(border):
    from opencv_dataflow.registration import BORDERS

    source = np.arange(20 * 29, dtype=np.uint16).reshape(20, 29)
    _, result = run(
        source,
        vectors(source.shape, 3, 0),
        {"interpolation": "nearest", "border": border, "invalid_fill": "native_border"},
    )
    mx, my = result.outputs["map_x"], result.outputs["map_y"]
    np.testing.assert_array_equal(
        result.outputs["image"],
        cv2.remap(source, mx, my, cv2.INTER_NEAREST, borderMode=BORDERS[border]),
    )
    assert not result.outputs["coverage"][:, -3:].any()
    assert result.outputs["report"]["uncovered_pixels"] == 60


def test_source_selection_moves_with_donors_and_keeps_holes():
    source = np.arange(24 * 32, dtype=np.uint16).reshape(24, 32)
    mask = np.zeros(source.shape, np.uint8)
    mask[6:14, 6:16] = 255
    mask[9, 10] = 0
    region = {
        "model": "image_region_v1",
        "image_size": [32, 24],
        "shape": "rectangle",
        "rectangle": [8, 8, 6, 4],
    }
    _, result = run(
        source,
        vectors(source.shape, -3, -2),
        {"interpolation": "nearest"},
        mask=mask,
        region=region,
    )
    selected = np.zeros(source.shape, np.uint8)
    selected[8:12, 8:14] = 255
    selected[9, 10] = 0
    expected = np.zeros_like(selected)
    expected[2:, 3:] = selected[:-2, :-3]
    np.testing.assert_array_equal(result.outputs["coverage"], expected)
    assert not result.outputs["image"][expected == 0].any()
    partition(result)


def test_invalid_primary_and_reverse_fields_are_distinct_from_occlusion_proxy():
    source = np.arange(20 * 29, dtype=np.uint16).reshape(20, 29)
    flow = vectors(source.shape)
    flow[2, 3] = np.nan
    flow[3, 3] = 1e12
    reverse = vectors(source.shape)
    reverse[4, 3] = np.inf
    reverse[5:7, 8:11, 0] = 3
    _, result = run(source, flow, {"consistency": "required"}, reverse_flow=reverse)
    assert result.outputs["report"]["invalid_pixels"] == 3
    assert result.outputs["report"]["occlusion_proxy_pixels"] == 6
    assert result.outputs["invalid"][4, 3] == 255 and result.outputs["occluded"][4, 3] == 0
    assert np.isnan(result.outputs["roundtrip_error"][4, 3])
    assert (result.outputs["occluded"][5:7, 8:11] == 255).all()
    assert not result.outputs["image"][5:7, 8:11].any()
    partition(result)
    json.dumps(result.outputs["report"], allow_nan=False)


def test_forward_nonconvergence_withholds_image_when_coverage_gate_fails():
    y, x = np.indices((20, 29))
    flow = vectors(x.shape)
    flow[..., 0] = 2 * x
    block, result = run(
        x.astype(np.uint8),
        flow,
        {"direction": "source_to_target", "minimum_coverage": 0.9, "inverse_iterations": 4},
    )
    assert block.state == "fail" and result.outputs["image"] is NO_VALUE
    assert result.outputs["invalid"].any() and not block.current_emissions.get("image")
    partition(result)


def test_flow_support_and_required_reverse_are_enforced():
    source = np.arange(20 * 29, dtype=np.uint16).reshape(20, 29)
    support = np.ones(source.shape, np.uint8)
    support[4:8, 5:9] = 0
    _, result = run(source, vectors(source.shape), flow_support=support)
    assert (result.outputs["invalid"][4:8, 5:9] == 255).all()
    block = builtin_registry().create("cv.warp_flow", parameters={"consistency": "required"})
    assert block.execute({"image": source, "flow": vectors(source.shape)}) is None
    assert "required" in block.error and not block.current_emissions


def test_exported_donor_maps_reuse_generic_remap_and_keep_target_frame():
    source = ImageArray(
        np.arange(20 * 29, dtype=np.uint16).reshape(20, 29),
        "gray",
        {"spatial": frame([29, 20], "source")},
    )
    _, result = run(source, vectors(source.shape, -3, -2), {"interpolation": "nearest"})
    warp = builtin_registry().create("cv.remap", parameters={"interpolation": "nearest"})
    replay = warp.execute(
        {"image": source, "map_x": result.outputs["map_x"], "map_y": result.outputs["map_y"]}
    )
    assert replay is not None, warp.error
    assert replay.payloads["image"]["spatial"] == result.payloads["image"]["spatial"]
    valid = result.outputs["coverage"] != 0
    np.testing.assert_array_equal(replay.outputs["image"][valid], result.outputs["image"][valid])


def test_existing_farneback_flow_can_pull_current_image_onto_previous_grid():
    registry = builtin_registry()
    source = np.random.default_rng(19).integers(0, 256, (96, 128), np.uint8)
    source = cv2.GaussianBlur(source, (3, 3), 0.6)
    current = np.zeros_like(source)
    current[2:, 3:] = source[:-2, :-3]
    context = {"spatial": frame([128, 96], "camera")}
    a, b = ImageArray(source, "gray", context), ImageArray(current, "gray", context)
    estimate = registry.create("sequence.farneback", parameters={"max_gap_ms": 0})
    assert not estimate.execute({"image": a}).outputs["ready"]
    motion = estimate.execute({"image": b})
    assert motion is not None, estimate.error
    _, result = run(b, motion.outputs["flow"], target=a)
    central = np.s_[20:-20, 20:-20]
    before = np.abs(current.astype(float) - source)[central].mean()
    after = np.abs(result.outputs["image"].astype(float) - source)[central].mean()
    assert after < before * 0.2 and after < 5
    assert (result.outputs["coverage"][central] == 255).all()


def test_optional_coverage_preview_is_skipped_with_graphics_off_unless_exposed():
    source = np.arange(20 * 29, dtype=np.uint16).reshape(20, 29)
    block = builtin_registry().create("cv.warp_flow")
    block.set_graphics_mode("off")
    result = block.execute({"image": source, "flow": vectors(source.shape)})
    assert result is not None, block.error
    assert result.outputs["coverage_preview"] is NO_VALUE and not result.drawings
    block.exposed_outputs.append("coverage_preview")
    result = block.execute({"image": source, "flow": vectors(source.shape)})
    assert result is not None, block.error
    assert result.outputs["coverage_preview"].shape == (20, 29, 3) and not result.drawings


def test_known_pull_grid_retains_target_pyramid_level_instead_of_source_level():
    def context(name, level):
        scale = 2**level
        matrix = np.diag([scale, scale, 1]).tolist()
        spatial = frame([16, 12], name)
        spatial.update(root_size=[16 * scale, 12 * scale], to_root=matrix)
        return {
            "spatial": spatial,
            "pyramid": {
                "level": level,
                "largest_size": [16 * scale, 12 * scale],
                "to_largest": matrix,
            },
        }

    source_context, target_context = context("source", 1), context("target", 2)
    source = ImageArray(np.ones((12, 16), np.uint16), "gray", source_context)
    _, result = run(source, DenseFlow(vectors(source.shape), target_context))
    assert result.payloads["image"]["spatial"] == target_context["spatial"]
    assert result.payloads["image"]["pyramid"] == target_context["pyramid"]


@pytest.mark.parametrize(
    "case", ["shape", "depth", "frame", "metadata_size", "budget", "f64_cubic", "border"]
)
def test_bad_input_controls_and_known_native_failure_are_contained(case):
    source = ImageArray(
        np.zeros((20, 29), np.uint8), "gray", {"spatial": frame([29, 20], "source")}
    )
    flow = vectors(source.shape)
    params = {"direction": "source_to_target"}
    if case == "shape":
        flow = flow[:3]
    elif case == "depth":
        flow = flow.astype(np.uint8)
    elif case == "frame":
        flow = DenseFlow(flow, {"spatial": frame([29, 20], "other")})
    elif case == "metadata_size":
        flow = DenseFlow(flow, {"spatial": frame([28, 20])})
        params["direction"] = "target_to_source"
    elif case == "budget":
        params["maximum_work"] = 1
    elif case == "f64_cubic":
        source = source.astype(np.float64)
        params["interpolation"] = "cubic"
    else:
        params["border_value"] = -1
    block = builtin_registry().create("cv.warp_flow", parameters=params)
    assert block.execute({"image": source, "flow": flow}) is None
    assert block.error_kind == "invalid" and not block.current_emissions


def test_flow_loader_has_strict_bounded_format_and_unicode_paths(tmp_path):
    path = tmp_path / "位移.flo"
    values = vectors((9, 13), 2, -3)
    values[1, 2] = np.nan
    path.write_bytes(struct.pack("<fii", 202021.25, 13, 9) + values.astype("<f4").tobytes())
    block = builtin_registry().create("input.flow_file", parameters={"path": str(path)})
    result = block.execute({})
    assert result is not None, block.error
    np.testing.assert_array_equal(result.outputs["flow"].vectors, values)
    assert result.outputs["info"]["nonfinite_vectors"] == 1
    json.dumps(result.outputs["flow"].describe(), allow_nan=False)
    path.write_bytes(struct.pack("<fii", 202021.25, 50000, 50000))
    assert block.execute({}) is None and "dimensions" in block.error
    path.write_bytes(struct.pack("<fii", 202021.25, 13, 9) + b"\0" * 4)
    assert block.execute({}) is None and "bytes" in block.error


def test_original_flo_fields_match_native_opencv_reader():
    block = builtin_registry().create(
        "input.flow_file", parameters={"path": str(EXAMPLE / "forward.flo")}
    )
    result = block.execute({})
    assert result is not None, block.error
    native = cv2.readOpticalFlow(str(EXAMPLE / "forward.flo"))
    np.testing.assert_array_equal(result.outputs["flow"].vectors, native)


@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize(
    "name", ["dense_flow_warp", "invalid_and_occlusion_proxy", "coverage_gate_failure"]
)
def test_runnable_projects_truth_graphics_and_xml(profile, name, tmp_path):
    graph = Graph.load(EXAMPLE / (name + ".json"), builtin_registry())
    truth = json.loads((EXAMPLE / "truth.json").read_text())
    try:
        graph.set_run_profile(profile)
        graph.run()
        block = graph.blocks["warp"]
        result = block.last_result
        assert result is not None, block.error
        assert bool(result.drawings) == (profile == "debug")
        assert result.outputs["report"]["valid_pixels"] == truth["translation_coverage"] - (
            236 if name == "invalid_and_occlusion_proxy" else 0
        )
        if name == "coverage_gate_failure":
            assert block.state == "fail" and result.outputs["image"] is NO_VALUE
        else:
            assert block.state == "pass", block.error
            expected = cv2.imread(str(EXAMPLE / "target.png"), cv2.IMREAD_UNCHANGED)[..., ::-1]
            valid = result.outputs["coverage"] != 0
            np.testing.assert_array_equal(result.outputs["image"][valid], expected[valid])
        if name == "invalid_and_occlusion_proxy":
            assert result.outputs["report"]["invalid_pixels"] == 20
            assert result.outputs["report"]["occlusion_proxy_pixels"] == 216
        partition(result)
        path = tmp_path / "preset.cv.warp_flow.xml"
        save_block_xml(block, path)
        assert read_block_xml(path) == block.to_project_item()
    finally:
        graph.close()
