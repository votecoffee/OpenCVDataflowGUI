"""Independent motion/error references, native controls, support semantics and safe copies."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import opencv_dataflow.flow_refinement as refinement
from opencv_dataflow.block_xml import save_block_xml
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.flow_refinement import SOLVER_CONTROLS, RefineDenseFlow, solver
from opencv_dataflow.images import ImageArray, format_image
from opencv_dataflow.payloads import NO_VALUE, frame
from opencv_dataflow.project_xml import read_block_xml
from opencv_dataflow.sequence_tools import DenseFlow
from opencv_dataflow.tickets import TicketClock, ticket_key

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/flow_refinement"
TUNED = {"alpha": 1.0, "delta": 10.0, "fixed_point_iterations": 10, "sor_iterations": 20}


def pair():
    a = cv2.GaussianBlur(
        np.random.default_rng(17).integers(0, 256, (96, 144), dtype=np.uint8), (7, 7), 1.1
    )
    b = cv2.warpAffine(
        a, np.float32([[1, 0, 2], [0, 1, 1]]), (144, 96), borderMode=cv2.BORDER_REFLECT_101
    )
    seed = np.full((*a.shape, 2), [1.25, 0.5], np.float32)
    return a, b, seed


def run(a, b, seed, parameters=None, **extra):
    block = RefineDenseFlow(parameters=parameters or {})
    result = block.execute({"image": a, "target": b, "flow": seed, **extra})
    assert result is not None, block.error
    return block, result


def test_all_native_controls_match_getters_and_calc_without_mutating_seed():
    a, b, seed = pair()
    controls = {
        "fixed_point_iterations": 7,
        "sor_iterations": 9,
        "alpha": 2.0,
        "delta": 3.0,
        "gamma": 4.0,
        "omega": 1.2,
        "epsilon": 0.02,
    }
    native = solver(controls)
    for name, (setter, _) in SOLVER_CONTROLS.items():
        assert getattr(native, setter.replace("set", "get", 1))() == pytest.approx(controls[name])
    expected = native.calc(a, b, seed.copy())
    block, result = run(a, b, seed, controls)
    assert block.state == "pass"
    np.testing.assert_allclose(result.outputs["flow"].vectors, expected, atol=1e-6)
    np.testing.assert_array_equal(seed, np.full_like(seed, [1.25, 0.5]))


def test_refinement_improves_independent_translation_and_comparable_residual():
    a, b, seed = pair()
    _, result = run(a, b, seed, TUNED)
    flow, report = result.outputs["flow"].vectors, result.outputs["report"]
    before = np.linalg.norm(seed[16:-16, 16:-16] - [2, 1], axis=2).mean()
    after = np.linalg.norm(flow[16:-16, 16:-16] - [2, 1], axis=2).mean()
    assert after < 0.6 and after < before * 0.65
    assert report["refined_mean_residual"] < report["initial_mean_residual"] * 0.6
    assert not result.outputs["support"][:4].any() and not result.outputs["support"][-4:].any()
    assert np.isnan(result.outputs["residual"][result.outputs["support"] == 0]).all()


@pytest.mark.parametrize("encoding", ["gray", "bgr", "rgb", "bgra", "rgba", "hsv"])
def test_color_inputs_use_canonical_gray_and_keep_source_ticket_and_grid(encoding):
    a, b, seed = pair()
    clock = TicketClock()
    context_a = {
        "ticket": clock.issue("camera", "fixed", "A"),
        "spatial": frame([144, 96], "A"),
        "pyramid": {"level": 0, "largest_size": [144, 96], "to_largest": np.eye(3).tolist()},
        "alpha_representation": "straight",
        "acquisition": {"encoder": 107},
    }
    context_b = {"ticket": clock.issue("camera", "fixed", "B"), "spatial": frame([144, 96], "B")}
    a, b = format_image(a, encoding), format_image(b, encoding)
    a.metadata, b.metadata = context_a, context_b
    original_a, original_b, original_seed = a.copy(), b.copy(), seed.copy()
    block, result = run(a, b, DenseFlow(seed, context_a), {"alignment": "pixel_grid", **TUNED})
    assert block.state == "pass"
    payload = result.payloads["flow"]
    assert ticket_key(payload["ticket"]) == ticket_key(context_a["ticket"])
    assert ticket_key(payload["ticket"]) != ticket_key(context_b["ticket"])
    assert payload["spatial"] == context_a["spatial"] and payload["pyramid"] == context_a["pyramid"]
    assert payload["acquisition"] == {"encoder": 107} and "alpha_representation" not in payload
    assert payload["flow_refinement"]["target"]["ticket"] == context_b["ticket"]
    np.testing.assert_array_equal(a, original_a)
    np.testing.assert_array_equal(b, original_b)
    np.testing.assert_array_equal(seed, original_seed)


def test_selection_retains_initial_vectors_outside_and_intersects_support_holes():
    a, _, seed = pair()
    seed[:] = 0
    mask = np.zeros(a.shape, np.uint8)
    mask[20:70, 20:100] = 255
    region = {
        "model": "image_region_v1",
        "image_size": [144, 96],
        "shape": "rectangle",
        "rectangle": [30, 25, 50, 35],
    }
    support = np.full(a.shape, 255, np.uint8)
    support[35:40, 50:55] = 0
    _, result = run(
        a, a, seed, {"border_margin": 2}, mask=mask, region=region, target_support=support
    )
    expected = np.zeros(a.shape, np.uint8)
    expected[25:60, 30:80] = 255
    expected[33:42, 48:57] = 0  # 2-pixel support erosion around the independent hole
    np.testing.assert_array_equal(result.outputs["support"], expected)
    np.testing.assert_array_equal(result.outputs["flow"].vectors, 0)


def test_source_selection_limits_writes_but_native_solver_keeps_full_context():
    a, b, seed = pair()
    mask = np.zeros(a.shape, np.uint8)
    mask[25:60, 30:80] = 255
    _, full = run(a, b, seed, TUNED)
    _, selected = run(a, b, seed, TUNED, mask=mask)
    np.testing.assert_array_equal(selected.outputs["flow"].vectors[mask == 0], seed[mask == 0])
    np.testing.assert_allclose(
        selected.outputs["flow"].vectors[mask != 0],
        full.outputs["flow"].vectors[mask != 0],
        atol=1e-6,
    )


def test_support_holes_follow_forward_donors_not_source_positions(monkeypatch):
    y, x = np.indices((24, 32))
    a = (x + 3 * y).astype(np.uint8)
    b = cv2.warpAffine(a, np.float32([[1, 0, 2], [0, 1, 1]]), (32, 24))
    seed = np.full((*a.shape, 2), [2, 1], np.float32)

    class Fixed:
        def calc(self, a, b, incoming):
            return incoming

    monkeypatch.setattr(refinement, "solver", lambda p: Fixed())
    source_support = np.full(a.shape, 255, np.uint8)
    target_support = source_support.copy()
    source_support[3:5, 3:5] = 0
    target_support[10:14, 14:18] = 0
    _, result = run(
        a,
        b,
        seed,
        {"border_margin": 0},
        source_support=source_support,
        target_support=target_support,
    )
    expected = np.full(a.shape, 255, np.uint8)
    expected[-1], expected[:, -2:] = 0, 0
    expected[3:5, 3:5], expected[9:13, 12:16] = 0, 0
    np.testing.assert_array_equal(result.outputs["support"], expected)
    np.testing.assert_array_equal(result.outputs["residual"][expected != 0], 0)


def test_residual_increase_rejects_bad_native_proposal_and_native_exceptions_are_contained(
    monkeypatch,
):
    a, b, seed = pair()
    seed[:] = [2, 1]

    class BadProposal:
        def calc(self, a, b, incoming):
            return np.zeros_like(incoming)

    monkeypatch.setattr(refinement, "solver", lambda p: BadProposal())
    block, result = run(a, b, seed, {"reject_residual_increase": True})
    assert block.state == "fail" and result.outputs["report"]["reason"] == "residual_increased"

    def raises(p):
        raise RuntimeError("synthetic solver failure")

    monkeypatch.setattr(refinement, "solver", raises)
    assert block.execute({"image": a, "target": b, "flow": seed}) is None
    assert block.error_kind == "exception" and not block.current_emissions


@pytest.mark.parametrize(
    "parameters,reason",
    [
        ({"minimum_valid_fraction": 1.0}, "insufficient_support"),
        ({"maximum_mean_residual": 1.0}, "high_residual"),
        ({"border_margin": 64}, "no_valid_support"),
    ],
)
def test_quality_failure_withholds_field_but_retains_diagnostics(parameters, reason):
    a, b, seed = pair()
    block, result = run(a, b, seed, {**TUNED, **parameters})
    assert block.state == "fail" and result.outputs["flow"] is NO_VALUE
    assert not block.current_emissions.get("flow") and block.current_emissions["report"]
    assert result.outputs["report"]["reason"] == reason
    assert result.fail_reasons[0]["code"] == "flow_refinement_" + reason
    json.dumps(result.outputs["report"], allow_nan=False)


def test_empty_selection_skips_native_and_fails_without_nan_report(monkeypatch):
    a, b, seed = pair()
    monkeypatch.setattr(
        cv2, "VariationalRefinement_create", lambda: pytest.fail("empty selection invoked native")
    )
    block, result = run(a, b, seed, mask=np.zeros(a.shape, np.uint8))
    assert block.state == "fail" and result.outputs["report"]["supported_mean_residual"] is None
    json.dumps(result.outputs["report"], allow_nan=False)


@pytest.mark.parametrize("value", [np.nan, 1e30])
def test_unusable_native_fields_are_contained_before_reporting_or_emission(value, monkeypatch):
    a, b, seed = pair()

    class Unusable:
        def calc(self, a, b, incoming):
            incoming[10, 10, 0] = value
            return incoming

    monkeypatch.setattr(refinement, "solver", lambda p: Unusable())
    block = RefineDenseFlow()
    assert block.execute({"image": a, "target": b, "flow": seed}) is None
    assert block.error_kind == "invalid" and not block.current_emissions


def test_flow64_is_copied_to_native32_and_graphics_off_avoids_optional_previews():
    a, b, seed = pair()
    block = RefineDenseFlow()
    block.set_graphics_mode("off")
    result = block.execute({"image": a, "target": b, "flow": seed.astype(np.float64)})
    assert result is not None, block.error
    assert result.outputs["flow"].vectors.dtype == np.float32
    assert not result.drawings and result.outputs["visualization"] is NO_VALUE
    block.exposed_outputs.append("visualization")
    result = block.execute({"image": a, "target": b, "flow": seed})
    assert result.outputs["visualization"].shape == (*a.shape, 3) and not result.drawings


def test_coercible_real_parameters_use_validated_values_in_quality_checks():
    a, b, seed = pair()
    block, result = run(
        a,
        b,
        seed,
        {
            "maximum_motion": "256",
            "minimum_valid_fraction": "0.1",
            "maximum_mean_residual": "20",
            "visual_max_motion": "10",
        },
    )
    assert block.state == "pass" and result.outputs["report"]["valid_fraction"] > 0.1


@pytest.mark.parametrize(
    "case",
    [
        "depth",
        "shape",
        "flow_type",
        "nan",
        "motion",
        "frame",
        "flow_grid",
        "support",
        "work",
        "weights",
        "weights_string",
        "omega",
        "epsilon",
        "boolean",
    ],
)
def test_invalid_inputs_and_controls_are_contained(case):
    a, b, seed = pair()
    parameters, extra = {}, {}
    if case == "depth":
        a = a.astype(np.uint16)
    elif case == "shape":
        b = b[:20]
    elif case == "flow_type":
        seed = seed.astype(np.int16)
    elif case == "nan":
        seed[20, 30] = np.nan
    elif case == "motion":
        seed[20, 30] = 257
    elif case == "frame":
        a = ImageArray(a, "gray", {"spatial": frame([144, 96], "A")})
        b = ImageArray(b, "gray", {"spatial": frame([144, 96], "B")})
    elif case == "flow_grid":
        a = ImageArray(a, "gray", {"spatial": frame([144, 96], "A")})
        seed = DenseFlow(seed, {"spatial": frame([144, 96], "B")})
    elif case == "support":
        extra["target_support"] = np.zeros((20, 30), np.uint8)
    elif case == "work":
        parameters["maximum_work"] = 1
    elif case == "weights":
        parameters = {"delta": 0, "gamma": 0}
    elif case == "weights_string":
        parameters = {"delta": "0", "gamma": "0"}
    elif case == "omega":
        parameters["omega"] = 2.0
    elif case == "epsilon":
        parameters["epsilon"] = 0.0
    else:
        parameters["reject_residual_increase"] = "yes"
    block = RefineDenseFlow(parameters=parameters)
    assert block.execute({"image": a, "target": b, "flow": seed, **extra}) is None
    assert block.error_kind == "invalid" and not block.current_emissions


@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize("name", ["flow_refinement", "partial_support", "residual_gate_failure"])
def test_runnable_projects_independent_truth_xml_and_warp_interoperability(profile, name, tmp_path):
    graph = Graph.load(EXAMPLE / (name + ".json"), builtin_registry())
    truth = json.loads((EXAMPLE / "truth.json").read_text())
    try:
        graph.set_run_profile(profile)
        graph.run()
        block = graph.blocks["refine"]
        result = block.last_result
        assert result is not None, block.error
        assert bool(result.drawings) == (profile == "debug")
        if name == "residual_gate_failure":
            assert block.state == "fail" and result.outputs["flow"] is NO_VALUE
            assert result.outputs["report"]["reason"] == "high_residual"
        else:
            assert block.state == "pass"
            field = result.outputs["flow"].vectors
            if name == "flow_refinement":
                margin = truth["evaluation_border"]
                values = field[margin:-margin, margin:-margin]
                error = np.linalg.norm(values - truth["translation"], axis=2).mean()
                assert error < truth["maximum_mean_endpoint_error"]
                assert (
                    error
                    < np.linalg.norm(
                        np.subtract(truth["initial_translation"], truth["translation"])
                    )
                    * truth["maximum_error_ratio"]
                )
            else:
                np.testing.assert_array_equal(field[0, 0], truth["initial_translation"])
                np.testing.assert_array_equal(
                    graph.blocks["trusted"].last_result.outputs["mask"],
                    cv2.imread(str(EXAMPLE / "target_support.png"), cv2.IMREAD_GRAYSCALE),
                )
            warp = graph.blocks["warp"]
            assert warp.state == "pass", warp.error
            assert (
                warp.last_result.payloads["image"]["spatial"]
                == graph.blocks["target"].last_result.payloads["image"]["spatial"]
            )
        path = tmp_path / "refine.cv.refine_flow.xml"
        save_block_xml(block, path)
        assert read_block_xml(path) == block.to_project_item()
    finally:
        graph.close()
