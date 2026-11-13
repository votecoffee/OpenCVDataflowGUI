"""Saved inspection graphs and independent smoothing comparison geometry."""

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.runner import main

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/stack_blur"


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize(
    "case",
    [
        "stack_vs_gaussian",
        "selected_smoothing",
        "uint16_smoothing",
        "unsafe_native",
        "kernel_clamp",
    ],
)
def test_saved_stack_examples(case, profile, tmp_path):
    registry = builtin_registry()
    graph = Graph.load(EXAMPLE / (case + ".json"), registry)
    try:
        graph.set_run_profile(profile)
        graph.run()
        block = graph.blocks["stack"]
        if case == "unsafe_native":
            assert block.state == "error" and block.error_kind == "invalid"
            assert not block.current_emissions
        else:
            assert all(b.state == "pass" for b in graph.processing_blocks()), [
                (b.id, b.error) for b in graph.processing_blocks()
            ]
            result = block.last_result
            if case == "kernel_clamp":
                assert result.outputs["info"]["effective_kernel"] == [127, 95]
            if case == "uint16_smoothing":
                assert result.outputs["image"].dtype == np.uint16
            if case == "selected_smoothing":
                source = graph.blocks["source"].last_result.outputs["image"]
                y, x = np.indices(source.shape)
                outside = (x - 512) ** 2 + (y - 384) ** 2 > 220**2
                np.testing.assert_array_equal(result.outputs["image"][outside], source[outside])
        graph.save(tmp_path / "saved.json")
        restored = Graph.load(tmp_path / "saved.json", registry)
        restored.run()
        assert restored.blocks["stack"].state == block.state
        restored.close()
    finally:
        graph.close()


@pytest.mark.local_examples
def test_normal_installed_runner_route(tmp_path):
    assert (
        main([str(EXAMPLE / "stack_vs_gaussian.json"), "--report", str(tmp_path / "report.json")])
        == 0
    )


@pytest.mark.local_examples
def test_matched_variance_and_measured_edge_noise_truth():
    spec = importlib.util.spec_from_file_location(
        "stack_comparison", EXAMPLE / "compare_filters.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    clean, noisy = module.fixture()
    assert np.array_equal(clean[:, :512], np.full((768, 512), 60, np.uint8))
    assert np.array_equal(clean[:, 512:], np.full((768, 512), 180, np.uint8))
    saved = cv2.imdecode(
        np.frombuffer((EXAMPLE / "assets/noisy.png").read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED
    )
    np.testing.assert_array_equal(saved, noisy)
    for kernel in (31, 127, 255):
        sigma = module.matched_sigma(kernel)
        radius = kernel // 2
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        gaussian = np.exp(-x * x / (2 * sigma * sigma))
        gaussian /= gaussian.sum()
        assert np.sum(gaussian * x * x) == pytest.approx(radius * (radius + 2) / 6, rel=1e-10)
        result = module.compare(kernel, clean, noisy)
        for name in ("stack", "gaussian"):
            measured = result[name]
            assert measured["flat_rms_dn"] < result["input"]["flat_rms_dn"] / 10
            assert measured["center_error_px"] < 0.3
            assert measured["edge_10_90_width_px"] > result["input"]["edge_10_90_width_px"]
    truth = json.loads((EXAMPLE / "truth.json").read_bytes())
    assert truth["edge_center_px"] == 511.5
    assert truth["reference"]["stack"]["edge_10_90_width_px"] == pytest.approx(70.71, abs=0.1)


@pytest.mark.local_examples
def test_recorded_benchmark_conditions_and_samples_are_explicit():
    report = json.loads((EXAMPLE / "benchmark_reference.json").read_bytes())
    assert (
        report["width"],
        report["height"],
        report["working_depth"],
        report["opencv_threads"],
    ) == (1920, 1080, "float32", 1)
    for row in report["results"]:
        for name in ("stack", "gaussian"):
            times = row["timings_ms"][name]
            assert len(times["samples"]) == report["repeats"] == 7
            assert times["median"] == pytest.approx(float(np.median(times["samples"])))
            assert 0 < times["p25"] <= times["median"] <= times["p75"]
