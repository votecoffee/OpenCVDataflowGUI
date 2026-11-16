"""Saved known-transfer projects, source tickets, spectral maps and exact assets."""

import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE

HERE = Path(__file__).resolve().parents[1] / "examples/opencv tools/frequency_response"
TRUTH = json.loads((HERE / "truth.json").read_bytes())


@pytest.mark.parametrize("profile", ["debug", "lean"])
@pytest.mark.parametrize("case", list(TRUTH["cases"]))
def test_saved_known_transfers_and_expected_failure_states(case, profile):
    expected = TRUTH["cases"][case]
    graph = Graph.load(HERE / (case + ".json"), builtin_registry())
    graph.set_run_profile(profile)
    try:
        graph.run()
        block = graph.blocks["measure"]
        assert block.state == expected["expected_state"], block.error
        result = block.last_result
        if expected["expected_state"] == "error":
            assert result is None and block.error_kind == "invalid"
            assert not any(block.current_emissions.values())
            return
        assert result is not None
        metrics = result.outputs["metrics"]
        assert metrics["model"] == "image_frequency_response_v1"
        assert [c["ticket"]["value"] for c in metrics["source_contexts"]] == [
            expected["sample"],
            expected["reference"],
        ]
        assert result.payloads["gain"]["ticket"]["value"] == expected["sample"]
        assert result.payloads["gain"]["spatial"]["root_id"] not in [
            c["spatial"]["root_id"] for c in metrics["source_contexts"]
        ]
        source = cv2.imread(str(HERE / "assets" / expected["sample"]), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(result.outputs["image"], source)
        assert result.outputs["image"].dtype == source.dtype
        if case == "no_reference_energy":
            assert (
                result.outputs["mean_gain"] is result.outputs["maximum_phase_degrees"] is NO_VALUE
            )
            assert not result.outputs["supported"].any()
            assert result.fail_reasons[0]["code"] == "frequency_reference_support_insufficient"
            return
        fy, fx = np.fft.fftshift(np.fft.fftfreq(96)), np.fft.fftshift(np.fft.fftfreq(128))
        np.testing.assert_array_equal(metrics["frequency_x"], fx)
        np.testing.assert_array_equal(metrics["frequency_y"], fy)
        transfer = np.ones((96, 128), complex)
        if case == "binomial_blur":
            transfer = ((1 + np.cos(2 * np.pi * fy)) / 2)[:, None] * (
                (1 + np.cos(2 * np.pi * fx)) / 2
            )[None, :]
        elif case == "shift_phase_fail":
            transfer = np.exp(-2j * np.pi * (2 * fy[:, None] - 3 * fx[None, :]))
        elif case in ("gain_half_fail", "color_red"):
            transfer *= 0.5
        valid, phase_valid = result.outputs["supported"] != 0, result.outputs["phase_valid"] != 0
        np.testing.assert_allclose(
            result.outputs["gain"][valid], np.abs(transfer[valid]), atol=1e-10, rtol=0
        )
        error = np.angle(np.exp(1j * (result.outputs["phase"] - np.angle(transfer))))
        assert np.max(np.abs(error[phase_valid])) < 1e-9
        if case == "horizontal_grating":
            assert valid.sum() == 2
        if case == "masked_identity":
            aperture = result.outputs["aperture"]
            mask = graph.blocks["selection"].last_result.outputs["mask"]
            assert not aperture[mask == 0].any()
        assert metrics["supported_bins"] == int(valid.sum())
        assert metrics["mean_gain"] == pytest.approx(
            float(np.abs(transfer[valid]).mean()), abs=1e-10
        )
        if case == "gain_half_fail":
            assert result.fail_reasons[0]["code"] == "frequency_gain_out_of_tolerance"
        elif case == "shift_phase_fail":
            assert result.fail_reasons[0]["code"] == "frequency_phase_out_of_tolerance"
        if profile == "lean":
            assert not result.graphics and not result.drawings
        else:
            assert result.graphics["radial response"].shape == (320, 640, 3)
            assert result.drawings["response status"].source == "image"
    finally:
        graph.close()


def test_regeneration_reproduces_every_original_png_and_saved_project(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "frequency_response_example", HERE / "build_example.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    builder.HERE = tmp_path
    builder.build()
    for source in HERE.glob("*.json"):
        assert (tmp_path / source.name).read_bytes() == source.read_bytes(), source
    for name, sha in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((HERE / "assets" / name).read_bytes()).hexdigest() == sha
        assert hashlib.sha256((tmp_path / "assets" / name).read_bytes()).hexdigest() == sha
