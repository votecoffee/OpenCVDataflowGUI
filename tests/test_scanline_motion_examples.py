"""Original clip truth, near-edge geometry, current moving parts and saved app paths."""

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
from opencv_dataflow.remap_maps import COORDINATE_MAP_TAG

HERE = Path(__file__).resolve().parents[1] / "examples/opencv tools/rolling_shutter"
TRUTH = json.loads((HERE / "truth.json").read_bytes())


@pytest.mark.parametrize("profile", ("debug", "lean"))
@pytest.mark.parametrize("case", list(TRUTH["cases"]))
def test_saved_workflows_correct_real_current_pixels_with_known_row_geometry(case, profile):
    graph = Graph.load(HERE / (case + ".json"), builtin_registry())
    graph.set_run_profile(profile)
    expected = TRUTH["cases"][case]
    previous_ticket = None
    try:
        for index in range(expected["runs"]):
            graph.run()
            fit = graph.blocks["motion"]
            assert fit.state == expected["state"], fit.error
            result = fit.last_result
            assert result is not None
            report = result.outputs["report"]
            assert report["reason"] == expected["reason"]
            if expected["state"] != "pass":
                assert result.outputs["flow"] is NO_VALUE and not fit.current_emissions["flow"]
                assert graph.blocks["correct"].last_result is None
                continue
            corrected = graph.blocks["correct"].last_result
            assert corrected is not None and corrected.passed
            coverage = corrected.outputs["coverage"] != 0
            assert coverage.mean() > 0.9
            flow = result.outputs["flow"].vectors[:, 0]
            h = flow.shape[0]
            u = 2 * np.arange(h) / (h - 1) - 1
            coefficients = np.asarray(TRUTH["coefficients_xy"])
            amplitude = TRUTH["video_amplitudes"][index] if case == "video" else 1
            expected_flow = (
                np.column_stack(
                    [np.polynomial.polynomial.polyval(u, coefficients[:, axis]) for axis in (0, 1)]
                )
                * amplitude
            )
            supported = result.outputs["support"][:, 0] != 0
            errors = np.linalg.norm(flow[supported] - expected_flow[supported], axis=1)
            assert errors.max() < 0.4
            assert np.linalg.norm(flow[8] - expected_flow[8]) < 0.35
            assert np.linalg.norm(flow[-9] - expected_flow[-9]) < 0.35
            before = graph.blocks["before"].last_result.outputs["value"]
            after = graph.blocks["verify"].last_result.outputs["value"]
            assert after < before * 0.4 if amplitude else after == pytest.approx(0, abs=1e-10)
            current = graph.blocks["capture"].last_result
            assert corrected.payloads["image"]["ticket"] == current.payloads["image"]["ticket"]
            assert corrected.payloads["image"]["ticket"] != previous_ticket
            previous_ticket = corrected.payloads["image"]["ticket"]
            assert (
                corrected.payloads["image"]["spatial"]
                == graph.blocks["reference"].last_result.payloads["image"]["spatial"]
            )
            assert (
                corrected.payloads["map_x"][COORDINATE_MAP_TAG]["source_spatial"]
                == current.payloads["image"]["spatial"]
            )
            if case == "highdepth":
                assert corrected.outputs["image"].dtype == np.uint16
                assert corrected.outputs["image"].color_format == "bgr"
            if case == "video":
                assert corrected.payloads["image"]["video_frame"]["frame_index"] == index
                assert corrected.payloads["image"]["video_frame"]["timestamp_ms"] == pytest.approx(
                    index * TRUTH["video_period_ms"], abs=1e-6
                )
            if case == "moving_part":
                image = corrected.outputs["image"]
                assert np.all(image[78:143, 182:228] >= 240)
                ys, xs = np.nonzero(image > 235)
                assert abs(xs.min() - 180) <= 1 and abs(xs.max() - 229) <= 1
                assert abs(ys.min() - 76) <= 1 and abs(ys.max() - 144) <= 1
            if profile == "lean":
                assert not result.graphics and not result.drawings
            assert all(block.state == "pass" for block in graph.processing_blocks())
    finally:
        graph.close()


def test_examples_regenerate_exactly_and_video_decode_matches_known_original_frames(tmp_path):
    spec = importlib.util.spec_from_file_location("row_example", HERE / "build_example.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = tmp_path
    module.build()
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (HERE / path.relative_to(tmp_path)).read_bytes(), path.name
    for name, expected in TRUTH["asset_sha256"].items():
        assert hashlib.sha256((HERE / "assets" / name).read_bytes()).hexdigest() == expected
    _, frames = module.fixtures()
    capture = cv2.VideoCapture(str(HERE / "assets/row_wobble.avi"), cv2.CAP_FFMPEG)
    try:
        for expected in frames:
            ok, decoded = capture.read()
            assert ok
            np.testing.assert_array_equal(cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY), expected)
        assert not capture.read()[0]
    finally:
        capture.release()
