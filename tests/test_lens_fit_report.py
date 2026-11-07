"""Exercise native fit diagnostics and per-view calibration judgments."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.calibration import find_grid, grid_settings
from opencv_dataflow.core import Graph

_PROJECT = Path(__file__).parents[1] / "examples/custom tools/lens_fit_report/lens_fit_report.json"


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_saved_calibration_report_matches_the_rendered_views(profile):
    graph = Graph.load(_PROJECT, builtin_registry())
    graph.set_run_profile(profile)
    graph.run()
    for block in graph.blocks.values():
        assert block.last_result and block.last_result.passed, (block.name, block.error)
    fit = graph.blocks["fit"]
    result = fit.last_result.outputs
    report = result["fit_report"]
    views = report["views"]
    assert report["pose_direction"] == "board_to_camera"
    assert report["translation_units"] == "mm"
    assert [view["view_index"] for view in views] == list(range(6))
    assert [view["image_name"] for view in views] == [f"view-{i:02}.png" for i in range(1, 7)]
    assert result["rms"] == pytest.approx(np.sqrt(np.mean([v["rms_pixels"] ** 2 for v in views])))
    assert report["worst_view_index"] == 5
    assert result["worst_view_rms"] == pytest.approx(views[5]["rms_pixels"])
    assert 0.4 < result["worst_view_rms"] < 0.6
    deviations = report["intrinsic_stddev"]
    assert len(deviations) == 18 and deviations["fx"] > 0 and deviations["k4"] == 0
    assert all(
        len(view["rotation_stddev_radians"]) == len(view["translation_stddev"]) == 3
        for view in views
    )

    view = views[5]
    image = cv2.imdecode(
        np.frombuffer((_PROJECT.parent / "assets" / view["image_name"]).read_bytes(), np.uint8),
        cv2.IMREAD_GRAYSCALE,
    )
    size, world = grid_settings(fit.parameters)
    observed, _ = find_grid(image, size, fit.parameters)
    calibration = result["calibration"]
    predicted, _ = cv2.projectPoints(
        world,
        np.asarray(view["rotation_vector_radians"]),
        np.asarray(view["translation_vector"]),
        np.asarray(calibration["camera_matrix"]),
        np.asarray(calibration["distortion"]),
    )
    error = np.linalg.norm(predicted.reshape(-1, 2) - observed, axis=1)
    assert view["rms_pixels"] == pytest.approx(float(np.sqrt(np.mean(error**2))), abs=1e-4)


@pytest.mark.local_examples
@pytest.mark.parametrize(
    ("overall_limit", "view_limit", "code"),
    [
        (0.4, 0.3, "calibration_view_rms_exceeds_limit"),
        (0.1, 0.0, "calibration_rms_exceeds_limit"),
    ],
)
def test_fit_thresholds_emit_separate_reasons_and_keep_the_report(overall_limit, view_limit, code):
    fit = Graph.load(_PROJECT, builtin_registry()).blocks["fit"]
    fit.parameters.update(max_rms_pixels=overall_limit, max_view_rms_pixels=view_limit)
    result = fit.execute({})
    assert result is not None and not result.passed, fit.error
    assert [reason["code"] for reason in result.fail_reasons] == [code]
    assert result.outputs["fit_report"]["worst_view_index"] == 5
    assert result.outputs["calibration"]["views"] == 6


@pytest.mark.local_examples
def test_unavailable_standard_deviations_are_serializable(monkeypatch):
    fit = Graph.load(_PROJECT, builtin_registry()).blocks["fit"]
    native = cv2.calibrateCameraExtended

    def missing_deviations(*args, **kwargs):
        result = native(*args, **kwargs)
        result[5].flat[0] = np.nan
        result[6].flat[3] = np.inf
        return result

    monkeypatch.setattr(cv2, "calibrateCameraExtended", missing_deviations)
    result = fit.execute({})
    assert result is not None and result.passed, fit.error
    report = result.outputs["fit_report"]
    assert report["intrinsic_stddev"]["fx"] is None
    assert report["views"][0]["translation_stddev"][0] is None
