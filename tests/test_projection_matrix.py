"""Projection factorization, unit conventions and independent reprojection evidence."""

from pathlib import Path
from runpy import run_path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/projection_matrix"
TYPE = "3d.inspect_projection_matrix"


@pytest.fixture(scope="module")
def fixture():
    return run_path(str(EXAMPLE / "build_example.py"))


def _assert_geometry(result, fixture):
    k, r, t, _, _, _ = fixture["fixture_geometry"]()
    report = result.outputs["report"]
    np.testing.assert_allclose(report["camera_matrix"], k, atol=2e-9, rtol=0)
    np.testing.assert_allclose(report["camera_center_world"], fixture["CENTER"], atol=2e-12)
    world = result.outputs["world_to_camera"]
    local = result.outputs["camera_to_world"]
    assert (world["source_frame"], world["target_frame"]) == (fixture["WORLD"], fixture["OPTICAL"])
    assert (local["source_frame"], local["target_frame"]) == (fixture["OPTICAL"], fixture["WORLD"])
    transform = np.array(world["source_to_target"])
    np.testing.assert_allclose(transform[:3, :3], r, atol=2e-12)
    np.testing.assert_allclose(transform[:3, 3], t, atol=2e-12)
    np.testing.assert_allclose(
        transform @ np.array(local["source_to_target"]), np.eye(4), atol=2e-12
    )
    expected_fov = np.degrees(
        np.arctan2(k[:2, 2], k.diagonal()[:2])
        + np.arctan2(fixture["SIZE"] - k[:2, 2], k.diagonal()[:2])
    )
    np.testing.assert_allclose(report["fov_deg"], expected_fov, atol=1e-10)
    assert report["relative_recomposition_error"] < 1e-12


def _settings(fixture, **overrides):
    return {
        "width": fixture["SIZE"][0],
        "height": fixture["SIZE"][1],
        "world_frame": fixture["WORLD"],
        "camera_frame": fixture["OPTICAL"],
        "units": "m",
        **overrides,
    }


@pytest.mark.parametrize("scale", [1e-220, -3.7e-60, 2.0, -1e200])
def test_native_factorization_is_invariant_under_matrix_sign_and_scale(fixture, scale):
    k, _, _, obj, pixels, projection = fixture["fixture_geometry"]()
    block = builtin_registry().create(TYPE, parameters=_settings(fixture))
    result = block.execute(
        {
            "projection": (projection * scale).tolist(),
            "object_points": obj.tolist(),
            "image_points": pixels.tolist(),
        }
    )
    assert result and result.passed, block.error
    _assert_geometry(result, fixture)
    report = result.outputs["report"]
    assert report["world_units"] == "m"
    assert report["focal_lengths_mm"] is None and report["principal_point_mm"] is None
    np.testing.assert_allclose(report["principal_point_px"], k[:2, 2], atol=1e-10)
    assert report["reprojection"]["rms_px"] < 1e-9
    assert result.outputs["camera"]["distortion"] == []


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_example_verifies_camera_model_and_sensor_values(fixture, profile):
    graph = Graph.load(EXAMPLE / "projection_matrix.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        tool = graph.blocks["inspect"]
        result = tool.last_result
        assert result and result.passed, tool.error
        _assert_geometry(result, fixture)
        spec = result.outputs["report"]
        k, _, _, _, points, _ = fixture["fixture_geometry"]()
        np.testing.assert_allclose(spec["pixel_pitch_mm"], [0.016, 0.016])
        np.testing.assert_allclose(spec["focal_lengths_mm"], k.diagonal()[:2] * 0.016)
        np.testing.assert_allclose(spec["principal_point_mm"], k[:2, 2] * 0.016)
        assert spec["reprojection"]["valid_depth_count"] == 8
        assert ("Projection checks" in result.drawings) == (profile == "debug")
        assert result.payloads["image"]["spatial"] == tool.input_payloads["image"]["spatial"]
        assert result.outputs["camera"]["units"] == "m"
        replay = graph.blocks["reproject"].last_result
        assert replay and replay.passed
        np.testing.assert_allclose(replay.outputs["points"], points, atol=1e-8, rtol=0)
    finally:
        graph.close()


def test_nonzero_skew_keeps_exact_intrinsics_and_pixel_checks(fixture):
    k, _, _, obj, uv, projection = fixture["fixture_geometry"](skew=28.75)
    block = builtin_registry().create(TYPE, parameters=_settings(fixture, units="mm"))
    millimeter_projection = projection.copy()
    millimeter_projection[:, 3] *= 1000
    packet = ValuePacket((obj * 1000).tolist(), {"object_point_units": "mm"})
    result = block.execute(
        {
            "projection": millimeter_projection.tolist(),
            "object_points": packet,
            "image_points": uv.tolist(),
        }
    )
    assert result and result.passed, block.error
    assert result.outputs["camera"] is NO_VALUE
    report = result.outputs["report"]
    assert not report["camera_model_compatible"]
    np.testing.assert_allclose(report["camera_matrix"], k, atol=1e-9)
    np.testing.assert_allclose(report["camera_center_world"], fixture["CENTER"] * 1000, atol=1e-9)
    assert report["reprojection"]["rms_px"] < 1e-8
    assert result.outputs["world_to_camera"]["units"] == "mm"


@pytest.mark.parametrize("left", [np.zeros((3, 3)), np.diag([1, 0.3, 0]), np.diag([1, 1, 2e-12])])
def test_singular_and_ill_conditioned_left_blocks_fail_without_geometry(left):
    projection = np.column_stack((left, np.ones(3)))
    block = builtin_registry().create(TYPE)
    result = block.execute({"projection": projection.tolist()})
    assert result and not result.passed, block.error
    assert result.fail_reasons[0]["code"] == "projection_singular"
    assert all(
        result.outputs[name] is NO_VALUE
        for name in ("camera", "camera_to_world", "world_to_camera")
    )


def test_mismatch_and_camera_side_are_distinct_inspection_failures(fixture):
    _, rotation, _, obj, pixels, projection = fixture["fixture_geometry"]()
    block = builtin_registry().create(TYPE, parameters=_settings(fixture))
    inputs = {"projection": projection.tolist(), "object_points": obj.tolist()}
    result = block.execute({**inputs, "image_points": (pixels + [2, -3]).tolist()})
    assert result and not result.passed
    assert result.fail_reasons[0]["code"] == "projection_check_rms"
    assert result.outputs["report"]["reprojection"]["rms_px"] == pytest.approx(np.sqrt(13))
    assert result.outputs["camera"] is not NO_VALUE

    obj[3] = fixture["CENTER"] + np.array([0, 0, -2]) @ rotation
    invalid = block.execute(
        {**inputs, "object_points": obj.tolist(), "image_points": pixels.tolist()}
    )
    assert invalid and not invalid.passed
    assert invalid.fail_reasons[0]["code"] == "projection_check_invalid_depth"
    report = invalid.outputs["report"]["reprojection"]
    assert report["reprojected_points_px"][3] is None and report["errors_px"][3] is None
    assert report["valid_depth_count"] == 7


def test_rejects_incomplete_sensor_or_observation_settings_and_wrong_grids(fixture):
    _, _, _, obj, pixels, projection = fixture["fixture_geometry"]()
    block = builtin_registry().create(TYPE, parameters=_settings(fixture, sensor_width_mm=9))
    args = {"projection": projection.tolist(), "object_points": obj.tolist()}
    assert block.execute(args) is None
    assert "both active sensor dimensions" in block.error
    block.parameters["sensor_height_mm"] = 7
    assert block.execute(args) is None
    assert "both ordered object and image points" in block.error
    xy = ValuePacket(pixels.tolist(), {"spatial": frame(fixture["SIZE"], "different")})
    image = ValuePacket(
        np.zeros(fixture["SIZE"][::-1], np.uint8), {"spatial": frame(fixture["SIZE"], "target")}
    )
    assert block.execute({**args, "image_points": xy, "image": image}) is None
    assert "attached image frame" in block.error

    block.parameters["projection"] = [[0, 1]]
    assert block.execute({}) is None
    assert "finite 3×4 projection matrix" in block.error
