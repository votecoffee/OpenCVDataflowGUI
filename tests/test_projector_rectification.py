"""Qualify native map direction, pattern landing and calibrated projector provenance."""

import json
from pathlib import Path
from runpy import run_path

import cv2
import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import camera_record
from opencv_dataflow.images import ImageArray, image_format
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.projector_rectification import _inverse_maps

EXAMPLE = (
    Path(__file__).resolve().parents[1] / "examples/opencv tools/projector_inverse_rectification"
)
TYPE = "3d.projector_inverse_rectify"


@pytest.fixture(scope="module")
def fixture():
    return run_path(str(EXAMPLE / "build_example.py"))


def _optics(fixture):
    raw = camera_record(
        fixture["RAW_K"], fixture["DISTORTION"], fixture["RAW_SIZE"], frame="projector"
    )
    ideal = camera_record(fixture["IDEAL_K"], [], fixture["IDEAL_SIZE"], frame="ideal")
    return raw, ideal


def _target_landing(projected, fixture):
    yy, xx = np.indices(fixture["IDEAL_SIZE"][::-1])
    target_pixels = np.stack((xx, yy), axis=-1)
    target_xyz = fixture["front_plane_points"](target_pixels)
    raw_yx = fixture["project_brown"](target_xyz)[..., ::-1]
    observed = map_coordinates(projected.astype(float), np.moveaxis(raw_yx, -1, 0), order=1)
    errors = []
    for cx, cy in fixture["MARKERS"]:
        win = np.s_[cy - 14 : cy + 15, cx - 14 : cx + 15]
        marker = observed[win]
        assert marker.max() > 120
        estimate = np.array([(marker * axis[win]).sum() / marker.sum() for axis in (xx, yy)])
        errors.append(float(np.linalg.norm(estimate - [cx, cy])))
    return errors


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_pattern_lands_on_independent_plane_and_recovers_ideal_view(profile, fixture):
    graph = Graph.load(EXAMPLE / "projector_inverse_rectification.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        prewarp = graph.blocks["prewarp"].last_result
        assert prewarp and prewarp.passed, graph.blocks["prewarp"].error
        report = prewarp.outputs["report"]
        assert report["direction"] == "raw_projector_pixel_to_ideal_pattern_sample"
        assert 0.94 < report["valid_fraction"] < 1
        assert report["roundtrip"]["maximum_px"] < 0.001
        actual = prewarp.outputs["image"]
        assert actual.shape == fixture["RAW_SIZE"][::-1]
        mask = prewarp.outputs["valid_mask"]
        assert np.count_nonzero(mask) == report["valid_pixels"]
        assert not actual[mask == 0].any()
        assert max(_target_landing(actual, fixture)) < 0.09
        assert ("Projector support" in prewarp.drawings) == (profile == "debug")
        comparison = graph.blocks["compare"].last_result
        assert comparison and comparison.passed, graph.blocks["compare"].error
        assert comparison.outputs["similarity"] > 0.997
    finally:
        graph.close()


def test_banded_native_mapping_matches_full_and_independent_forward_model(fixture, monkeypatch):
    raw, ideal = _optics(fixture)
    k, d, size = np.array(raw["camera_matrix"]), np.array(raw["distortion"]), fixture["RAW_SIZE"]
    vector = np.array(fixture["ROTATION"])
    rot, target = fixture["rotation_matrix"](vector), fixture["IDEAL_K"]
    monkeypatch.setattr("opencv_dataflow.projector_rectification._BAND_PIXELS", 2900)
    mx, my, finite, front = _inverse_maps(k, d, rot, target, size, lambda: False)
    full = cv2.initInverseRectificationMap(k, d, rot, target, size, cv2.CV_32FC1)
    np.testing.assert_array_equal(mx, full[0])
    np.testing.assert_array_equal(my, full[1])
    assert finite == front == size[0] * size[1]

    pixels = np.array([[0, 2], [319, 238], [22, 81], [172, 96], [218, 151]])
    expected = np.stack([a[pixels[:, 1], pixels[:, 0]] for a in (mx, my)], axis=-1)
    target_xyz = fixture["front_plane_points"](expected)
    raw_pixel = fixture["project_brown"](target_xyz)
    np.testing.assert_allclose(raw_pixel, pixels, atol=0.0005)


def test_reports_unsupported_forward_rays_and_can_gate_coverage_or_roundtrip(fixture):
    raw, ideal = _optics(fixture)
    registry = builtin_registry()
    data = {"pattern": fixture["make_pattern"](), "projector": raw, "pattern_camera": ideal}
    for settings, reason in (
        ({"rotation_vector": [0, np.pi, 0]}, "projector_support"),
        (
            {"rotation_vector": list(fixture["ROTATION"]), "minimum_valid_fraction": 1},
            "projector_support",
        ),
        (
            {"rotation_vector": list(fixture["ROTATION"]), "maximum_roundtrip_error_px": 1e-7},
            "projector_roundtrip",
        ),
    ):
        block = registry.create(TYPE, parameters=settings)
        result = block.execute(data)
        assert result and not result.passed, block.error
        assert result.fail_reasons[0]["code"] == reason
        assert all(result.outputs[name] is NO_VALUE for name in ("image", "map_x", "map_y"))
        if settings["rotation_vector"][1] == np.pi:
            assert result.outputs["report"]["forward_pixels"] == 0
            assert not result.outputs["valid_mask"].any()


def test_split_and_packed_maps_apply_projector_frame_and_check_source_grid(fixture):
    raw, ideal = _optics(fixture)
    pattern = fixture["make_pattern"]()
    size = fixture["IDEAL_SIZE"]
    # Independent tagged color channels also check encoding and per-channel fill.
    pattern = ImageArray(
        np.stack((pattern, pattern // 2, pattern // 3), axis=2) * np.uint16(17), "rgb"
    )
    payload = {
        "spatial": frame(size, "test-source"),
        "optical_frame": "ideal",
        "stereo_calibration_grid": {"from": "upstream"},
        "applied_calibration": {"from": "upstream"},
        "recipe_marker": "test-recipe",
    }
    packet = ValuePacket(pattern, payload)
    registry = builtin_registry()
    parameters = {
        "rotation_vector": list(fixture["ROTATION"]),
        "border_value": 13,
        "interpolation": "nearest",
    }
    tool = registry.create(TYPE, parameters=parameters)
    result = tool.execute({"pattern": packet, "projector": raw, "pattern_camera": ideal})
    assert result and result.passed, tool.error
    out = result.outputs["image"]
    assert image_format(out) == "rgb" and out.dtype == np.uint16
    assert (out[result.outputs["valid_mask"] == 0] == 13).all()
    context = result.payloads["image"]
    assert context["recipe_marker"] == "test-recipe"
    assert context["optical_frame"] == "projector"
    assert context["spatial"]["root_id"] != payload["spatial"]["root_id"]
    assert not {"stereo_calibration_grid", "applied_calibration"} & context.keys()
    maps = {
        key: ValuePacket(result.outputs[key], result.payloads[key]) for key in ("map_x", "map_y")
    }

    remap_settings = {"interpolation": "nearest", "border_value": 13}
    split = registry.create("cv.remap", parameters=remap_settings)
    split_out = split.execute({"image": packet, **maps})
    assert split_out and split_out.passed, split.error
    np.testing.assert_array_equal(split_out.outputs["image"], out)
    assert split_out.payloads["image"] == context

    pack = registry.create("cv.pack_remap_map", parameters={"sampling": "nearest"})
    packed = pack.execute(maps)
    assert packed and packed.passed, pack.error
    packed_map = ValuePacket(packed.outputs["packed_map"], packed.payloads["packed_map"])
    replay = registry.create("cv.remap_packed", parameters=remap_settings)
    packed_out = replay.execute({"image": packet, "packed_map": packed_map})
    assert packed_out and packed_out.passed, replay.error
    np.testing.assert_array_equal(packed_out.outputs["image"], out)
    assert packed_out.payloads["image"] == context

    other = ValuePacket(pattern, {**payload, "spatial": frame(size, "other-source")})
    assert split.execute({"image": other, **maps}) is None
    assert "Source coordinate frame differs" in split.error
    split.parameters["border"] = "wrap"
    assert split.execute({"image": packet, **maps}) is None
    assert "interpolation and constant border" in split.error
    assert (
        tool.execute(
            {"pattern": packet, "projector": raw, "pattern_camera": {**ideal, "frame": "unrelated"}}
        )
        is None
    )
    assert "optical frame differs" in tool.error


def test_identity_and_budget_and_readable_rig_input():
    size = [36, 23]
    raw = camera_record([[45, 0, 17], [0, 42, 11], [0, 0, 1]], [], size, frame="projector")
    rig = {"model": "camera_projector_rig_v1", "projector": raw}
    pattern = np.arange(size[0] * size[1], dtype=np.uint16).reshape(size[::-1])
    block = builtin_registry().create(TYPE)
    result = block.execute({"pattern": pattern, "projector": rig})
    assert result and result.passed, block.error
    np.testing.assert_array_equal(result.outputs["image"], pattern)
    assert result.outputs["report"]["valid_fraction"] == 1
    block.parameters["maximum_pixels"] = pattern.size - 1
    assert block.execute({"pattern": pattern, "projector": rig}) is None
    assert "configured pixel limit" in block.error


def test_original_fixture_truth_regenerates(fixture):
    rows = json.loads((EXAMPLE / "assets/target_truth.json").read_text(encoding="utf-8"))
    xyz = fixture["front_plane_points"](fixture["MARKERS"])
    np.testing.assert_allclose([row["target_xyz_m"] for row in rows], xyz, atol=1e-12)
    np.testing.assert_allclose(
        [row["projector_pixel"] for row in rows], fixture["project_brown"](xyz), atol=1e-10
    )
