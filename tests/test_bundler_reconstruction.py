"""Validate Bundler import, camera axes, scale and independent held-out projection."""

import json
from pathlib import Path
from runpy import run_path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.bundler_reconstruction import ImportBundlerReconstruction
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/bundler_reconstruction"


@pytest.fixture(scope="module")
def fixture():
    return run_path(str(EXAMPLE / "build_example.py"))


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_bundle_projects_into_held_out_camera_with_metric_depth_and_color(fixture, profile):
    truth = json.loads((EXAMPLE / "assets/truth.json").read_text(encoding="utf-8"))
    graph = Graph.load(EXAMPLE / "bundler_reconstruction.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.set_branch_workers(2)
        graph.run()
        block = graph.blocks["bundle"]
        result = block.last_result
        assert result and result.passed, block.error
        out = result.outputs
        assert [item["registered"] for item in out["cameras"]] == [True, True, True, False, True]
        assert out["cloud"].units == "m" and out["cloud"].frame == "fixture_world"
        np.testing.assert_array_equal(out["cloud"].attributes["source_index"], np.arange(42))
        np.testing.assert_allclose(out["cloud"].points, truth["point_xyz_m"], atol=1e-7)
        np.testing.assert_array_equal(out["cloud"].colors, truth["color_rgb"])
        assert out["report"]["scale"]["meters_per_bundle_unit"] == pytest.approx(0.4)
        assert out["report"]["observations"] == out["report"]["scored_observations"] == 117
        assert out["report"]["maximum_error_px"] < 1e-8
        assert out["observations"]["point_indices"] == []
        for i, record in enumerate(out["cameras"]):
            if not record["registered"]:
                continue
            c2w = np.asarray(record["camera_to_world"]["source_to_target"])
            w2c = np.asarray(record["world_to_camera"]["source_to_target"])
            np.testing.assert_allclose(c2w[:3, 3], truth["camera_centers_m"][i], atol=1e-12)
            np.testing.assert_allclose(
                c2w[:3, :3], fixture["rotation"](fixture["ANGLES"][i]).T, atol=1e-12
            )
            np.testing.assert_allclose(c2w @ w2c, np.eye(4), atol=1e-12)
            assert record["camera"]["distortion"][:2] == list(fixture["RADIAL"][i])
        pose = out["world_to_camera"]
        k = np.asarray(out["camera"]["camera_matrix"])
        assert tuple(k[:2, 2]) == (320.5, 240.5)
        proj = cv2.projectPoints(
            out["cloud"].points.astype(np.float64),
            np.asarray(pose["rvec"]),
            np.asarray(pose["translation"]),
            k,
            np.asarray(out["camera"]["distortion"]),
        )[0].reshape(-1, 2)
        np.testing.assert_allclose(proj, truth["held_out_pixels"], atol=5e-5, rtol=0)
        raster = graph.blocks["held_out"].last_result
        assert raster and raster.passed
        u, v = np.floor(np.asarray(truth["held_out_pixels"]) + 0.5).astype(int).T
        np.testing.assert_allclose(raster.outputs["depth"][v, u], truth["held_out_z_m"], atol=2e-7)
        np.testing.assert_array_equal(raster.outputs["color"][v, u], truth["color_rgb"])
        assert ("point cloud" in result.graphics) == (profile == "debug")
    finally:
        graph.close()


@pytest.mark.local_examples
def test_default_scale_preserves_raw_geometry_and_sizes_enable_selected_intrinsics(fixture):
    tool = ImportBundlerReconstruction(parameters={"path": str(EXAMPLE / "assets/bundle.out")})
    out = tool.execute({})
    assert out and out.passed, tool.error
    assert out.outputs["camera"] is NO_VALUE
    assert out.outputs["camera_to_world"]["units"] == "unitless"
    assert out.outputs["observations"]["image_points"] is None
    cloud = out.outputs["cloud"]
    assert cloud.units == "unitless"
    xyz, _ = fixture["scene"]()
    np.testing.assert_allclose(cloud.points, xyz * fixture["RAW_UNITS_PER_METER"], atol=3e-7)
    assert out.outputs["report"]["scale"]["meters_per_bundle_unit"] is None

    sized = tool.execute({"image_sizes": (640, 480)})
    assert sized and sized.passed, tool.error
    assert sized.outputs["camera"]["image_size"] == [640, 480]
    rows = sized.outputs["observations"]
    assert rows["point_indices"] == list(range(42))
    assert rows["feature_indices"] == [10 + index * 3 for index in range(42)]
    uv, _ = fixture["project"](0, xyz)
    np.testing.assert_allclose(rows["image_points"], uv, atol=1e-10)
    assert sized.payloads["camera"]["spatial"] == sized.payloads["observations"]["spatial"]
    tool.parameters["camera_index"] = 3
    unregistered = tool.execute({})
    assert unregistered and unregistered.passed
    assert unregistered.outputs["camera_to_world"] is NO_VALUE
    assert len(unregistered.outputs["cloud"].points) == 42


def _small_bundle():
    return (
        "# Bundle file v0.3\n"
        "2 1\n"
        "400 0 0\n"
        "1 0 0\n0 1 0\n0 0 1\n0 0 0\n"
        "400 0 0\n"
        "1 0 0\n0 1 0\n0 0 1\n0.1 0 0\n"
        "0 0 -2\n12 36 66\n"
        "2 0 4 0 0 1 8 20 0\n"
    )


@pytest.mark.parametrize(
    ("old", "new", "error"),
    [
        ("# Bundle file v0.3", "NVM_V3", "Unsupported reconstruction format"),
        ("# Bundle file v0.3", "# Bundle file v0.4", "Unsupported reconstruction format"),
        ("2 1", "2049 1", "Camera count"),
        ("2 1", "2 100001", "Point count"),
        ("12 36 66", "12 36 256", "RGB sample"),
        ("2 0 4 0 0 1 8 20 0", "2 0 4 0 0 0 8 20 0", "Duplicate Bundler observation"),
        ("2 0 4 0 0 1 8 20 0", "2 0 4 0 0 2 8 20 0", "camera index"),
        ("0 0 -2", "0 NaN -2", "Invalid numeric token"),
        ("1 0 0\n0 1 0\n0 0 1", "-1 0 0\n0 1 0\n0 0 1", "proper rotation"),
        ("1 8 20 0\n", "1 8 20\n", "file ended"),
        ("1 8 20 0\n", "1 8 20 0\n7\n", "trailing data"),
    ],
)
def test_malformed_imports_fail_explicitly(tmp_path, old, new, error):
    path = tmp_path / "bundle.out"
    path.write_text(_small_bundle().replace(old, new, 1), encoding="utf-8")
    tool = ImportBundlerReconstruction(parameters={"path": str(path)})
    assert tool.execute({}) is None
    assert error in tool.error


def test_metric_baseline_and_error_judgement_on_small_bundle(tmp_path):
    path = tmp_path / "fixture.out"
    # Bundler permits white space between any of the numeric fields.
    path.write_text(_small_bundle().replace("1 8 20 0", "1 8\n20 0"), encoding="utf-8-sig")
    tool = ImportBundlerReconstruction(
        parameters={
            "path": str(path),
            "scale_mode": "known_baseline",
            "baseline_length_m": 0.2,
            "maximum_reprojection_error_px": 0.1,
        }
    )
    outcome = tool.execute({})
    assert outcome and outcome.passed, tool.error
    assert outcome.outputs["report"]["rms_px"] < 1e-10
    assert outcome.outputs["report"]["scale"]["meters_per_bundle_unit"] == 2
    np.testing.assert_array_equal(outcome.outputs["cloud"].points, [[0, 0, -4]])

    path.write_text(_small_bundle().replace("1 8 20 0", "1 8 27 0"), encoding="utf-8")
    bad = tool.execute({})
    assert bad and not bad.passed
    assert bad.fail_reasons[0]["code"] == "bundler_reprojection_limit"
    assert bad.outputs["report"]["maximum_error_px"] == pytest.approx(7)

    path.write_text(_small_bundle().replace("0.1 0 0", "0 0 0"), encoding="utf-8")
    assert tool.execute({}) is None
    assert "coincident centers" in tool.error
