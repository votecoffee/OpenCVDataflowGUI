"""Measure mesh outputs against independent calibrated ray/triangle intersections."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import camera_record, pose_record
from opencv_dataflow.mesh_projection import ProjectMesh
from opencv_dataflow.meshes import TriangleMesh
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.tickets import TicketClock

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/mesh_projection"
W, H = 49, 41
K = np.array([[39, 0, 19.2], [0, 32, 23.7], [0, 0, 1]])
NEAR, FAR = 0.55, 2.7


def camera():
    return camera_record(K, [], [W, H], frame="inspection")


def _xyz(pixels, z):
    pixels, z = np.asarray(pixels), np.asarray(z)
    return np.column_stack(
        ((pixels[:, 0] - K[0, 2]) * z / K[0, 0], (pixels[:, 1] - K[1, 2]) * z / K[1, 1], z)
    )


def _oracle(mesh, faces, near=NEAR, far=FAR):
    yy, xx = np.mgrid[:H, :W]
    ray = np.stack(((xx - K[0, 2]) / K[0, 0], (yy - K[1, 2]) / K[1, 1], np.ones_like(xx)), axis=-1)
    depth = np.full((H, W), np.inf)
    rgb = np.zeros((H, W, 3), np.uint8)
    for indices in faces:
        a, b, c = mesh.vertices[indices].astype(np.float64)
        ab, ac = b - a, c - a
        normal = np.cross(ab, ac)
        z = (a @ normal) / (ray @ normal)
        relative = ray * z[:, :, None] - a
        alpha, beta = np.einsum(
            "ij,...j->i...", np.linalg.pinv(np.column_stack((ab, ac))), relative
        )
        # Avoid treating roundoff at an exact face edge as a geometric difference.
        hit = (alpha >= -1e-10) & (beta >= -1e-10) & (alpha + beta <= 1 + 1e-10)
        take = hit & (z >= near) & (z <= far) & (z < depth)
        weights = np.column_stack((1 - alpha[take] - beta[take], alpha[take], beta[take]))
        depth[take] = z[take]
        rgb[take] = np.rint(weights @ mesh.colors[indices]).astype(np.uint8)
    return depth, rgb


def _assert_outputs(result, truth_depth, truth_rgb):
    assert result and result.passed
    data = result.outputs
    hit = np.isfinite(truth_depth)
    np.testing.assert_array_equal(data["coverage_mask"], hit.astype(np.uint8) * 255)
    assert data["report"]["covered_pixels"] == np.count_nonzero(hit)
    if data["depth"] is not NO_VALUE:
        measured = data["depth"]
        assert measured.units == "m" and measured.dtype == np.float32
        assert np.isnan(measured[~hit]).all()
        np.testing.assert_allclose(measured[hit], truth_depth[hit], rtol=3e-6, atol=1e-9)
    if data["color"] is not NO_VALUE:
        assert data["color"].color_format == "rgb"
        np.testing.assert_allclose(data["color"], truth_rgb, atol=1, rtol=0)


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_mesh_compares_full_grid_to_independent_rays(profile):
    graph = Graph.load(EXAMPLE / "mesh_projection.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        mesh = graph.blocks["mesh"].last_result.outputs["mesh"]
        all_faces = graph.blocks["all"].last_result
        cull = graph.blocks["cull"].last_result
        _assert_outputs(all_faces, *_oracle(mesh, mesh.faces))
        _assert_outputs(cull, *_oracle(mesh, mesh.faces[:2]))
        assert all_faces.outputs["report"]["clipped_source_triangles"] == 2
        assert all_faces.outputs["report"]["raster_triangles"] == 6
        all_depth = all_faces.outputs["depth"]
        assert np.nanmin(all_depth) >= np.float32(NEAR)
        assert np.nanmax(all_depth) <= np.float32(FAR)
        assert np.any(all_depth == 2)
        assert np.any(all_depth < 1)
        camera_model = graph.blocks["camera"].last_result.outputs["camera"]
        direct = ProjectMesh(
            parameters={
                "minimum_z_m": NEAR,
                "maximum_z_m": FAR,
                "output_mode": "color_only",
            }
        )
        color_only = direct.execute({"mesh": mesh, "camera": camera_model})
        _assert_outputs(color_only, *_oracle(mesh, mesh.faces))
        assert color_only.outputs["depth"] is NO_VALUE
    finally:
        graph.close()


@pytest.mark.parametrize("mode", ["color_and_depth", "depth_only", "color_only"])
def test_metric_transform_asymmetric_camera_shading_and_ticket(mode):
    pixels = [(5.7, 8.2), (32.6, 8.37), (5.4, 30.64)]
    nominal = _xyz(pixels, [0.9, 1.6, 2.3])
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    offset = np.array([0.1, -0.25, 0.7])
    model = TriangleMesh(
        (nominal - offset) @ rotation * 1000,
        [[0, 1, 2]],
        "mm",
        "part",
        colors=np.eye(3) * 255,
    )
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, offset * 1000
    ticket = TicketClock().issue("mesh")
    mesh_payload = {"ticket": ticket, "spatial": frame([1, 1], "original")}
    block = ProjectMesh(parameters={"output_mode": mode, "minimum_z_m": NEAR, "maximum_z_m": FAR})
    result = block.execute(
        {
            "mesh": ValuePacket(model, mesh_payload),
            "camera": camera(),
            "transform": pose_record(pose, "part", "inspection", "mm"),
        }
    )
    reference = TriangleMesh(nominal, model.faces, colors=model.colors)
    assert result and result.passed, block.error
    _assert_outputs(result, *_oracle(reference, reference.faces))
    assert (result.outputs["color"] is NO_VALUE) == (mode == "depth_only")
    assert (result.outputs["depth"] is NO_VALUE) == (mode == "color_only")
    for name, payload in result.payloads.items():
        if isinstance(result.outputs.get(name), np.ndarray):
            assert payload["ticket"] == ticket
            assert payload["spatial"] == frame([W, H], f"{block.id}:target")
            assert payload["spatial"] != mesh_payload["spatial"]


def test_depth_preserves_far_bound_and_wide_numeric_range_with_black_vertices():
    low, high = 2e-6, 100_000.0
    xyz = np.vstack(
        (
            _xyz([(2.1, 2.3), (16.7, 2.6), (3.3, 16.2)], [low] * 3),
            _xyz([(25.1, 5.2), (43.7, 4.6), (24.2, 22.3)], [high] * 3),
        )
    )
    model = TriangleMesh(xyz, [[0, 1, 2], [3, 4, 5]], frame="inspection", colors=np.zeros((6, 3)))
    block = ProjectMesh(parameters={"minimum_z_m": 1e-6, "maximum_z_m": high})
    result = block.execute({"mesh": model, "camera": camera()})
    assert result and result.passed, block.error
    assert result.outputs["coverage_mask"][7, 7] == result.outputs["coverage_mask"][8, 29] == 255
    assert result.outputs["depth"][7, 7] == pytest.approx(low, abs=1e-12)
    assert result.outputs["depth"][8, 29] == pytest.approx(high, rel=1e-6)
    assert not np.any(result.outputs["color"])
    assert result.outputs["coverage_mask"][0, 0] == 0


def test_flat_shading_keeps_first_vertex_color_after_near_clipping():
    model = TriangleMesh(
        _xyz([(5.3, 5.4), (38.5, 5.4), (5.3, 31.7)], [0.3, 1.2, 1.2]),
        [[0, 1, 2]],
        frame="inspection",
        colors=np.eye(3) * 255,
    )
    block = ProjectMesh(parameters={"shading": "flat", "minimum_z_m": NEAR, "maximum_z_m": FAR})
    result = block.execute({"mesh": model, "camera": camera()})
    assert result and result.passed, block.error
    assert result.outputs["report"]["raster_triangles"] == 2
    pixels = result.outputs["color"][result.outputs["coverage_mask"] != 0]
    assert np.all(pixels == [255, 0, 0])


def test_triangle_and_pixel_work_budgets_withhold_partial_outputs():
    model = TriangleMesh(
        _xyz([(5.3, 5.4), (38.5, 5.4), (5.3, 31.7)], [0.3, 1.2, 1.2]),
        [[0, 1, 2]],
        frame="inspection",
    )
    for limits, code in (
        ({"max_triangles": 1}, "triangle_budget"),
        ({"max_pixel_tests": 1}, "pixel_budget"),
    ):
        block = ProjectMesh(parameters={"minimum_z_m": NEAR, "maximum_z_m": FAR, **limits})
        result = block.execute({"mesh": model, "camera": camera()})
        assert result is not None and not result.passed, block.error
        assert result.fail_reasons[0]["code"] == f"mesh_projection_{code}"
        assert all(result.outputs[k] is NO_VALUE for k in ("color", "depth", "coverage_mask"))


def test_clipped_and_distorted_inputs_have_distinct_results():
    model = TriangleMesh(
        _xyz([(5.3, 5.4), (38.5, 5.4), (5.3, 31.7)], [4] * 3),
        [[0, 1, 2]],
        frame="inspection",
    )
    block = ProjectMesh(parameters={"maximum_z_m": FAR})
    result = block.execute({"mesh": model, "camera": camera()})
    assert result is not None and not result.passed
    assert result.fail_reasons[0]["code"] == "mesh_projection_empty"
    assert np.isnan(result.outputs["depth"]).all() and not np.any(result.outputs["coverage_mask"])
    distorted = {**camera(), "distortion": [0.001, 0, 0, 0, 0]}
    assert block.execute({"mesh": model, "camera": distorted}) is None
    assert "rectified" in block.error
