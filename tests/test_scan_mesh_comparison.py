"""Independent surface checks for the scan-to-mesh inspection block."""

from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.mesh_comparison import CompareScanToMesh
from opencv_dataflow.mesh_distance import TriangleSurface
from opencv_dataflow.meshes import TriangleMesh
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame
from opencv_dataflow.pointclouds import PointCloud
from opencv_dataflow.tickets import TicketClock

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/custom tools/scan_mesh_comparison"


def _codes(result):
    return {reason["code"] for reason in result.fail_reasons}


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_panel_has_exact_rectangle_distances_and_distinct_judgments(profile):
    graph = Graph.load(EXAMPLE / "scan_mesh_comparison.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.run()
        scan = graph.blocks["scan"].last_result.outputs["cloud"]
        xyz = scan.points.astype(float)
        closest = np.column_stack((np.clip(xyz[:, :2], 0, 0.1), np.full(len(xyz), 0.5)))
        expected = np.linalg.norm(xyz - closest, axis=1)
        assert graph.blocks["pass"].last_result.passed
        size = graph.blocks["size"].last_result
        coverage = graph.blocks["coverage"].last_result
        assert _codes(size) == {"scan_mesh_tolerance"}
        assert _codes(coverage) == {"scan_mesh_low_support"}
        output = size.outputs
        report = output["report"]
        assert report["supported_points"] == report["signed_points"] == 5
        assert report["boundary_points"] == 2 and report["unsupported_points"] == 1
        assert report["in_tolerance_points"] == 3
        pairs = output["correspondences"]
        assert pairs["state"] == [2, 2, 3, 3, 1, 1, 0, 2]
        hit = np.array(pairs["triangle_index"]) >= 0
        np.testing.assert_allclose(
            np.array(pairs["distance_m"], float)[hit], expected[hit], atol=1e-10
        )
        np.testing.assert_allclose(np.stack([p for p in pairs["closest_xyz_m"] if p]), closest[hit])
        assert pairs["closest_xyz_m"][6] is None
        assert pairs["signed_distance_m"][4:7] == [None, None, None]
        assert pairs["signed_distance_m"][1] == pytest.approx(-0.0005, abs=4e-8)
        assert pairs["signed_distance_m"][2] == pytest.approx(0.0016, abs=4e-8)
        assert pairs["signed_distance_m"][3] == pytest.approx(-0.0015, abs=4e-8)
        assert ("Scan-to-mesh deviations" in size.graphics) == (profile == "debug")
        for port, rows in (("defects", [2, 3]), ("boundary", [4, 5]), ("unsupported", [6])):
            np.testing.assert_array_equal(output[port].points, scan.points[rows])
            np.testing.assert_array_equal(output[port].colors, scan.colors[rows])
    finally:
        graph.close()


def test_bounded_tree_matches_brute_force_on_separated_rectangles_and_multiple_batches():
    rng = np.random.default_rng(89)
    offsets = np.array([[x, y, 0.6 + 0.05 * (x + y)] for x in range(8) for y in range(5)])
    panel = np.array([[0, 0, 0], [0.7, 0, 0], [0, 0.8, 0], [0.7, 0.8, 0]])
    vertices = (offsets[:, None, :] + panel).reshape(-1, 3)
    faces = np.array([[0, 1, 2], [2, 1, 3]]) + np.arange(len(offsets))[:, None, None] * 4
    faces = faces.reshape(-1, 3)
    points = rng.uniform([-0.3, -0.3, 0.4], [8.1, 5.1, 1.4], (1300, 3))
    max_distance = 0.35
    low, high = offsets, offsets + [0.7, 0.8, 0]
    projected = np.clip(points[:, None, :], low, high)
    d = np.linalg.norm(points[:, None, :] - projected, axis=2)
    nearest = np.argmin(d, axis=1)
    rows = np.arange(len(points))
    hit = d[rows, nearest] <= max_distance

    surface = TriangleSurface(vertices, faces, 30)
    result, work = surface.query(points, max_distance, 100_000, lambda: False)
    assert result is not None
    np.testing.assert_array_equal(result.face >= 0, hit)
    np.testing.assert_allclose(result.distance[hit], d[rows, nearest][hit], atol=1e-12)
    np.testing.assert_allclose(result.closest[hit], projected[rows, nearest][hit], atol=1e-12)
    np.testing.assert_array_equal(result.face[hit] // 2, nearest[hit])
    expected_boundary = np.any(
        (projected[rows, nearest, :2] == low[nearest, :2])
        | (projected[rows, nearest, :2] == high[nearest, :2]),
        axis=1,
    )
    np.testing.assert_array_equal(result.boundary, hit & expected_boundary)
    assert 0 < work["triangle_tests"] < len(points) * len(faces) // 2


def test_closest_face_uses_interior_of_sloped_triangle_and_preserves_payload():
    a, b, c = np.array([[0, 0, 0.5], [2, 0, 0.75], [0, 2, 1]])
    normal = np.cross(b - a, c - a)
    normal /= np.linalg.norm(normal)
    expected = a + 0.25 * (b - a) + 0.35 * (c - a)
    mesh = TriangleMesh([a, b, c], [[0, 1, 2]], frame="part", normals=np.tile(-normal, (3, 1)))
    scan = PointCloud(
        [expected + normal * 0.02],
        frame="part",
        colors=[[1, 2, 3]],
        pixels=[[3, 4]],
        source_size=(5, 6),
    )
    ticket = TicketClock().issue("panel")
    spatial = frame([5, 6], "pixels")
    block = CompareScanToMesh(parameters={"comparison": "normal", "maximum_distance_m": 0.03})
    result = block.execute(
        {"scan": ValuePacket(scan, {"ticket": ticket, "spatial": spatial}), "reference": mesh}
    )
    assert result is not None and _codes(result) == {"scan_mesh_tolerance"}
    pair = result.outputs["correspondences"]
    assert pair["signed_distance_m"][0] == pytest.approx(0.02, abs=1e-7)
    np.testing.assert_allclose(pair["closest_xyz_m"][0], expected, atol=1e-7)
    np.testing.assert_array_equal(result.outputs["defects"].pixels, scan.pixels)
    assert result.payloads["defects"]["ticket"] == ticket
    assert result.payloads["defects"]["spatial"] == spatial


def test_sharp_edge_requires_compatible_incident_normals_for_signed_distance():
    mesh = TriangleMesh(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0, 1, 2], [0, 2, 3]],
    )
    scan = PointCloud([[-0.125, 0.375, -0.125]], frame="world")
    block = CompareScanToMesh(parameters={"comparison": "normal", "maximum_distance_m": 0.25})
    result = block.execute({"scan": scan, "reference": mesh})
    assert result is not None and _codes(result) == {"scan_mesh_low_signed_coverage"}
    out = result.outputs
    assert out["correspondences"]["state"] == [4]
    assert out["report"]["boundary_points"] == 0
    assert out["correspondences"]["signed_distance_m"] == [None]
    block.parameters.update(maximum_shared_normal_angle_deg=100, negative_tolerance_m=0.25)
    result = block.execute({"scan": scan, "reference": mesh})
    assert result and result.passed, block.error
    assert result.outputs["correspondences"]["signed_distance_m"] == pytest.approx(
        [-0.125 * np.sqrt(2)]
    )


def test_tied_surfaces_withhold_sign_but_choose_stable_triangle_id():
    mesh = TriangleMesh(
        [[0, 0, 0.5], [1, 0, 0.5], [0, 1, 0.5], [0, 0, -0.5], [1, 0, -0.5], [0, 1, -0.5]],
        [[0, 1, 2], [3, 4, 5]],
    )
    scan = PointCloud([[0.25, 0.25, 0]], frame="world")
    result = CompareScanToMesh(
        parameters={
            "maximum_distance_m": 0.5,
            "distance_tolerance_m": 0.5,
        }
    ).execute({"scan": scan, "reference": mesh})
    assert result and result.passed
    assert result.outputs["report"]["ambiguous_points"] == 1
    pairs = result.outputs["correspondences"]
    assert pairs["triangle_index"] == [0]
    assert pairs["distance_m"] == [0.5] and pairs["signed_distance_m"] == [None]


def test_degenerate_reference_and_work_limit_have_distinct_handoffs():
    scan = PointCloud([[0.25, 0.25, 0.01]], frame="world")
    degenerate = TriangleMesh([[0, 0, 0], [0.5, 0, 0], [1, 0, 0]], [[0, 1, 2]])
    tool = CompareScanToMesh(parameters={"maximum_distance_m": 0.05})
    no_surface = tool.execute({"scan": scan, "reference": degenerate})
    assert no_surface is not None and "scan_mesh_degenerate_reference" in _codes(no_surface)
    assert no_surface.outputs["report"]["degenerate_triangles"] == 1
    assert len(no_surface.outputs["unsupported"].points) == 1

    mesh = TriangleMesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
    tool.parameters["max_work"] = 1
    limited = tool.execute({"scan": scan, "reference": mesh})
    assert limited is not None and _codes(limited) == {"scan_mesh_work_limit"}
    assert limited.outputs["report"]["triangle_tests"] == 0
    assert limited.outputs["correspondences"] is NO_VALUE
    assert limited.outputs["defects"] is NO_VALUE
