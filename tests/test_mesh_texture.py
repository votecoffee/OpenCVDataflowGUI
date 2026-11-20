"""Calibrated forward texture baking against independent panel and donor geometry."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import camera_record, pose_record
from opencv_dataflow.images import ImageArray
from opencv_dataflow.mesh_texture import BakeMeshTexture
from opencv_dataflow.meshes import TriangleMesh
from opencv_dataflow.payloads import NO_VALUE

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/opencv tools/mesh_texture"


def _inside(x, y, bounds):
    left, top, right, bottom = bounds
    return left < x < right and top < y < bottom


def _expected(sampling, image, selection=None):
    """Evaluate the two panel rectangles and source pixels without native rasterization."""
    height, width = image.shape[:2]
    atlas = np.zeros((64, 128) + image.shape[2:], image.dtype)
    valid = np.zeros((64, 128), np.uint8)
    back = (-6.2, 3.2, 79.2, 56.2)
    front = (27.2, 19.3, 53.8, 38.7)
    for left, top, columns, rows, bounds in ((0, 0, 80, 64, back), (88, 12, 40, 40, front)):
        x0, y0, x1, y1 = bounds
        corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
        plane_depth = np.ones(4)
        if left:
            plane_depth /= 1 - 0.3 * (corners[:, 0] - 39.5) / 71 + 0.1 * (corners[:, 1] - 29.5) / 64
        for j in range(rows):
            for i in range(columns):
                u, v = (i + 0.5) / columns, (j + 0.5) / rows
                weights = [1 - u, u - v, 0, v] if u >= v else [1 - v, 0, v - u, u]
                x, y = np.average(corners, weights=np.multiply(weights, plane_depth), axis=0)
                if not -0.5 <= x < width - 0.5 or not -0.5 <= y < height - 0.5:
                    continue
                if sampling == "nearest":
                    donors = [(int(x + 0.5), int(y + 0.5), 1.0)]
                else:
                    low_x, low_y = int(np.floor(x)), int(np.floor(y))
                    donors = [
                        (a, b, (1 - abs(a - x)) * (1 - abs(b - y)))
                        for a in (low_x, low_x + 1)
                        for b in (low_y, low_y + 1)
                        if (1 - abs(a - x)) * (1 - abs(b - y)) > 1e-7
                    ]
                value = 0.0
                for a, b, weight in donors:
                    if (
                        not 0 <= a < width
                        or not 0 <= b < height
                        or not _inside(a, b, bounds)
                        or (not left and _inside(a, b, front))
                        or (selection is not None and not selection[b, a])
                        or not np.isfinite(image[b, a]).all()
                    ):
                        break
                    value = value + image[b, a] * weight
                else:
                    atlas[top + j, left + i] = np.rint(value) if image.dtype.kind == "u" else value
                    valid[top + j, left + i] = 255
    return atlas, valid


def _assert_same_mesh(actual, expected):
    assert actual.frame == expected.frame and actual.units == expected.units
    for name in ("vertices", "faces", "colors", "texcoords", "normals"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


@pytest.mark.local_examples
@pytest.mark.parametrize("profile", ["debug", "lean"])
def test_saved_texture_atlas_against_camera_and_uv_reference(profile):
    graph = Graph.load(EXAMPLE / "mesh_texture.json", builtin_registry())
    try:
        graph.set_run_profile(profile)
        graph.set_branch_workers(1 if profile == "debug" else 2)
        graph.run()
        source = graph.blocks["image"].last_result
        mesh = graph.blocks["panels"].last_result.outputs["mesh"]
        for sampling, count in (("bilinear", 5490), ("nearest", 5835)):
            result = graph.blocks[sampling].last_result
            assert result.passed
            outputs = result.outputs
            reference, support = _expected(sampling, source.outputs["image"])
            np.testing.assert_array_equal(outputs["texture_mask"], support)
            np.testing.assert_allclose(outputs["texture"], reference, atol=1, rtol=0)
            _assert_same_mesh(outputs["mesh"], mesh)
            assert outputs["texture"].color_format == "bgr"
            assert outputs["report"]["atlas_texels"] == 6720
            assert outputs["report"]["valid_texels"] == count
            assert outputs["texture_mask"][32, 42] == 0  # Rear panel behind the foreground.
            assert outputs["texture_mask"][32, 107] == 255  # The slanted foreground itself.
            for port in ("texture", "texture_mask"):
                assert result.payloads[port]["spatial"]["root_id"] == f"{sampling}:uv"
                assert result.payloads[port]["ticket"] == source.payloads["image"]["ticket"]
    finally:
        graph.close()


def _planar_inputs():
    k = np.array([[8, 0, 3.5], [0, 8, 3.5], [0, 0, 1]])
    model = camera_record(k, [], [8, 8], frame="camera")
    surface = [[-0.5, -0.5, 1], [0.5, -0.5, 1], [-0.5, 0.5, 1], [0.5, 0.5, 1]]
    faces = [[0, 1, 3], [0, 3, 2]]
    uv = [[0, 1], [1, 1], [0, 0], [1, 0]]
    mesh = TriangleMesh(
        surface,
        faces,
        frame="camera",
        texcoords=uv,
        colors=np.full((4, 3), 64),
        normals=np.tile([0, 0, 1], (4, 1)),
    )
    yy, xx = np.mgrid[:8, :8]
    return {
        "image": ImageArray((xx + 8 * yy).astype(np.uint16) * 900, "gray"),
        "mesh": mesh,
        "camera": model,
    }


def test_strict_donors_intersect_mask_region_and_preserve_binary16_values():
    inputs = _planar_inputs()
    mesh = inputs["mesh"]
    pose = np.eye(4)
    pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    pose[:3, 3] = [125, -250, 500]
    local = replace(
        mesh, vertices=(mesh.vertices * 1000 - pose[:3, 3]) @ pose[:3, :3], frame="part", units="mm"
    )
    inputs.update(mesh=local, transform=pose_record(pose, "part", "camera", "mm"))
    inputs["mask"] = np.full((8, 8), 255, np.uint8)
    inputs["mask"][2, 3] = 0
    inputs["region"] = {
        "model": "image_region_v1",
        "image_size": [8, 8],
        "shape": "rectangle",
        "rectangle": [2, 1, 4, 6],
    }
    block = BakeMeshTexture(parameters={"texture_width": 4, "texture_height": 4})
    result = block.execute(inputs)
    assert result is not None and result.passed, block.error
    outputs = result.outputs
    _assert_same_mesh(outputs["mesh"], local)
    # Each atlas center maps to the center of a disjoint 2×2 source group.
    expected_mask = np.zeros((4, 4), np.uint8)
    expected_mask[2, 1] = expected_mask[2, 2] = expected_mask[1, 2] = 255
    np.testing.assert_array_equal(outputs["texture_mask"], expected_mask)
    assert outputs["texture"].dtype == np.uint16
    assert outputs["texture"][1, 2] == 22_050
    assert outputs["texture"][2, 2] == 36_450
    assert not outputs["texture"][1, 1]


def test_nonfinite_and_transparent_donors_are_invalid_while_black_is_valid():
    inputs = _planar_inputs()
    a = np.zeros((8, 8, 4), np.float32)
    a[..., 3] = 1
    a[4, 5, 0] = np.nan
    a[2, 3, 3] = 0
    inputs["image"] = ImageArray(a, "rgba")
    result = BakeMeshTexture(parameters={"texture_width": 8, "texture_height": 8}).execute(inputs)
    data = result.outputs
    assert result.passed and data["report"]["valid_texels"] == 62
    assert data["texture"].dtype == np.float32 and data["texture"].color_format == "rgba"
    assert data["texture_mask"][0, 0] == 255
    assert data["texture_mask"][2, 3] == data["texture_mask"][4, 5] == 0
    assert (data["texture"][data["texture_mask"] == 0] == 0).all()


def test_budgets_overlaps_invalid_uvs_and_coverage_withhold_mesh():
    inputs = _planar_inputs()
    mesh = inputs["mesh"]
    for parameters, model, mask, code in (
        ({"max_triangles": 1}, mesh, None, "triangle_budget"),
        ({"max_pixel_tests": 1}, mesh, None, "pixel_budget"),
        ({}, replace(mesh, faces=[[0, 1, 3], [3, 1, 0]]), None, "overlapping_uv"),
        ({}, mesh, np.zeros((8, 8), np.uint8), "no_visible_texels"),
        ({"minimum_coverage": 1}, mesh, np.eye(8, dtype=np.uint8), "insufficient_coverage"),
    ):
        block = BakeMeshTexture(parameters={"texture_width": 8, "texture_height": 8, **parameters})
        result = block.execute(
            {**inputs, "mesh": model, **({"mask": mask} if mask is not None else {})}
        )
        assert result is not None and not result.passed, block.error
        assert result.outputs["report"]["status"] == code
        assert result.outputs["mesh"] is NO_VALUE
    block = BakeMeshTexture()
    assert block.execute({**inputs, "mesh": replace(mesh, texcoords=None)}) is None
    assert "UV coordinates" in block.error
    distorted = {**inputs["camera"], "distortion": [0.001, 0, 0, 0, 0]}
    assert block.execute({**inputs, "camera": distorted}) is None
    assert "rectified" in block.error
