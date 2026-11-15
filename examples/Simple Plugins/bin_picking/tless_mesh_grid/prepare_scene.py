"""Prepare a mesh-backed T-LESS 13 fixture; no simulator or robot I/O."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull


def read_ply(path):
    # The pinned upstream member is ASCII triangle PLY; no general-format claims.
    lines = path.read_text(encoding="ascii").splitlines()
    assert lines[1] == "format ascii 1.0"
    nv = int(next(s for s in lines if s.startswith("element vertex ")).split()[-1])
    nf = int(next(s for s in lines if s.startswith("element face ")).split()[-1])
    start = lines.index("end_header") + 1
    vertices = (
        np.array([[float(v) for v in s.split()[:3]] for s in lines[start : start + nv]]) * 0.001
    )
    faces = np.array([[int(v) for v in s.split()[1:]] for s in lines[start + nv : start + nv + nf]])
    assert vertices.shape == (nv, 3) and faces.shape == (nf, 3)
    return vertices, faces


def write_stl(path, triangles):
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-15)
    record = np.zeros(
        len(triangles), dtype=[("normal", "<f4", 3), ("xyz", "<f4", (3, 3)), ("attribute", "<u2")]
    )
    record["normal"], record["xyz"] = normal, triangles
    path.write_bytes(
        b"T-LESS CC BY 4.0; converted mm to m".ljust(80, b" ")
        + len(record).to_bytes(4, "little")
        + record.tobytes()
    )


def prepare(source, output, base):
    output.mkdir(parents=True, exist_ok=False)
    assets = output / "assets"
    assets.mkdir()
    raw = source.read_bytes()
    vertices, faces = read_ply(source)
    (assets / source.name).write_bytes(raw)
    visual = assets / "tless13.stl"
    collision = assets / "tless13_convex.stl"
    write_stl(visual, vertices[faces])
    hull = ConvexHull(vertices)
    triangles = vertices[hull.simplices].copy()
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    inward = np.einsum("ij,ij->i", normals, triangles.mean(1) - vertices.mean(0)) < 0
    triangles[inward] = triangles[inward][:, [0, 2, 1]]
    write_stl(collision, triangles)
    from opencv_simple_bin_picking.top_template import train_mesh_top

    template, _ = train_mesh_top(
        vertices,
        faces,
        source={
            "dataset": "T-LESS",
            "obj_id": 13,
            "model_sha256": hashlib.sha256(raw).hexdigest(),
            "units": "m",
        },
    )
    template_sha256 = template.save(assets / "top-template.json")
    scene = copy.deepcopy(base)
    for i, part in enumerate(scene["parts"]):
        part["size"] = [0.04, 0.04, 0.046]
        part["pose"][2] = 0.2235 + (i % 2) * 0.0475
        part["pose"][5] = [0, 0.17, -0.1][i // 2]
        part["mesh"] = {
            "visual": "assets/tless13.stl",
            "collision": "assets/tless13_convex.stl",
            "visual_sha256": hashlib.sha256(visual.read_bytes()).hexdigest(),
            "collision_sha256": hashlib.sha256(collision.read_bytes()).hexdigest(),
        }
        part["axial_grasp"] = {
            "yaw_symmetry": "continuous_z",
            "jaw_width_m": 0.04,
            "tcp_above_center_m": 0.018,
        }
        part["dataset"] = {
            "name": "T-LESS",
            "obj_id": 13,
            "license": "CC-BY-4.0",
            "model_revision": "5fd309a04476a842d93abfb584fba9ee7caecdf1",
        }
    (output / "scene.json").write_text(
        json.dumps(scene, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    provenance = dict(
        dataset="T-LESS",
        obj_id=13,
        license="CC-BY-4.0",
        publisher="https://cmp.felk.cvut.cz/t-less/",
        revision="5fd309a04476a842d93abfb584fba9ee7caecdf1",
        archive="tless_models.zip",
        archive_sha256="6a29c59766b8d2af05e62e71739f0ca7243ad81bc3c7f9a24925504a8cb37928",
        member="models_cad/obj_000013.ply",
        member_sha256=hashlib.sha256(raw).hexdigest(),
        conversion=(
            "original model origin/axes; mm to m; original visual triangles; "
            "outward convex collision hull"
        ),
        extents_m=np.ptp(vertices, axis=0).tolist(),
        visual_triangles=len(faces),
        collision_triangles=len(triangles),
        ground_truth=(
            "new simulator poses from native pose feedback; "
            "original BOP pose labels retained separately"
        ),
        authored=(
            "bin arrangement, poses, mass, envelope inertia, convex collision "
            "approximation and axial grasp frame"
        ),
        planning_collision="conservative 40x40x46 mm oriented box, not exact mesh",
        source_symmetry="continuous model-Z rotation declared in upstream models_info.json",
        top_template_sha256=template_sha256,
    )
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return provenance


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    from opencv_bin_picking_fixture.scene import default_scene

    print(json.dumps(prepare(args.model, args.output, default_scene()), indent=2))
