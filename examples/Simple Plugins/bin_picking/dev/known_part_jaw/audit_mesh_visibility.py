"""Compare packaged mesh diagnostics with the frozen renderer on every retained pose."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.plugins import load_plugin


def audit(root, output):
    from opencv_simple_3d_locate import Camera as ReferenceCamera
    from opencv_simple_3d_locate import read_ply_mesh
    from opencv_simple_3d_locate.projective import render_depth as reference
    from opencv_simple_bin_picking import Camera, PartInstance, PartMesh
    from opencv_simple_bin_picking.jaw_mesh import inspect_mesh
    from opencv_simple_bin_picking.jaw_persistence import load_jaw_recipe
    from opencv_simple_bin_picking.jaw_raster import render_depth
    from opencv_simple_bin_picking.scene import Scene

    output.mkdir(parents=True, exist_ok=False)
    folder = root / "pose_agreement"
    study = json.loads((folder / "comparison.json").read_bytes())
    base = load_jaw_recipe(root / "tless/frame000013/assets/trained.sbpj").bin
    meshes, scenes = {}, {}
    rows = []
    for record in study["cases"]:
        scene_id, frame = record["scene"], record["frame"]
        key = record["object_id"], record["model"]
        if key not in meshes:
            path = folder / f"source/models_{key[1]}/obj_{key[0]:06}.ply"
            original = read_ply_mesh(path, scale=0.001)
            meshes[key] = original, PartMesh(original.triangles, f"tless{key[0]}")
        if (scene_id, frame) not in scenes:
            source = folder / f"source/test_primesense/{scene_id:06}"
            metadata = json.loads((source / "scene_camera.json").read_bytes())[str(frame)]
            raw = cv2.imread(str(source / f"depth/{frame:06}.png"), -1)
            depth = (raw.astype(float) * metadata["depth_scale"] * 0.001).astype(np.float32)
            matrix = np.array(metadata["cam_K"]).reshape(3, 3)
            camera = Camera(matrix, depth.shape[1::-1])
            scene = Scene(
                replace(
                    base,
                    camera=camera,
                    reference_depth=depth,
                    bin_to_camera=np.eye(4),
                    frame="study_camera",
                ),
                depth,
            )
            scenes[scene_id, frame] = scene
        scene = scenes[scene_id, frame]
        original, mesh = meshes[key]
        for stage, field in (("before", "annotation_pose"), ("after", "research_pose")):
            instance = PartInstance("audit", np.array(record[field]), True, "frozen_study")
            pose = instance.camera_from_part
            expected, _, stride, reason = reference(
                original,
                pose,
                ReferenceCamera(
                    np.array(scene.recipe.camera.matrix), scene.recipe.camera.image_size, "m"
                ),
            )
            actual, _, actual_stride, actual_reason = render_depth(mesh, pose, scene.recipe.camera)
            assert actual_stride == stride and actual_reason == reason
            finite = np.isfinite(expected)
            np.testing.assert_array_equal(np.isfinite(actual), finite)
            error = float(np.max(abs(actual[finite] - expected[finite]))) if finite.any() else 0
            assert error <= 1e-12
            review = inspect_mesh(mesh, scene, instance, 0.003)
            observed = scene.depth[::stride, ::stride]
            known = finite & np.isfinite(observed) & (observed > 0)
            delta = expected - observed
            counts = dict(
                supported=int((known & (abs(delta) <= 0.003)).sum()),
                contradiction=int((known & (delta < -0.003)).sum()),
                occluded=int((known & (delta > 0.003)).sum()),
                unknown=int((finite & ~known).sum()),
                ambiguous=0,
                label_mismatch=0,
            )
            assert review.summary["counts"] == counts
            assert review.summary["acceptance_applied"] is False
            rows.append(
                dict(
                    scene=scene_id,
                    frame=frame,
                    object_id=key[0],
                    model=key[1],
                    stage=stage,
                    stride=stride,
                    first_surface_mask_exact=True,
                    state_counts_exact=True,
                    maximum_depth_difference_m=error,
                )
            )
    receipt = dict(
        model="retained_mesh_renderer_audit_v1",
        poses_checked=len(rows),
        cases=rows,
        qualification="Shared algorithm parity and analytic controls; no new pose gate",
        runtime_poses_or_gates_changed=False,
    )
    (output / "audit.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            dict(
                poses_checked=len(rows),
                state_counts_exact=True,
                maximum_depth_difference_m=max(r["maximum_depth_difference_m"] for r in rows),
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    registry = builtin_registry()
    load_plugin(args.root.resolve() / "plugin.py", registry)
    audit(args.root.resolve(), args.output.resolve())
