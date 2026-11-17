"""Compare retained pose support to an independent CPU triangle-render diagnostic."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def diagnose(root, mesh_recipe, output):
    # Optional geometry utilities only: no learned inference or weights loaded.
    from opencv_simple_foundation_pose.persistence import load_recipe
    from opencv_simple_foundation_pose.render import render

    from opencv_simple_bin_picking.jaw import JawPlanner
    from opencv_simple_bin_picking.jaw_persistence import load_jaw_recipe
    from opencv_simple_bin_picking.jaw_specs import PartInstance

    output.mkdir(parents=True, exist_ok=False)
    mesh = load_recipe(mesh_recipe).mesh
    rows = []
    for frame in (3, 13):
        case = root / f"tless/frame{frame:06}"
        recipe = load_jaw_recipe(case / "assets/trained.sbpj")
        record = json.loads((case / "assets/records.json").read_bytes())
        instance = PartInstance.from_record(
            record["part_instances"][0],
            part_frame=recipe.part.frame,
            camera_frame=recipe.bin.camera.frame,
        )
        depth = np.load(case / "assets/depth.npy", allow_pickle=False)
        baseline = JawPlanner(recipe).plan(depth, [instance], capture_id=record["capture_id"])
        pose = instance.camera_from_part.copy()
        # Renderer takes a centered model pose; explicitly preserve the original origin.
        pose[:3, 3] += pose[:3, :3] @ mesh.center
        _, xyz = render(mesh, pose, np.array(recipe.bin.camera.matrix), shape=depth.shape)
        predicted = xyz[..., 2]
        mask, known = predicted > 0, np.isfinite(depth) & (depth > 0)
        delta = predicted - depth
        tolerance = recipe.settings.pose_tolerance
        support = mask & known & (abs(delta) <= tolerance)
        contradiction = mask & known & (delta < -tolerance)
        occluded = mask & known & (delta > tolerance)
        unknown = mask & ~known
        image = cv2.imread(str(root / f"tless/source/test_primesense/000002/rgb/{frame:06}.png"))
        if image is None:
            raise ValueError("Retain the source RGB for the diagnostic overlay")
        colors = np.zeros_like(image)
        for selected, color in (
            (support, (70, 230, 80)),
            (contradiction, (230, 70, 230)),
            (occluded, (230, 150, 70)),
            (unknown, (40, 170, 255)),
        ):
            colors[selected] = color
        image[mask] = np.rint(0.45 * image[mask] + 0.55 * colors[mask]).astype(np.uint8)
        cv2.putText(
            image,
            "ANNOTATION-POSE DIAGNOSTIC / REJECTED",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 210, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(output / f"frame{frame:06}_triangle_overlay.png"), image)
        count = int(mask.sum())
        coverage, bad = float(support.sum() / count), float(contradiction.sum() / count)
        rows.append(
            dict(
                frame=frame,
                annotation_pose_assistance=True,
                tolerance_m=tolerance,
                current_four_neighbor_point_evidence=baseline.pose_evidence[0],
                independent_full_resolution_triangle=dict(
                    rendered_rays=count,
                    coverage=coverage,
                    contradiction=bad,
                    unknown=int(unknown.sum()),
                    occluded=int(occluded.sum()),
                    accepted=coverage >= recipe.settings.minimum_pose_support
                    and bad <= recipe.settings.maximum_pose_contradiction,
                ),
                method_difference="Four-neighbor core vs exact-pixel triangle diagnostic",
                conclusion="Both reject; a sampling-only explanation is not established",
            )
        )
    receipt = dict(
        rows=rows,
        mesh_recipe_sha256=hashlib.sha256(mesh_recipe.read_bytes()).hexdigest(),
        no_pose_or_clearance_gate_changed=True,
        robot_execution=False,
        scope="Diagnostic only, not independent localization or calibrated accuracy",
        renderer_scope="Development CPU rasterizer, unqualified against native CUDA rendering",
        legend="Green support; magenta free-space contradiction; blue occluded; orange unknown",
    )
    (output / "diagnostic.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "mesh_recipe", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    diagnose(args.root, args.mesh_recipe, args.output)
