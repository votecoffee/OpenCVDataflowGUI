"""Annotated-pose geometry diagnostics. These runs are never detections."""

import argparse
import json
from pathlib import Path

import cv2
from build_tless_demo import tless_mesh, write_json
from opencv_simple_3d_locate import Locator, train_cad
from opencv_simple_3d_locate.depth import DepthBuffer
from opencv_simple_3d_locate.model import prepared
from opencv_simple_3d_locate.observation import Observation, first_surface
from opencv_simple_3d_locate.projective import score
from tless_validation import validation_captures


def diagnose(source, output):
    source = Path(source)
    selection = json.loads((source / "selection.json").read_bytes())
    rows = []
    for item in selection["models"]:
        recipe = train_cad(
            tless_mesh(source / f"models_cad/obj_{item['object_id']:06}.ply"), units="mm"
        )
        locator, settings = Locator(recipe), recipe.settings
        for capture in validation_captures(source, item):
            if not capture.isolated:
                continue
            data = capture.points[first_surface(capture.points, camera=capture.camera)]
            data = prepared(
                data,
                settings.sampling_distance * recipe.diameter,
                settings.normal_neighbors,
                settings.scene_viewpoint,
                100000,
            )
            center = (data[:, :3].min(0).astype(float) + data[:, :3].max(0)) / 2
            data[:, :3] = (data[:, :3] - center) / recipe.diameter
            truth = capture.truth[0]
            normalized = truth.copy()
            normalized[:3, 3] = (
                truth[:3, :3] @ recipe.center + truth[:3, 3] - center
            ) / recipe.diameter
            observation = Observation(
                center,
                recipe.diameter,
                settings.scene_viewpoint,
                settings.sampling_distance * recipe.diameter,
                capture.camera,
            )
            evidence = locator._evidence(
                data,
                cv2.flann_Index(data[:, :3].copy(), {"algorithm": 4}),
                normalized,
                observation=observation,
            )
            rays = score(
                recipe.cad.mesh,
                truth,
                DepthBuffer(capture.depth.points, capture.camera),
                tolerance=settings.distance_tolerance * recipe.diameter,
                minimum_coverage=settings.min_depth_coverage,
                minimum_rays=settings.min_unique_support,
            )
            rows.append(
                dict(
                    object_id=item["object_id"],
                    case=capture.name,
                    diagnostic="ANNOTATED_POSE_ONLY_NOT_DETECTION",
                    coverage=evidence[0],
                    physical_rmse=evidence[1] * recipe.diameter,
                    normal_spread=evidence[2],
                    unique_support=len(evidence[4]),
                    geometric_pass=bool(locator._passes(evidence)),
                    ray_score=rays["score"],
                )
            )
    write_json(
        output,
        dict(
            scope="Annotated poses supplied only to evidence diagnostics. "
            "No search is run; no diagnostic pose counts as a detection.",
            cases=rows,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    diagnose(args.source, args.output)
