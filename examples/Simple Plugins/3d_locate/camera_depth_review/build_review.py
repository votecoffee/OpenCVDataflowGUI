"""Retain exact camera-ray evidence on frozen real captures; annotations are not detections."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

if __package__:
    from .build_tless_demo import tless_mesh, write_json
    from .tless_validation import read_capture
else:
    from build_tless_demo import tless_mesh, write_json
    from tless_validation import read_capture

from opencv_simple_3d_locate import train_cad
from opencv_simple_3d_locate.depth import DepthBuffer
from opencv_simple_3d_locate.depth_review import RAY_STATES, inspect_depth


def make_project(folder, title):
    from opencv_dataflow.blocks import builtin_registry
    from opencv_dataflow.core import Graph

    registry, graph = builtin_registry(), Graph()
    try:
        block = graph.add(registry.create("input.cloud", block_id="camera_depth", parameters={
            "path": "expected_and_measured.ply", "units": "mm", "frame": "camera",
        }))
        block.preview_output, block.position = "cloud", (0, 80)
        block.name = "Annotated pose ray evidence - not a detection"
        note = graph.add(registry.create("layout.comment", block_id="scope"))
        note.name = title + " - ANNOTATION ONLY"
        note.description = (
            "This stored evidence graph runs no locator. Colored points are expected CAD "
            "geometry, not extra sensor returns; gray points are measured depth. "
            "Green: ray agreement, not pose acceptance. Red: farther return/free space. "
            "Orange: nearer return. Magenta: missing exact depth. Blue-gray: grazing ray. "
            "RGB axes: supplied annotated pose. Orbit the native RGB cloud; full native maps, "
            "camera, counts, physical tolerance and source hashes are exported alongside it. "
            "T-LESS, Hodan et al., WACV 2017, CC BY 4.0; see ../ATTRIBUTION.md and "
            "https://cmp.felk.cvut.cz/t-less/. No annotated pose counts as a detection."
        )
        note.node_size, note.position = "large", (0, -250)
        graph.save(folder / "project.json")
        graph.run()
        if block.error:
            raise RuntimeError(block.error)
    finally:
        graph.close()


def build(source, output):
    from opencv_simple_3d_locate.qt_depth_review import review_cloud

    from opencv_dataflow.cloud_io import write_cloud

    source, output = Path(source), Path(output)
    if any(output.glob("object*/*.ply")):
        raise ValueError("Camera-ray evidence exists; choose a fresh output folder.")
    selection = json.loads((source / "selection.json").read_bytes())
    provenance = json.loads((source / "provenance.json").read_bytes())
    for name, expected in provenance["sha256"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Pinned source mismatch: {name}")
    output.mkdir(parents=True, exist_ok=True)
    (output / ".gitattributes").write_bytes(b"*.json -text\nATTRIBUTION.md -text\n")
    write_json(output / "source_hashes.json", provenance["sha256"])
    rows = []
    for item in selection["models"]:
        recipe = train_cad(
            tless_mesh(source / f"models_cad/obj_{item['object_id']:06}.ply"), units="mm"
        )
        settings = recipe.settings
        for frame in item["positive_frames"]:
            capture = read_capture(source, item["positive_scene"], frame,
                                   item["object_id"], isolated=True)
            review = inspect_depth(
                recipe.cad.mesh, capture.truth[0],
                DepthBuffer(capture.depth.points, capture.camera),
                tolerance=settings.distance_tolerance * recipe.diameter,
                minimum_coverage=settings.min_depth_coverage,
                minimum_rays=settings.min_unique_support,
            )
            folder = output / f"object{item['object_id']:06}_{capture.name}"
            folder.mkdir(exist_ok=True)
            write_cloud(folder / "expected_and_measured.ply",
                        review_cloud(review, scene=capture.points), binary=True)
            np.savez_compressed(folder / "camera_ray_maps.npz", expected=review.expected,
                                measured=review.measured, residual=review.residual,
                                states=review.states, stride=review.stride)
            report = {
                **review.report, "object_id": item["object_id"], "case": capture.name,
                "camera": capture.camera.record(), "settings": asdict(settings),
                "annotation_only": True, "search_executed": False,
                "depth_source": "full_measured_frame", "display_scene": "supplied_visible_mask",
                "state_codes": dict(enumerate(RAY_STATES)),
            }
            write_json(folder / "report.json", report)
            make_project(folder, f"T-LESS object {item['object_id']} capture {frame}")
            rows.append({key: report[key] for key in (
                "object_id", "case", "status", "known_rays", "rendered_rays", "depth_coverage",
                "counts", "residual_percentiles", "distance_tolerance",
            )})
            print(json.dumps(rows[-1]), flush=True)
    write_json(output / "summary.json", {
        "scope": "ANNOTATED_POSES_ONLY_NOT_DETECTIONS", "cases": rows,
        "localization_changed": False, "interpolation_or_hole_filling": False,
    })
    (output / "ATTRIBUTION.md").write_bytes(
        b"# T-LESS camera-depth review attribution\n\n"
        b"T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects, "
        b"Tomas Hodan, Pavel Haluza, Stepan Obdrzalek, Jiri Matas, Manolis Lourakis "
        b"and Xenophon Zabulis, WACV 2017.\n\n"
        b"[Original project/license](https://cmp.felk.cvut.cz/t-less/) | "
        b"[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).\n\n"
        b"BOP mirror revision 5fd309a04476a842d93abfb584fba9ee7caecdf1; source "
        b"member hashes are in source_hashes.json. Modifications: backproject real "
        b"depth, render CAD at supplied annotations, compare exact pixels and color "
        b"expected surfaces. No sensor returns are invented and no annotations are "
        b"detections. Derived data retains CC BY 4.0; code is MIT. No endorsement implied.\n"
    )
    files = [output / n for n in ("summary.json", "source_hashes.json", "ATTRIBUTION.md")]
    files.extend(p for p in output.glob("object*/*")
                 if p.suffix in (".ply", ".npz") or p.name in ("project.json", "report.json"))
    write_json(output / "artifact_hashes.json", {
        p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(files)
    })
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.source, args.output)
