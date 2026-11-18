"""Frozen supplied-pose ablations, native colored clouds and explicit failure views."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from .build_tless_demo import tless_mesh, write_json
    from .tless_validation import read_capture
else:
    from build_tless_demo import tless_mesh, write_json
    from tless_validation import read_capture

from opencv_simple_3d_locate import Locator, train_cad
from opencv_simple_3d_locate.diagnostics import PALETTE, diagnose_pose

VARIANTS = (
    "runtime_baseline", "full_model_voxels", "dense_scene_normals", "six_normal_neighbors",
    "dense_scene", "triangle_visibility", "triangle_visibility_full_model",
)


def _display_cloud(diagnostic, pose, diameter):
    from opencv_dataflow.pointclouds import PointCloud

    points = [diagnostic.scene[:, :3], diagnostic.model[:, :3]]
    colors = [np.full((len(points[0]), 3), 115, np.uint8), diagnostic.colors]
    for axis in range(3):
        line = np.zeros((32, 3))
        line[:, axis] = np.linspace(0, diameter * 0.2, len(line))
        points.append(line @ pose[:3, :3].T + pose[:3, 3])
        color = np.zeros(3, np.uint8)
        color[axis] = 255
        colors.append(np.tile(color, (len(line), 1)))
    return PointCloud(np.concatenate(points), "mm", "camera",
                      colors=np.asarray(np.concatenate(colors), np.uint8))


def _photo_overlay(capture, diagnostic):
    image = capture.image.copy()
    points = diagnostic.model[:, :3]
    pixels = points @ capture.camera.matrix.T
    valid = pixels[:, 2] > 1e-9
    pixels = np.rint(pixels[:, :2] / np.maximum(pixels[:, 2:], 1e-9)).astype(int)
    for name in PALETTE:
        keep = valid & (diagnostic.categories == name)
        for (u, v), color in zip(pixels[keep], diagnostic.colors[keep]):
            if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                cv2.circle(image, (int(u), int(v)), 1, tuple(int(x) for x in color[::-1]), -1)
    canvas = np.full((image.shape[0] + 130, image.shape[1], 3), 28, np.uint8)
    canvas[60:60 + image.shape[0]] = image
    coverage = diagnostic.report["gates"]["coverage"]["value"]
    lines = ["ANNOTATED POSE ONLY - NOT A DETECTION",
             f"{diagnostic.report['variant']} | geometric support {coverage:.1%}",
             "Green support | Red distance rejected | Orange normal rejected",
             "Blue-gray hidden | Magenta unresolved | Scene photograph"]
    for text, y in zip(lines, (22, 45, canvas.shape[0] - 44, canvas.shape[0] - 18)):
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .47,
                    (235, 235, 235), 1, cv2.LINE_AA)
    return canvas


def _project(folder, title):
    from opencv_dataflow.blocks import builtin_registry
    from opencv_dataflow.core import Graph

    registry, graph = builtin_registry(), Graph()
    try:
        note = graph.add(registry.create("layout.comment", block_id="diagnostic_scope"))
        note.name = "ANNOTATED POSE ONLY - NOT A DETECTION"
        note.description = (
            f"{title}. Stored supplied-pose diagnostics; this graph runs no locator. "
            "Orbit the native RGB clouds to compare sampled and all-triangle visibility. "
            "Green: supported; red: distance rejected; orange: normal rejected; "
            "blue-gray: hidden; magenta: unresolved; gray: measured scene; RGB: pose axes. "
            "Full gate values, ablations and hashes are in report.json. "
            "T-LESS by Tomas Hodan, Pavel Haluza, Stepan Obdrzalek, Jiri Matas, "
            "Manolis Lourakis and Xenophon Zabulis, WACV 2017, CC BY 4.0. "
            "https://cmp.felk.cvut.cz/t-less/ ; see ../ATTRIBUTION.md."
        )
        note.node_size, note.position = "large", (0, -300)
        for i, name in enumerate(("runtime_baseline", "triangle_visibility")):
            block = graph.add(registry.create("input.cloud", block_id=name, parameters={
                "path": f"{name}.ply", "units": "mm", "frame": "camera",
            }))
            block.name, block.position, block.preview_output = name, (i * 420, 80), "cloud"
        graph.save(folder / "project.json")
        graph.run()
        for block in graph.blocks.values():
            if block.error:
                raise RuntimeError(block.error)
    finally:
        graph.close()


def build(source, output, *, host_projects=False):
    source, output = Path(source), Path(output)
    if any(output.glob("object*/*.ply")):
        raise ValueError("Diagnostic clouds already exist; choose a fresh output folder.")
    output.mkdir(parents=True, exist_ok=True)
    (output / ".gitattributes").write_bytes(
        b"# Generated evidence hashes refer to exact bytes on every platform.\n"
        b"*.json -text\nATTRIBUTION.md -text\n"
    )
    selection = json.loads((source / "selection.json").read_bytes())
    provenance = json.loads((source / "provenance.json").read_bytes())
    for name, expected in provenance["sha256"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Pinned source mismatch: {name}")
    (output / "ATTRIBUTION.md").write_text(
        "# T-LESS diagnostic data attribution\n\n"
        "T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects, "
        "Tomas Hodan, Pavel Haluza, Stepan Obdrzalek, Jiri Matas, Manolis Lourakis, "
        "and Xenophon Zabulis, WACV 2017.\n\n"
        "[Original project, citation and license](https://cmp.felk.cvut.cz/t-less/) · "
        "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · "
        "[Official BOP mirror](https://huggingface.co/datasets/bop-benchmark/tless).\n\n"
        "Source revision: `5fd309a04476a842d93abfb584fba9ee7caecdf1`. "
        "Object 6 uses scene 2 captures 3/255/494; object 8 uses scene 3 captures "
        "5/261/493. Source member hashes are in source_hashes.json.\n\n"
        "Modifications: backproject measured depth in millimeters, apply supplied "
        "visible masks, sample CAD, color evidence categories at annotated poses, "
        "and draw secondary photographic overlays. No scene returns are invented. "
        "All poses here are supplied annotations, never detections.\n\n"
        "Raw data and these derived clouds/photographs retain CC BY 4.0; "
        "the diagnostic implementation is MIT. No endorsement is implied.\n",
        encoding="utf-8",
    )
    # Relative upstream member names only: do not persist a local machine path.
    write_json(output / "source_hashes.json", provenance["sha256"])
    rows, pictures = [], []
    for item in selection["models"]:
        recipe = train_cad(
            tless_mesh(source / f"models_cad/obj_{item['object_id']:06}.ply"), units="mm"
        )
        locator = Locator(recipe)
        for frame in item["positive_frames"]:
            capture = read_capture(source, item["positive_scene"], frame,
                                   item["object_id"], isolated=True)
            folder = output / f"object{item['object_id']:06}_{capture.name}"
            folder.mkdir(exist_ok=True)
            reports, views = [], []
            for variant in VARIANTS:
                diagnostic = diagnose_pose(locator, capture.points, capture.truth[0],
                                           camera=capture.camera, variant=variant)
                reports.append(diagnostic.report)
                print(f"object {item['object_id']} {frame} {variant}: "
                      f"{diagnostic.report['gates']['coverage']['value']:.3f}", flush=True)
                if variant in ("runtime_baseline", "triangle_visibility"):
                    from opencv_dataflow.cloud_io import write_cloud

                    write_cloud(folder / f"{variant}.ply",
                                _display_cloud(diagnostic, capture.truth[0], recipe.diameter),
                                binary=True)
                    view = _photo_overlay(capture, diagnostic)
                    cv2.imwrite(str(folder / f"{variant}.png"), view)
                    views.append(view)
            row = {
                "scope": "ANNOTATED_POSE_ONLY_NOT_A_DETECTION",
                "object_id": item["object_id"], "case": capture.name,
                "settings": asdict(recipe.settings), "reports": reports,
                "palette_rgb": PALETTE,
            }
            write_json(folder / "report.json", row)
            if host_projects:
                _project(folder, f"T-LESS object {item['object_id']}, {capture.name}")
            pictures.append(cv2.resize(np.concatenate(views, axis=1), (960, 447)))
            rows.append(row)
    summary = {
        "scope": "ANNOTATED_POSE_ONLY_NOT_A_DETECTION",
        "search_executed": False, "annotation_inputs_never_count_as_detections": True,
        "cases": len(rows), "variants": {
            variant: {"geometry_pass": sum(next(r for r in row["reports"]
                if r["variant"] == variant)["geometry_pass"] for row in rows)}
            for variant in VARIANTS
        },
        "matching_results_unchanged": True,
        "note": "Offline diagnostics explain pose feasibility only. No runtime algorithm, "
                "recipe or acceptance gate was changed. These are development cases.",
    }
    write_json(output / "summary.json", summary)
    cv2.imwrite(str(output / "overview.jpg"), np.concatenate(pictures, axis=0))
    generated = [output / name for name in
                 ("ATTRIBUTION.md", "source_hashes.json", "summary.json", "overview.jpg")]
    generated.extend(p for p in output.glob("object*/*")
                     if p.suffix in (".ply", ".png") or p.name in ("project.json", "report.json"))
    write_json(output / "artifact_hashes.json", {
        p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(generated)
    })
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--host-projects", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, host_projects=args.host_projects), indent=2))
