"""Compare fixed CAD, reconstructed mesh and captured-surface model sources."""

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np

if __package__:
    from .build_tless_demo import write_json
    from .tless_validation import read_capture
else:
    from build_tless_demo import write_json
    from tless_validation import read_capture

from opencv_simple_3d_locate import Locator, Settings, read_mesh, save_recipe, train, train_cad
from opencv_simple_3d_locate.optimization import pose_agreement

SOURCES = ("cad", "reconstructed", "captured")


def model_recipe(source, item, kind):
    """Only the frozen teaching capture's annotation can enter captured training."""
    source = Path(source)
    teaching = read_capture(source, item["positive_scene"], item["training_frame"],
                            item["object_id"], isolated=True)
    inverse = np.linalg.inv(teaching.truth[0])
    settings = Settings(model_viewpoint=tuple(inverse[:3, 3]), execution_workers=4)
    frame = f"tless_object{item['object_id']}"
    if kind == "captured":
        reference = teaching.points.astype(float) @ inverse[:3, :3].T + inverse[:3, 3]
        recipe = train(reference, units="mm", frame=frame, settings=settings)
    elif kind in ("cad", "reconstructed"):
        folder = "models_cad" if kind == "cad" else "models_reconst"
        recipe = train_cad(read_mesh(source / f"{folder}/obj_{item['object_id']:06}.ply"),
                           units="mm", frame=frame, settings=settings)
    else:
        raise ValueError("Choose cad, reconstructed or captured reference sources.")
    return recipe, teaching


def evaluation_cases(source, item):
    if item["training_frame"] in item["near_frames"] + item["far_frames"]:
        raise ValueError("The teaching capture must not enter independent test totals.")
    cases = []
    for group in ("near", "far"):
        for frame in item[group + "_frames"]:
            for isolated in (True, False):
                case = read_capture(source, item["positive_scene"], frame,
                                    item["object_id"], isolated=isolated)
                if not case.truth:
                    raise ValueError("A positive capture has no annotated target.")
                cases.append((group, case))
    for frame in item["negative_frames"]:
        case = read_capture(source, item["negative_scene"], frame, item["object_id"])
        if case.truth:
            raise ValueError("A target-absent capture has an annotated target.")
        cases.append(("absent", case))
    return cases


def metrics(result, capture):
    return pose_agreement([m.pose for m in result.matches], capture.truth,
                          position_tolerance=5, angle_degrees=5, symmetries=capture.symmetries)


def _signature(result):
    return [(m.pose.tolist(), m.score) for m in result.matches]


def make_project(output, name, recipe, capture, expected, title):
    from opencv_dataflow.blocks import builtin_registry
    from opencv_dataflow.cloud_io import write_cloud
    from opencv_dataflow.core import Graph
    from opencv_dataflow.pointclouds import PointCloud

    inputs = output / "inputs"
    inputs.mkdir(exist_ok=True)
    full = f"scene{capture.scene_id:06}_frame{capture.frame_id:06}_full.ply"
    scene_name = capture.name + ".ply" if capture.isolated else full
    for filename, points in ((scene_name, capture.points), (full, capture.depth.points)):
        path = inputs / filename
        if not path.exists():
            write_cloud(path, PointCloud(points, "mm", "camera"), binary=True)
    registry, graph = builtin_registry(), Graph()
    try:
        for block_id, filename in (("scene", scene_name), ("depth", full)):
            block = graph.add(registry.create("input.cloud", block_id=block_id, parameters={
                "path": f"inputs/{filename}", "units": "mm", "frame": "camera",
            }))
            block.position = (0, 0 if block_id == "scene" else 200)
        k = capture.camera.matrix
        camera = graph.add(registry.create("3d.camera", block_id="camera", parameters={
            "fx": float(k[0, 0]), "fy": float(k[1, 1]), "cx": float(k[0, 2]),
            "cy": float(k[1, 2]), "width": capture.camera.size[0],
            "height": capture.camera.size[1], "units": "mm", "frame": "camera",
        }))
        camera.position = (0, 400)
        path = f"models/{name.split('__')[0]}.s3l"
        model_hash = hashlib.sha256((output / path).read_bytes()).hexdigest()
        locate = graph.add(registry.create("3d.simple_locate", block_id="locate", parameters={
            "recipe_path": path, "sha256": model_hash, "render_overlay": False,
        }))
        locate.preview_output, locate.position = "fit", (400, 0)
        for source_id, port, target in (("scene", "cloud", "scene"),
                                        ("depth", "cloud", "depth_scene"),
                                        ("camera", "camera", "camera")):
            graph.connect(source_id, port, "locate", target)
        note = graph.add(registry.create("layout.comment", block_id="scope"))
        note.name, note.node_size, note.position = title, "large", (400, -300)
        note.description = (
            "Actual unseeded localization with fixed settings. Green model support is not "
            "ground-truth correctness. The report distinguishes correct, missed and extra "
            "poses. Known false-positive examples intentionally retain accepted wrong poses. "
            "Evaluation annotations are not connected to any runtime port. "
            "Supplied masks isolate a target only where documented; full scenes use all returns. "
            "T-LESS by Hodan et al., WACV 2017, CC BY 4.0. See source/ATTRIBUTION.md and "
            "https://cmp.felk.cvut.cz/t-less/. Model sources have distinct capability limits."
        )
        graph.save(output / f"project_{name}.json")
        graph.run()
        if locate.error:
            raise RuntimeError(locate.error)
        poses = [np.asarray(p["source_to_target"])
                 for p in locate.last_result.outputs["report"]["matches"]]
        actual = pose_agreement(poses, capture.truth, position_tolerance=5,
                                angle_degrees=5, symmetries=capture.symmetries)
        for key in ("matched", "missed", "extra"):
            if actual[key] != expected[key]:
                raise ValueError("Saved host project differs from recorded comparison.")
        write_cloud(output / f"result_{name}.ply", locate.last_result.outputs["fit"], binary=True)
        return {"project": f"project_{name}.json", "expected": expected,
                "accepted_count": locate.last_result.outputs["count"]}
    finally:
        graph.close()


def build(source, output, *, repeats=2, host_projects=False):
    source, output = Path(source), Path(output)
    if type(repeats) is not int or not 1 <= repeats <= 5:
        raise ValueError("Use 1..5 comparison repeats.")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a fresh comparison output folder.")
    selection = json.loads((source / "reference_selection.json").read_bytes())
    provenance = json.loads((source / "reference_provenance.json").read_bytes())
    for name, expected in provenance["sha256"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Pinned source mismatch: {name}")
    output.mkdir(parents=True, exist_ok=True)
    (output / ".gitattributes").write_bytes(b"*.json -text\n")
    (output / "models").mkdir()
    (output / "reports").mkdir()
    rows, controls, models, projects, gallery = [], [], [], [], set()
    for item in selection["models"]:
        cases = evaluation_cases(source, item)
        for kind in SOURCES:
            name = f"{kind}_object{item['object_id']:06}"
            recipe, teaching = model_recipe(source, item, kind)
            digest = save_recipe(output / f"models/{name}.s3l", recipe)
            locator = Locator(recipe)
            control = locator.locate(teaching.points, units="mm", frame="camera",
                                     camera=teaching.camera, depth_scene=teaching.depth)
            controls.append({"source": kind, "object_id": item["object_id"],
                             "case": teaching.name, "training_replay_only": True,
                             **metrics(control, teaching)})
            models.append({"source": kind, "object_id": item["object_id"],
                           "recipe_sha256": digest, "health": recipe.health,
                           "settings": asdict(recipe.settings),
                           "training_frame": item["training_frame"]})
            for group, capture in cases:
                runs, signatures, first = [], [], None
                for _ in range(repeats):
                    result = locator.locate(capture.points, units="mm", frame="camera",
                                            camera=capture.camera, depth_scene=capture.depth)
                    runs.append({**metrics(result, capture), "result": result.report()})
                    signatures.append(_signature(result))
                    if first is None:
                        first = result
                report_name = f"{name}__{capture.name}.json"
                record = {"source": kind, "object_id": item["object_id"], "group": group,
                          "case": capture.name, "runs": runs,
                          "repeatable_matches": all(s == signatures[0] for s in signatures[1:])}
                write_json(output / "reports" / report_name, record)
                row = {k: record[k] for k in ("source", "object_id", "group", "case")}
                row.update({k: runs[0][k] for k in ("matched", "missed", "extra")})
                row["repeatable_matches"] = record["repeatable_matches"]
                row["report"] = "reports/" + report_name
                rows.append(row)
                outcome = ("false_positive" if group == "absent" and row["extra"] else
                           "correct" if row["matched"] else
                           "missed" if row["missed"] else None)
                key = (kind, outcome)
                if host_projects and outcome and key not in gallery:
                    gallery.add(key)
                    title = f"{kind} reference - {outcome.replace('_', ' ')}"
                    project = make_project(output, name + "__" + outcome, recipe, capture,
                                           {k: row[k] for k in ("matched", "missed", "extra")},
                                           title)
                    projects.append({"source": kind, "outcome": outcome, **project})
                print(json.dumps(row), flush=True)
    summary = {
        "scope": "Fixed model-source development comparison, not production qualification.",
        "sources": {kind: {
            "matched": sum(r["matched"] for r in rows if r["source"] == kind),
            "missed": sum(r["missed"] for r in rows if r["source"] == kind),
            "extra": sum(r["extra"] for r in rows if r["source"] == kind),
            "negative_extra": sum(r["extra"] for r in rows
                                  if r["source"] == kind and r["group"] == "absent"),
        } for kind in SOURCES},
        "cases": rows, "training_controls": controls, "models": models,
        "projects": projects, "repeats": repeats,
        "teaching_captures_excluded_from_test_totals": True,
        "runtime_pose_seeds": False, "gate_settings_tuned": False,
        "gallery_selection": "First correct, missed-positive and target-absent false-positive "
                             "outcome per model-source kind; all cases retained in reports.",
    }
    write_json(output / "summary.json", summary)
    shutil.copyfile(source / "reference_selection.json", output / "selection.json")
    write_json(output / "source_hashes.json", provenance["sha256"])
    write_json(output / "artifact_hashes.json", {
        p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob("*")) if p.is_file()
        and p.name not in ("artifact_hashes.json", ".gitattributes")
    })
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--host-projects", action="store_true")
    args = parser.parse_args()
    build(args.source, args.output, repeats=args.repeats, host_projects=args.host_projects)
