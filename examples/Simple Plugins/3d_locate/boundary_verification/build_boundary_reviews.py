"""Review three frozen runtime poses without repeating localization."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

if __package__:
    from .build_tless_demo import write_json
    from .tless_validation import read_capture
else:
    from build_tless_demo import write_json
    from tless_validation import read_capture

from opencv_simple_3d_locate import load_recipe
from opencv_simple_3d_locate.boundary import BOUNDARY_STATES, inspect_boundary
from opencv_simple_3d_locate.depth import DepthBuffer


def build(source, comparison, output):
    from opencv_simple_3d_locate.qt_boundary_review import boundary_cloud

    from opencv_dataflow.blocks import builtin_registry
    from opencv_dataflow.cloud_io import write_cloud
    from opencv_dataflow.core import Graph

    source, comparison, output = Path(source), Path(comparison), Path(output)
    for path, digest in json.loads((comparison / "artifact_hashes.json").read_bytes()).items():
        if hashlib.sha256((comparison / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Comparison hash mismatch: {path}")
    for path, digest in json.loads((comparison / "source_hashes.json").read_bytes()).items():
        if hashlib.sha256((source / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source hash mismatch: {path}")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a fresh boundary review output folder.")
    output.mkdir(parents=True, exist_ok=True)
    (output / ".gitattributes").write_bytes(b"*.json -text\n")
    cases = (
        ("rejected_false_pose", 8, 2, 3, "full", True,
         "REJECTED false candidate - not a detection"),
        ("retained_false_pose", 8, 2, 3, "full", False,
         "Accepted wrong pose - target absent; remaining limitation"),
        ("retained_correct_pose", 6, 2, 255, "isolated", False,
         "Accepted correct isolated pose - limited development example"),
    )
    records = []
    for name, object_id, scene_id, frame_id, mode, rejected, title in cases:
        report_path = comparison / (
            f"reports/reconstructed_object{object_id:06}__scene{scene_id:06}"
            f"_frame{frame_id:06}_{mode}.json"
        )
        runtime = json.loads(report_path.read_bytes())["runs"][0]["result"]
        row = (next(r for r in runtime["execution"]["candidate_review"]
                    if r["stage"] == "boundary") if rejected else runtime["matches"][0])
        recipe = load_recipe(comparison / f"models/reconstructed_object{object_id:06}.s3l")
        capture = read_capture(source, scene_id, frame_id, object_id, isolated=mode == "isolated")
        pose = np.asarray(row["source_to_target"])
        settings = recipe.settings
        evidence = inspect_boundary(
            recipe.cad.mesh, pose, DepthBuffer(capture.depth.points, capture.camera),
            spacing=settings.sampling_distance * recipe.diameter,
            tolerance=settings.distance_tolerance * recipe.diameter,
            minimum_support=settings.min_coverage, minimum_rays=settings.min_unique_support,
        )
        assert evidence.report["limit_exceeded"] == row["boundary_verification"]["limit_exceeded"]
        folder = output / name
        folder.mkdir()
        write_cloud(folder / "boundary_evidence.ply",
                    boundary_cloud(evidence, scene=capture.depth.points), binary=True)
        np.savez_compressed(
            folder / "boundary_pairs.npz", inside_pixels=evidence.inside_pixels,
            outside_pixels=evidence.outside_pixels, expected_depth=evidence.expected_depth,
            measured_depth=evidence.measured_depth, residual=evidence.residual,
            states=evidence.states,
        )
        record = {**evidence.report, "source_to_target": pose.tolist(),
                  "camera": capture.camera.record(),
                  "state_codes": dict(enumerate(BOUNDARY_STATES)),
                  "review_only": True, "search_executed": False,
                  "runtime_accepted": not rejected, "selection": title,
                  "runtime_record_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                  "source": "T-LESS, Hodan et al., WACV 2017, CC BY 4.0; source/ATTRIBUTION.md"}
        write_json(folder / "report.json", record)
        graph = Graph()
        try:
            registry = builtin_registry()
            cloud = graph.add(registry.create("input.cloud", block_id="boundary", parameters={
                "path": "boundary_evidence.ply", "units": "mm", "frame": "camera",
            }))
            cloud.name, cloud.preview_output = title, "cloud"
            note = graph.add(registry.create("layout.comment", block_id="scope"))
            note.name, note.position = "Cached pose evidence only", (0, -200)
            note.node_size = "large"
            note.description = (
                "No localization runs here. Orange measured surfaces continue beyond the "
                "predicted outline; green depth changes; magenta is unknown. Gray is measured "
                "scene, RGB axes mark the cached pose. Unknown pairs count as possible support. "
                "This experimental check does not fix the overall detection counts. "
                "T-LESS by Hodan et al., WACV 2017, CC BY 4.0. See source/ATTRIBUTION.md "
                "and https://cmp.felk.cvut.cz/t-less/."
            )
            graph.save(folder / "project.json")
        finally:
            graph.close()
        records.append({"name": name, "report": record})
    write_json(output / "summary.json", records)
    write_json(output / "artifact_hashes.json", {
        p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob("*")) if p.is_file()
        and p.name not in ("artifact_hashes.json", ".gitattributes")
    })
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.source, args.comparison, args.output)
