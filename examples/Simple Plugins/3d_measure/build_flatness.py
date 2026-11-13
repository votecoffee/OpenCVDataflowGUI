"""Known height-sample zones and real first-return T-LESS flatness evidence."""

import argparse
import hashlib
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from build_tless import FRAMES, REVISION, capture
from flatness_samples import flatness_cloud
from host_examples import manifest, project, write_json
from opencv_simple_3d_measure import Measurement, PlaneRegion, train
from opencv_simple_3d_measure.persistence import strict_json

from opencv_dataflow.cloud_io import read_cloud
from opencv_dataflow.pointclouds import render_cloud


def preview(folder):
    cloud = read_cloud(folder / "result.ply")
    image = render_cloud(cloud, width=900, height=650, yaw=25, pitch=-55, point_size=2)
    cv2.imwrite(str(folder / "result.png"), image)


def build(output, source=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    measurements = (Measurement("flatness", "minimum_zone_flatness", "patch", upper=.45),
                    Measurement("datum_range", "surface_range", "patch"))
    reference = flatness_cloud()
    recipe = train(reference, (PlaneRegion("patch", (-11, -11, -1), (11, 11, 1)),),
                   measurements, units="mm", frame="part")
    generated = output / "generated"
    generated.mkdir()
    write_json(generated / "configuration.json", recipe.configuration())
    np.savez_compressed(generated / "reference.npz", points=reference, units="mm", frame="part")
    for name in ("good", "deep_feature", "geometry_budget", "missing_surface"):
        scene = flatness_cloud(seed=12, raised=name == "deep_feature")
        taught = recipe
        if name == "geometry_budget":
            taught = replace(recipe, measurements=(replace(measurements[0], zone_max_vertices=4),
                                                    measurements[1]))
        if name == "missing_surface":
            scene = scene[:10]
        folder = generated / name
        report = project(folder, taught, scene,
                         description="Known single-visible height samples; all selected extremes "
                                     "count. No datum-range substitution for unavailable flatness.")
        preview(folder)
        print(name, [(m["name"], m["value"], m["status"], m["reason"])
                     for m in report["measurements"]])
    if source is not None:
        source = Path(source)
        provenance_path = source / "provenance.json"
        if not provenance_path.exists():
            provenance_path = source / "reference_provenance.json"
        provenance = strict_json(provenance_path.read_bytes())
        if provenance["license"] != "CC-BY-4.0" or provenance["revision"] != REVISION:
            raise ValueError("Expected the pinned T-LESS CC BY 4.0 corpus")
        names = [f"test_primesense/000002/{kind}/{frame:06}.png"
                 for frame in FRAMES for kind in ("rgb", "depth")]
        names += ["test_primesense/000002/scene_camera.json",
                  "test_primesense/000002/scene_gt.json"]
        for name in names:
            digest = hashlib.sha256((source / name).read_bytes()).hexdigest()
            if digest != provenance["sha256"][name]:
                raise ValueError(f"Pinned source mismatch: {name}")
        xyz, pose = capture(source, 3)
        reference = (xyz - pose[:3, 3]) @ pose[:3, :3]
        planes = (PlaneRegion("table", (-200, -180, -40), (-80, -60, -10), normal_hint=(0, 0, 1)),
                  PlaneRegion("part_surface", (-50, -25, 8), (50, 25, 30), normal_hint=(0, 0, 1)))
        measurements = tuple(m for p in planes for m in (
            Measurement(p.name + "_zone", "minimum_zone_flatness", p.name),
            Measurement(p.name + "_range", "surface_range", p.name)))
        taught = train(reference, planes, measurements, units="mm", frame="object6")
        real = output / "real_tless"
        real.mkdir()
        write_json(real / "configuration.json", taught.configuration())
        for frame in FRAMES:
            xyz, pose = capture(source, frame)
            folder = real / f"frame{frame:06}"
            report = project(folder, taught, xyz, scene_frame="camera", fixture=pose,
                             description="Real T-LESS all-selected finite-point minimum zones; "
                                         "supplied pose fixture, no independent gauge truth.")
            preview(folder)
            print(frame, [(m["name"], m["value"], m["status"], m["reason"])
                          for m in report["measurements"]])
        write_json(real / "source_provenance.json", {**provenance,
                   "sha256": {name: provenance["sha256"][name] for name in names},
                   "raw_source_base": "../../real_tless/source",
                   "derived": "All measured first returns inside fixed ROIs; supplied fixture."})
    manifest(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, help="Pinned real T-LESS source folder")
    args = parser.parse_args()
    build(args.output, args.source)
