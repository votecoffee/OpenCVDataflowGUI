"""Generated line controls and measured T-LESS section-axis examples, with native graphics."""

import argparse
import hashlib
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from build_tless import FRAMES, REVISION, capture
from host_examples import manifest, project, write_json
from line_samples import line_cloud, line_configuration
from opencv_simple_3d_measure import LineRegion, Measurement, train
from opencv_simple_3d_measure.persistence import strict_json

from opencv_dataflow.cloud_io import read_cloud
from opencv_dataflow.pointclouds import render_cloud


def preview(folder):
    cloud = read_cloud(folder / "result.ply")
    image = render_cloud(cloud, width=900, height=650, yaw=25, pitch=-20, point_size=2)
    cv2.imwrite(str(folder / "result.png"), image)


def build(output, source=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    lines, measurements = line_configuration()
    measurements = tuple(replace(m, lower=19.5, upper=20.5) if m.kind == "line_span" else m
                         for m in measurements)
    recipe = train(line_cloud(), (), measurements, lines=lines, units="mm", frame="part")
    generated = output / "generated"
    generated.mkdir()
    write_json(generated / "configuration.json", recipe.configuration())
    np.savez_compressed(generated / "reference.npz", points=recipe.reference, units="mm",
                        frame="part")
    for name, options in (("good", {}), ("wrong_gap", {"gap": 4.3}),
                          ("missing_target", {"missing": True}),
                          ("nonparallel", {"tilt": 6}), ("shortened", {"shortened": True})):
        folder = generated / name
        report = project(folder, recipe, line_cloud(seed=12, **options),
                         description="Generated visible line traces; fixed 4 mm gap / 20 mm span. "
                                     "Independent noise and samples. No hidden surface points.")
        preview(folder)
        print(name, [(m["name"], m["value"], m["status"]) for m in report["measurements"]])
    if source is not None:
        source = Path(source)
        provenance_path = source / "provenance.json"
        if not provenance_path.exists():
            provenance_path = source / "reference_provenance.json"
        provenance = strict_json(provenance_path.read_bytes())
        if provenance["license"] != "CC-BY-4.0" or provenance["revision"] != REVISION:
            raise ValueError("Expected the pinned T-LESS CC BY 4.0 corpus")
        selected = [f"test_primesense/000002/{kind}/{frame:06}.png"
                    for frame in FRAMES for kind in ("depth", "rgb")]
        selected += ["test_primesense/000002/scene_camera.json",
                     "test_primesense/000002/scene_gt.json"]
        for name in selected:
            digest = hashlib.sha256((source / name).read_bytes()).hexdigest()
            if digest != provenance["sha256"][name]:
                raise ValueError(f"Pinned raw source mismatch: {name}")
        real = output / "real_sections"
        real.mkdir()
        xyz, pose = capture(source, 3)
        reference = (xyz - pose[:3, 3]) @ pose[:3, :3]
        # These are selected measured table sections, not automatically found physical edges.
        strips = (LineRegion("section_a", (-200, -151, -40), (-80, -149, -10)),
                  LineRegion("section_b", (-200, -101, -40), (-80, -99, -10)))
        measurements = (Measurement("section_gap", "line_separation", "section_a", "section_b"),
                        Measurement("section_angle", "line_angle", "section_a", "section_b"),
                        Measurement("observed_span", "line_span", "section_a"))
        taught = train(reference, (), measurements, lines=strips, units="mm", frame="object6")
        write_json(real / "configuration.json", taught.configuration())
        for frame in FRAMES:
            xyz, pose = capture(source, frame)
            folder = real / f"frame{frame:06}"
            report = project(folder, taught, xyz, scene_frame="camera", fixture=pose,
                             description="Real T-LESS table sections; supplied pose fixture. "
                                         "Observed section axes, not detected physical edges or "
                                         "independent dimensional qualification.")
            preview(folder)
            print(frame, [(m["name"], m["value"], m["status"])
                          for m in report["measurements"]])
        write_json(real / "source_provenance.json", {**provenance,
                   "sha256": {name: provenance["sha256"][name] for name in selected},
                   "raw_source_base": "../../real_tless/source",
                   "derived": "First returns in fixed section ROIs; supplied pose fixtures."})
    manifest(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, help="Pinned T-LESS raw source for real sections")
    args = parser.parse_args()
    build(args.output, args.source)
