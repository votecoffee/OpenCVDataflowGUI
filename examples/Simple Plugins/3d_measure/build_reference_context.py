"""Retained RGB/pixel/calibration evidence on unchanged real T-LESS measurement regions."""

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np
from build_tless import REVISION, capture, configuration
from host_examples import manifest, project, write_json
from opencv_simple_3d_measure import Measurer, ReferenceEvidence, train
from opencv_simple_3d_measure.graphics import result_cloud
from opencv_simple_3d_measure.persistence import strict_json

from opencv_dataflow.cloud_io import read_cloud
from opencv_dataflow.pointclouds import PointCloud, render_cloud


def build(source, output):
    source, output = Path(source), Path(output)
    provenance = strict_json((source / "provenance.json").read_bytes())
    if provenance["license"] != "CC-BY-4.0" or provenance["revision"] != REVISION:
        raise ValueError("Use the pinned T-LESS corpus")
    selected = [f"test_primesense/000002/{kind}/{frame:06}.png"
                for frame in (3, 13) for kind in ("rgb", "depth")]
    selected += ["test_primesense/000002/scene_camera.json", "test_primesense/000002/scene_gt.json"]
    for name in selected:
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != provenance["sha256"][name]:
            raise ValueError(f"Pinned source mismatch: {name}")
    xyz, fixture = capture(source, 3)
    reference = (xyz - fixture[:3, 3]) @ fixture[:3, :3]
    folder = source / "test_primesense/000002"
    depth = cv2.imread(str(folder / "depth/000003.png"), cv2.IMREAD_UNCHANGED)
    image = cv2.imread(str(folder / "rgb/000003.png"))
    y, x = np.nonzero(depth)
    colors, pixels = image[y, x, ::-1], np.c_[x, y]
    camera = strict_json((folder / "scene_camera.json").read_bytes())["3"]
    evidence = ReferenceEvidence(len(reference), colors=colors, pixels=pixels,
        source_size=(image.shape[1], image.shape[0]), metadata={
            "dataset": "T-LESS", "revision": REVISION, "license": "CC-BY-4.0",
            "source": "https://cmp.felk.cvut.cz/t-less/", "reference_frame_id": 3,
            "camera": camera, "source_frame": "camera", "reference_frame": "object6",
            "reference_to_source": fixture.tolist(), "cloud_units": "mm",
            "pixel_convention": "zero_based_integer_image_centers",
            "source_hashes": {name: provenance["sha256"][name] for name in selected},
            "qualification": "supplied_fixture_and_metadata_not_independent_gauge_truth"})
    recipe = train(reference, *configuration(), units="mm", frame="object6",
                   reference_evidence=evidence)
    scene, fixture = capture(source, 13)
    depth = cv2.imread(str(folder / "depth/000013.png"), cv2.IMREAD_UNCHANGED)
    image = cv2.imread(str(folder / "rgb/000013.png"))
    y, x = np.nonzero(depth)
    scene = PointCloud(scene, "mm", "camera", colors=image[y, x, ::-1])
    report = project(output, recipe, scene, scene_frame="camera", fixture=fixture,
        description="Real T-LESS frame 13; retained frame-3 source RGB, pixel rows, calibration "
                    "and supplied fixture evidence. No independent dimensional accuracy claim.")
    write_json(output / "configuration.json", recipe.configuration())
    write_json(output / "reference_context.json", {**evidence.summary(),
                                                 "metadata": evidence.metadata})
    views = {"reference_source.png": PointCloud(reference, "mm", "object6", colors=colors),
             "result.png": read_cloud(output / "result.ply")}
    fit = result_cloud(Measurer(recipe).measure(reference, units="mm", frame="object6"))
    views["reference_fit.png"] = PointCloud(fit.points, fit.units, fit.frame, colors=fit.colors)
    for name, cloud in views.items():
        cv2.imwrite(str(output / name), render_cloud(cloud, width=900, height=650,
            yaw=25, pitch=-55, point_size=1 if name == "reference_source.png" else 2,
            limit=len(cloud.points) if name == "reference_source.png" else 20000))
    write_json(output / "source_provenance.json", {**provenance,
        "raw_source_base": "../real_tless/source",
        "sha256": {name: provenance["sha256"][name] for name in selected}})
    manifest(output)
    print([(m["name"], m["value"], m["status"]) for m in report["measurements"]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.source, args.output)
