"""Real first-return RGB-D plane measurements with explicit supplied pose fixtures."""

import argparse
import hashlib
import shutil
from pathlib import Path

import cv2
import numpy as np
from host_examples import manifest, project, write_json
from opencv_simple_3d_measure import Measurement, PlaneRegion, train
from opencv_simple_3d_measure.persistence import strict_json
from opencv_simple_3d_measure.specs import rigid

FRAMES = (3, 13, 24)
REVISION = "5fd309a04476a842d93abfb584fba9ee7caecdf1"


def configuration():
    return (
        (PlaneRegion("table", (-200, -180, -40), (-80, -60, -10), normal_hint=(0, 0, 1)),
         PlaneRegion("part_surface", (-50, -25, 8), (50, 25, 30), normal_hint=(0, 0, 1))),
        (Measurement("height", "step_height", "table", "part_surface"),
         Measurement("angle", "plane_angle", "table", "part_surface"),
         Measurement("part_surface_range", "surface_range", "part_surface")),
    )


def capture(source, frame):
    folder = source / "test_primesense/000002"
    camera = strict_json((folder / "scene_camera.json").read_bytes())[str(frame)]
    depth = cv2.imread(str(folder / f"depth/{frame:06}.png"), cv2.IMREAD_UNCHANGED)
    image = cv2.imread(str(folder / f"rgb/{frame:06}.png"))
    if depth is None or image is None or depth.shape != image.shape[:2]:
        raise ValueError("Missing/mismatched pinned RGB-D capture.")
    k = np.asarray(camera["cam_K"]).reshape(3, 3)
    y, x = np.nonzero(depth)
    z = depth[y, x].astype(float) * camera["depth_scale"]
    # Exactly one measured first return per valid depth pixel. No masks/hidden geometry.
    xyz = np.c_[(x - k[0, 2]) * z / k[0, 0], (y - k[1, 2]) * z / k[1, 1], z]
    entries = strict_json((folder / "scene_gt.json").read_bytes())[str(frame)]
    [annotation] = [entry for entry in entries if entry["obj_id"] == 6]
    fixture = np.eye(4)
    fixture[:3, :3] = np.asarray(annotation["cam_R_m2c"]).reshape(3, 3)
    fixture[:3, 3] = annotation["cam_t_m2c"]
    return xyz, rigid(fixture)


def build(source, output):
    source, output = Path(source), Path(output)
    provenance = source / "reference_provenance.json"
    if not provenance.exists():
        provenance = source / "provenance.json"
    pinned = strict_json(provenance.read_bytes())
    if pinned["revision"] != REVISION or pinned["license"] != "CC-BY-4.0":
        raise ValueError("Expected the documented pinned T-LESS CC BY 4.0 corpus.")
    selected = ["test_primesense/000002/scene_camera.json", "test_primesense/000002/scene_gt.json"]
    selected += [f"test_primesense/000002/{kind}/{frame:06}.png"
                 for frame in FRAMES for kind in ("depth", "rgb")]
    for relative in selected:
        actual = hashlib.sha256((source / relative).read_bytes()).hexdigest()
        if actual != pinned["sha256"][relative]:
            raise ValueError(f"Pinned source hash mismatch: {relative}")
    output.mkdir(parents=True, exist_ok=False)
    for relative in selected:
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    (output / "source/.gitattributes").write_bytes(b"* -text\n")
    write_json(output / "source/provenance.json", {
        "dataset": "T-LESS", "revision": REVISION, "license": "CC-BY-4.0",
        "source": "https://cmp.felk.cvut.cz/t-less/",
        "mirror": "https://huggingface.co/datasets/bop-benchmark/tless",
        "test_archive_sha256": pinned["test_archive_sha256"],
        "verification": "Selected members SHA-256 verified; full upstream test ZIP not downloaded.",
        "sha256": {name: pinned["sha256"][name] for name in selected}})
    xyz, fixture = capture(source, FRAMES[0])
    reference = (xyz - fixture[:3, 3]) @ fixture[:3, :3]
    recipe = train(reference, *configuration(), units="mm", frame="object6")
    write_json(output / "configuration.json", recipe.configuration())
    np.savez_compressed(output / "reference.npz", points=reference, units="mm", frame="object6")
    reports = {}
    for frame in FRAMES:
        xyz, fixture = capture(source, frame)
        reports[str(frame)] = project(
            output / f"frame{frame:06}", recipe, xyz, scene_frame="camera", fixture=fixture,
            description="Real T-LESS first returns; supplied annotation fixture. "
                        "No detection or independent dimensional accuracy claim; "
                        + ("training-reference replay control." if frame == 3 else
                           "independent capture with frozen reference recipe."))
        print(frame, [(m["name"], m["value"], m["status"])
                      for m in reports[str(frame)]["measurements"]])
    write_json(output / "comparison.json", reports)
    (output / "ATTRIBUTION.md").write_text(
        "# T-LESS measurement workflow data\n\n"
        "T. Hodaň, P. Haluza, Š. Obdržálek, J. Matas, M. Lourakis and X. Zabulis. "
        "T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects. WACV 2017.\n\n"
        "[Publisher, paper and license](https://cmp.felk.cvut.cz/t-less/) · "
        "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · "
        "[Official mirror](https://huggingface.co/datasets/bop-benchmark/tless).\n\n"
        f"Pinned mirror revision `{REVISION}`. Original scene 2 RGB/depth images "
        "and calibration/pose metadata for frames 3, 13 and 24 are preserved under `source/`; "
        "SHA-256 provenance is included. The full upstream ZIP was not downloaded.\n\n"
        "Derived XYZ, trained reference recipes, result clouds and numerical reports "
        "backproject measured first returns. No object mask, CAD or hidden surface points "
        "are used. Published annotations supply the measurement fixture in all three graphs; "
        "they are not detected poses. Frame 3 is a training replay control. Frames 13/24 "
        "are independent captures of the same scene. Rounded rotations are projected "
        "onto a proper rotation. The same trained regions/bands and 5-degree parallel guard "
        "are reused. No acceptance limits or independent gauge truth are supplied.\n\n"
        "Raw and derived dataset material retain CC BY 4.0. Preserve this attribution on "
        "redistribution. No endorsement is implied. Builder code uses MIT.\n", encoding="utf-8")
    manifest(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Existing pinned T-LESS reference-source folder")
    parser.add_argument("output", type=Path, help="Fresh output folder")
    args = parser.parse_args()
    build(args.source, args.output)
