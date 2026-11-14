"""Fixed suction controls and real first-return T-LESS replay, with no per-frame tuning."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from opencv_dataflow.assistant_api import ProjectSession
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.pointclouds import render_cloud
from samples import control

from opencv_simple_bin_picking import Camera, Recipe, SuctionSettings
from opencv_simple_bin_picking.dataflow import register
from opencv_simple_bin_picking.persistence import save_recipe, strict_json
from opencv_simple_bin_picking.specs import rigid

REVISION = "5fd309a04476a842d93abfb584fba9ee7caecdf1"


def write_json(path, value):
    Path(path).write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def project(folder, recipe, depth, *, transform=None, image=None):
    folder = Path(folder)
    assets = folder / "assets"
    assets.mkdir(parents=True, exist_ok=False)
    digest = save_recipe(assets / "recipe.sbp", recipe)
    # These datasets/control arrays have integral stored depth. Preserve physical scale in recipe.
    if not np.isfinite(depth).all() or np.any(depth != np.rint(depth)) or np.any(depth > 65535):
        raise ValueError("Example PNGs require integral uint16 depth; use raw NPY for other inputs")
    cv2.imwrite(str(assets / "depth.png"), depth.astype(np.uint16))
    np.save(assets / "depth.npy", depth, allow_pickle=False)
    registry = builtin_registry()
    register(registry)
    session = ProjectSession(registry, directory=folder)
    graph = session.graph
    try:
        depth_block = graph.add(registry.create("input.file", block_id="depth", parameters={
            "path": "assets/depth.png", "path_base": "project", "pixel_depth": "unchanged",
            "color_format": "unchanged", "filename_ticket": False}))
        depth_block.position = [0, 0]
        k = recipe.camera.matrix
        camera = graph.add(registry.create("3d.camera", block_id="camera", parameters={
            "fx": k[0][0], "fy": k[1][1], "cx": k[0][2], "cy": k[1][2],
            "width": recipe.camera.image_size[0], "height": recipe.camera.image_size[1],
            "units": recipe.units, "frame": recipe.camera.frame}))
        camera.position = [0, 240]
        picker = graph.add(registry.create("3d.simple_bin_picking", block_id="picker", parameters={
            "recipe_path": "assets/recipe.sbp", "sha256": digest}))
        picker.position = [500, 0]
        graph.connect("depth", "image", "picker", "depth")
        graph.connect("camera", "camera", "picker", "camera")
        transform = recipe.bin_to_camera if transform is None else transform
        pose = graph.add(registry.create("3d.pose", block_id="bin_pose", parameters={
            "rvec": cv2.Rodrigues(transform[:3, :3])[0].ravel().tolist(),
            "translation": transform[:3, 3].tolist(), "source_frame": recipe.frame,
            "target_frame": recipe.camera.frame, "units": recipe.units}))
        pose.position = [0, 480]
        pose.name = "Supplied bin/camera calibration (not object detection)"
        graph.connect("bin_pose", "transform", "picker", "bin_pose")
        if image is not None:
            cv2.imwrite(str(assets / "rgb.png"), image[..., ::-1])
            rgb = graph.add(registry.create("input.file", block_id="rgb", parameters={
                "path": "assets/rgb.png", "path_base": "project", "color_format": "rgb",
                "filename_ticket": False}))
            rgb.position = [0, 720]
            graph.connect("rgb", "image", "picker", "image")
        if not session.inspect()["validation"]["valid"]:
            raise ValueError("Example graph did not pass static validation")
        session.save(folder / "project.json")
        session.run()
        if picker.state == "error":
            raise RuntimeError(picker.error)
        report = picker.last_result.outputs["report"]
        write_json(folder / "expected.json", report)
        cloud = picker.last_result.outputs["graphics"]
        write_cloud(folder / "result.ply", cloud, binary=True)
        cv2.imwrite(str(folder / "result.png"), render_cloud(cloud, width=1000, height=740,
                    yaw=25, pitch=-35, point_size=2, limit=40000))
        return report
    finally:
        session.close()


def shifted(image, dx, dy):
    output = np.zeros_like(image)
    h, w = image.shape[:2]
    x0, x1 = max(0, dx), min(w, w + dx)
    y0, y1 = max(0, dy), min(h, h + dy)
    output[y0:y1, x0:x1] = image[y0-dy:y1-dy, x0-dx:x1-dx]
    return output


def build(output, source=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    recipe, _ = control()
    for name in ("good", "empty", "missing", "holes", "narrow_bin"):
        _, depth = control("good" if name == "narrow_bin" else name)
        fixed = (replace(recipe, minimum=(-5, -20, 0), maximum=(5, 20, 100))
                 if name == "narrow_bin" else recipe)
        report = project(output / "generated" / name, fixed, depth)
        print(name, report["eligible_picks"], report["evaluated_candidates"])
    write_json(output / "generated/configuration.json", recipe.configuration())
    np.save(output / "generated/reference.npy", recipe.reference_depth, allow_pickle=False)
    if source is not None:
        source = Path(source)
        provenance = strict_json((source / "provenance.json").read_bytes())
        if provenance["revision"] != REVISION or provenance["license"] != "CC-BY-4.0":
            raise ValueError("Use the pinned, attributed T-LESS corpus")
        names = [f"test_primesense/000002/{kind}/{frame:06}.png"
                 for frame in (3, 13, 24) for kind in ("rgb", "depth")]
        names += ["test_primesense/000002/scene_camera.json"]
        for name in names:
            raw = (source / name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != provenance["sha256"][name]:
                raise ValueError(f"Pinned source mismatch: {name}")
            target = output / "real_tless/source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        raw = output / "real_tless/source"
        (raw / ".gitattributes").write_bytes(b"* -text\n")
        write_json(raw / "provenance.json", {**provenance,
            "sha256": {name: provenance["sha256"][name] for name in names}})
        folder = source / "test_primesense/000002"
        cameras = strict_json((folder / "scene_camera.json").read_bytes())
        k0 = np.array(cameras["3"]["cam_K"]).reshape(3, 3)
        taught = None
        shifts = {}
        for frame in (3, 13, 24):
            c = cameras[str(frame)]
            k = np.array(c["cam_K"]).reshape(3, 3)
            delta = k0[:2, 2] - k[:2, 2]
            if not np.allclose(delta, np.rint(delta), atol=1e-6) or not np.array_equal(
                    k[[0, 1], [0, 1]], k0[[0, 1], [0, 1]]):
                raise ValueError("This builder only normalizes integer principal-point crops")
            dx, dy = np.rint(delta).astype(int)
            depth = shifted(cv2.imread(str(folder / f"depth/{frame:06}.png"),
                                        cv2.IMREAD_UNCHANGED), dx, dy)
            image = shifted(cv2.imread(str(folder / f"rgb/{frame:06}.png"))[..., ::-1], dx, dy)
            transform = np.eye(4)
            transform[:3, :3] = np.array(c["cam_R_w2c"]).reshape(3, 3)
            transform[:3, 3] = c["cam_t_w2c"]
            transform = rigid(transform)
            if taught is None:
                taught = Recipe(Camera(k0, (depth.shape[1], depth.shape[0])), depth, transform,
                    (-200, -200, 0), (200, 200, 150), frame="bop_world",
                    depth_scale=c["depth_scale"], settings=SuctionSettings(), reference_rgb=image)
            shifts[str(frame)] = [int(dx), int(dy)]
            report = project(output / "real_tless" / f"frame{frame:06}", taught, depth,
                             transform=transform, image=image)
            print("TLESS", frame, report["eligible_picks"], report["evaluated_candidates"])
        write_json(output / "real_tless/normalization.json", {
            "method": "integer_pixel_translation_to_frame3_intrinsics_zero_padding_unknown",
            "pixel_shifts_xy": shifts, "fixture": "supplied_camera_world_extrinsics",
            "reference": 3, "held_out": [13, 24], "same_recipe_and_thresholds": True,
            "grasp_labels": "not_provided_no_physical_pick_success_claim"})
    hashes = {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(output.rglob("*")) if p.is_file() and p.name != "sha256.json"}
    write_json(output / "sha256.json", hashes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    build(args.output, args.source)
