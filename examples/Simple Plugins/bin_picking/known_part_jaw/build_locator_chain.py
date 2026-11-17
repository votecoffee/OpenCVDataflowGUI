"""Real first-return locator → explicit instances → jaw chain; no test poses supplied."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from opencv_dataflow.assistant_api import ProjectSession
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.cloud_io import write_cloud
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.plugins import load_plugin
from opencv_dataflow.pointclouds import render_cloud


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def build(root, locator_examples, raw_source, output):
    output.mkdir(parents=True, exist_ok=False)
    registry = builtin_registry()
    load_plugin(root / "plugin.py", registry)
    from opencv_simple_3d_locate import load_recipe as load_locator

    from opencv_simple_bin_picking.jaw_persistence import load_jaw_recipe, save_jaw_recipe
    from opencv_simple_bin_picking.specs import Camera, rigid

    # Existing published recipes are used unchanged; no per-case execution tuning.
    frame = 255
    source_name = "scene000002_frame000255"
    canonical = load_jaw_recipe(root / "tless/frame000013/assets/trained.sbpj")
    camera_meta = json.loads(
        (raw_source / "test_primesense/000002/scene_camera.json").read_bytes()
    )[str(frame)]
    provenance = json.loads((raw_source / "provenance.json").read_bytes())
    assert provenance["license"] == "CC-BY-4.0"
    assert provenance["revision"] == "5fd309a04476a842d93abfb584fba9ee7caecdf1"
    selected = [f"test_primesense/000002/{kind}/{frame:06}.png" for kind in ("rgb", "depth")]
    selected += [
        f"test_primesense/000002/mask_visib/{frame:06}_000001.png",
        "test_primesense/000002/scene_camera.json",
    ]
    for member in selected:
        raw_member = (raw_source / member).read_bytes()
        assert hashlib.sha256(raw_member).hexdigest() == provenance["sha256"][member]
        destination = output / "source" / member
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw_member)
    write_json(
        output / "source/provenance.json",
        dict(
            dataset="T-LESS",
            license="CC-BY-4.0",
            revision=provenance["revision"],
            source="https://cmp.felk.cvut.cz/t-less/",
            sha256={name: provenance["sha256"][name] for name in selected},
            scope="RGB-D/camera/visible mask only; object pose labels are not runtime inputs",
        ),
    )
    depth_path = raw_source / f"test_primesense/000002/depth/{frame:06}.png"
    raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    assert raw is not None
    depth = (raw.astype(float) * camera_meta["depth_scale"] * 0.001).astype(np.float32)
    camera = Camera(np.array(camera_meta["cam_K"]).reshape(3, 3), depth.shape[1::-1])
    bin_pose = np.eye(4)
    bin_pose[:3, :3] = np.array(camera_meta["cam_R_w2c"]).reshape(3, 3)
    bin_pose[:3, 3] = np.array(camera_meta["cam_t_w2c"]) * 0.001
    base = replace(
        canonical.bin, camera=camera, reference_depth=depth, bin_to_camera=rigid(bin_pose)
    )
    receipts = []
    for source_kind in ("reconstructed", "cad"):
        folder = output / source_kind
        assets = folder / "assets"
        assets.mkdir(parents=True)
        model_path = locator_examples / f"models/{source_kind}_object000006.s3l"
        locator_recipe = load_locator(model_path)
        assert locator_recipe.units == "mm" and locator_recipe.frame == "tless_object6"
        shutil.copy2(model_path, assets / "locator.s3l")
        copied = {}
        for kind in ("isolated", "full"):
            path = locator_examples / f"inputs/{source_name}_{kind}.ply"
            shutil.copy2(path, assets / f"{kind}.ply")
            copied[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        recipe = replace(
            canonical, bin=base, part=replace(canonical.part, frame=locator_recipe.frame)
        )
        recipe_hash = save_jaw_recipe(assets / "jaw.sbpj", recipe)
        model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        np.save(assets / "depth.npy", depth, allow_pickle=False)
        identity = f"tless_scene2_frame{frame}"
        write_json(
            assets / "records.json",
            dict(
                capture_id=identity,
                part_instances=[],
                camera=base.camera.record("m"),
                scope="Recorded acquisition identity; no annotation object pose supplied",
            ),
        )
        with ProjectSession(registry, directory=folder) as session:
            graph = session.graph

            def add(kind, identifier, parameters, position, name=None):
                block = graph.add(registry.create(kind, block_id=identifier, parameters=parameters))
                block.position = position
                if name:
                    block.name = name
                return block

            add(
                "input.simple_jaw_example",
                "capture",
                dict(
                    depth_path="assets/depth.npy",
                    records_path="assets/records.json",
                    locator_scene_path="assets/isolated.ply",
                    locator_depth_path="assets/full.ply",
                ),
                [0, 550],
                "Recorded metric depth / explicit acquisition ID",
            )
            k = camera.matrix
            add(
                "3d.camera",
                "locator_camera",
                dict(
                    fx=k[0][0],
                    fy=k[1][1],
                    cx=k[0][2],
                    cy=k[1][2],
                    width=depth.shape[1],
                    height=depth.shape[0],
                    units="mm",
                    frame="camera",
                ),
                [0, 400],
            )
            locator = add(
                "3d.simple_locate",
                "locate",
                dict(recipe_path="assets/locator.s3l", sha256=model_hash, render_overlay=False),
                [420, 0],
            )
            bridge = add(
                "3d.simple_part_instances",
                "instances",
                dict(
                    part_frame=locator_recipe.frame,
                    camera_frame="camera",
                    instance_prefix="tless6",
                    allow_legacy_capture=False,
                ),
                [850, 100],
            )
            picker = add(
                "3d.simple_bin_picking_jaw",
                "jaw",
                dict(recipe_path="assets/jaw.sbpj", sha256=recipe_hash),
                [1250, 100],
            )
            for source, port, target, inlet in (
                ("capture", "locator_scene", "locate", "scene"),
                ("capture", "locator_depth", "locate", "depth_scene"),
                ("locator_camera", "camera", "locate", "camera"),
                ("locate", "report", "instances", "locator_report"),
                ("capture", "capture_id", "locate", "capture_id"),
                ("capture", "capture_id", "instances", "capture_id"),
                ("instances", "part_instances", "jaw", "part_instances"),
                ("capture", "capture_id", "jaw", "capture_id"),
                ("capture", "camera", "jaw", "camera"),
                ("capture", "depth", "jaw", "depth"),
            ):
                graph.connect(source, port, target, inlet)
            assert session.inspect()["validation"]["valid"]
            session.save(folder / "project.json")
            session.run()
            assert locator.last_result is not None, locator.error
            locate_report = locator.last_result.outputs["report"]
            write_json(folder / "locator-report.json", locate_report)
            if bridge.last_result is None or picker.last_result is None:
                raise RuntimeError(
                    f"Chain did not publish: {bridge.state}:{bridge.error}; "
                    f"{picker.state}:{picker.error}"
                )
            adapted = bridge.last_result.outputs["report"]
            report = picker.last_result.outputs["report"]
            assert adapted["emitted_instances"] == len(locate_report["matches"])
            assert not report["robot_execution"]
            if not report["eligible_grasps"]:
                assert picker.last_result.outputs["tcp"] is NO_VALUE
            write_json(folder / "adapter-report.json", adapted)
            write_json(folder / "expected.json", report)
            cloud = picker.last_result.outputs["graphics"]
            write_cloud(folder / "result.ply", cloud, binary=True)
            write_json(folder / "graphics.json", cloud.metadata)
            cv2.imwrite(
                str(folder / "result.png"), render_cloud(cloud, 1000, 740, yaw=155, pitch=-35)
            )
            receipts.append(
                dict(
                    case=source_kind,
                    locator_instances=len(locate_report["matches"]),
                    converted_instances=adapted["emitted_instances"],
                    eligible_grasps=report["eligible_grasps"],
                    jaw_reasons=[c["reasons"] for c in report["candidates"]],
                    ground_truth_object_pose_supplied=False,
                    supplied_visible_instance_mask=True,
                    first_return_measured_depth=True,
                    locator_model_sha256=model_hash,
                    source_cloud_sha256=copied,
                    depth_png_sha256=hashlib.sha256(depth_path.read_bytes()).hexdigest(),
                    tested_locator_version=importlib.metadata.version("opencv-simple-3d-locate"),
                    capture_binding=adapted["capture_binding"],
                    upstream_capture_binding=adapted["upstream_capture_binding"],
                )
            )
    write_json(output / "checks.json", receipts)
    print(json.dumps(receipts, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "locator_examples", "raw_source", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    build(
        *(
            getattr(args, name).resolve()
            for name in ("root", "locator_examples", "raw_source", "output")
        )
    )
