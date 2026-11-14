"""Run real supplied-instance-mask T-LESS captures; pose labels enter evaluation only."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_simple_foundation_pose.persistence import Recipe, save_recipe
from opencv_simple_foundation_pose.register import Estimator
from opencv_simple_foundation_pose.review import export_review
from opencv_simple_foundation_pose.runtime import OnnxModels
from opencv_simple_foundation_pose.specs import Frame, Mesh, Settings, rigid


def read_case(source, scene, frame, target, mask_object=None):
    source = Path(source)
    folder = source / f"test_primesense/{scene:06}"
    cameras = json.loads((folder / "scene_camera.json").read_bytes())
    entries = json.loads((folder / "scene_gt.json").read_bytes())[str(frame)]
    camera = cameras[str(frame)]
    mask_object = target if mask_object is None else mask_object
    masks = [i for i, value in enumerate(entries) if value["obj_id"] == mask_object]
    if len(masks) != 1:
        raise ValueError("This pilot selects exactly one explicitly supplied visible instance")
    image = cv2.imread(str(folder / f"rgb/{frame:06}.png"))[..., ::-1]
    depth = cv2.imread(str(folder / f"depth/{frame:06}.png"), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(folder / f"mask_visib/{frame:06}_{masks[0]:06}.png"), 0)
    observation = Frame(
        image,
        depth.astype(np.float32) * camera["depth_scale"] / 1000,
        np.array(camera["cam_K"]).reshape(3, 3),
        mask,
    )
    # Returned independently; caller never sends these evaluation poses to Estimator.register.
    truth = []
    for entry in entries:
        if entry["obj_id"] == target:
            pose = np.eye(4)
            pose[:3, :3] = np.array(entry["cam_R_m2c"]).reshape(3, 3)
            pose[:3, 3] = np.array(entry["cam_t_m2c"]) / 1000
            truth.append(rigid(pose))
    return observation, truth


def evaluate(result, truth, symmetries):
    candidate = result.best
    if candidate is None:
        return {"accepted": False, "missed": bool(truth), "extra": False}
    if not truth:
        return {"accepted": True, "missed": False, "extra": True}
    pose = np.array(candidate["source_to_target"])
    errors = []
    for expected in truth:
        for symmetry in symmetries:
            target = expected @ symmetry
            angle = np.rad2deg(
                np.arccos(np.clip((np.trace(pose[:3, :3] @ target[:3, :3].T) - 1) / 2, -1, 1))
            )
            position = np.linalg.norm(pose[:3, 3] - target[:3, 3]) * 1000
            errors.append((float(angle), float(position)))
    angle, position = min(errors, key=lambda x: max(x[0] / 5, x[1] / 5))
    correct = angle <= 5 and position <= 5
    return {
        "accepted": True,
        "correct_5mm_5deg": correct,
        "missed": not correct,
        "extra": not correct,
        "rotation_error_degrees": angle,
        "translation_error_mm": position,
    }


def run(source, weights, output, cases):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = source / "reference_provenance.json"
    if not manifest.exists():
        manifest = source / "provenance.json"
    provenance = json.loads(manifest.read_bytes())
    for name, expected in provenance["sha256"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Pinned source differs: {name}")
    models = OnnxModels(weights)
    settings = Settings()
    summary = []
    for scene, frame, target, mask_object in cases:
        print("Starting", scene, frame, target, "supplied mask:", mask_object, flush=True)
        mesh = Mesh.read(
            source / f"models_cad/obj_{target:06}.ply", scale=0.001, frame=f"tless_object{target}"
        )
        current, truth = read_case(source, scene, frame, target, mask_object)
        metadata = json.loads((source / "models_cad/models_info.json").read_bytes())[str(target)]
        if metadata.get("symmetries_continuous"):
            raise ValueError("This fixed-error pilot requires finite symmetries")
        symmetries = [np.eye(4)]
        for raw in metadata.get("symmetries_discrete", []):
            transform = np.array(raw).reshape(4, 4)
            transform[:3, 3] /= 1000
            symmetries.append(rigid(transform))

        def progress(stage, done, total):
            if done % 32 == 0 or done == total:
                print(stage, done, "/", total, flush=True)

        result = Estimator(mesh, models, settings).register(current, progress=progress)
        name = f"object{target:06}_scene{scene:06}_frame{frame:06}_mask{mask_object:06}"
        folder = export_review(output / name, result)
        choices = json.loads((source / "reference_selection.json").read_bytes())["models"]
        teaching = next(x for x in choices if x["object_id"] == target)
        reference, _ = read_case(source, teaching["positive_scene"],
                                 teaching["training_frame"], target)
        digest = save_recipe(folder / "trained.sfp", Recipe(mesh, reference, settings))
        assessment = {
            "case": name,
            "supplied_instance_mask": True,
            "ground_truth_pose_entered_runtime": False,
            "recipe_sha256": digest,
            "result": evaluate(result, truth, symmetries),
            "accepted_candidates": len(result.accepted),
            "seconds": result.execution.get("seconds", {}),
        }
        if result.candidates:
            from types import SimpleNamespace
            raw = evaluate(SimpleNamespace(best=result.candidates[0]), truth, symmetries)
            raw.pop("accepted")
            assessment["raw_top_proposal_evaluation_only"] = raw
        (folder / "evaluation.json").write_text(
            json.dumps(assessment, indent=2) + "\n", encoding="utf-8"
        )
        summary.append(assessment)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(assessment), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--case",
        nargs=4,
        type=int,
        action="append",
        metavar=("SCENE", "FRAME", "TARGET", "MASK_OBJECT"),
    )
    args = parser.parse_args()
    run(args.source, args.weights, args.output, args.case or [(2, 13, 6, 6), (3, 10, 6, 8)])
