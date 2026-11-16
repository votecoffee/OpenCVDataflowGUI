"""Published GDRNPP weights on real T-LESS captures; pose labels are evaluation-only."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from opencv_simple_foundation_pose.specs import Frame, Mesh

from opencv_simple_gdrnpp.estimate import Detection, Estimator
from opencv_simple_gdrnpp.review import export_review, refine_depth, verify
from opencv_simple_gdrnpp.runtime import TorchModel


def evaluate(pose, truth, metadata):
    if not truth:
        return dict(target_present=False, raw_pose_is_extra=pose is not None)
    errors = []
    symmetries = [np.eye(4)]
    for raw in metadata.get("symmetries_discrete", []):
        symmetry = np.array(raw).reshape(4, 4)
        symmetry[:3, 3] /= 1000
        symmetries.append(symmetry)
    if metadata.get("symmetries_continuous"):
        raise ValueError("This pilot uses finite symmetries")
    for expected in truth:
        for symmetry in symmetries:
            target = expected @ symmetry
            re = float(
                np.rad2deg(
                    np.arccos(np.clip((np.trace(pose[:3, :3].T @ target[:3, :3]) - 1) / 2, -1, 1))
                )
            )
            te = float(np.linalg.norm(pose[:3, 3] - target[:3, 3]) * 1000)
            errors.append((re, te))
    re, te = min(errors, key=lambda v: max(v[0] / 5, v[1] / 5))
    return dict(
        target_present=True,
        rotation_error_degrees=re,
        translation_error_mm=te,
        correct_5mm_5deg=re <= 5 and te <= 5,
    )


def run(source, models, output, threads=4, depth_refine=False):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError("Choose a new pilot output directory")
    provenance = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
    for relative, expected in provenance["sha256"].items():
        if hashlib.sha256((source / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Pinned source differs: " + relative)
    output.mkdir(parents=True)
    model = TorchModel(models, threads=threads)
    estimator = Estimator(model)
    mesh = Mesh.read(source / "models_cad/obj_000006.ply", scale=0.001, frame="tless_object6")
    metadata = json.loads((source / "models_cad/models_info.json").read_text())["6"]
    extents = np.ptp(mesh.vertices, axis=0)
    summary = []
    for scene, number, selected in [(2, 3, 6), (2, 13, 6), (3, 10, 8)]:
        folder = source / f"test_primesense/{scene:06d}"
        entries = json.loads((folder / "scene_gt.json").read_text())[str(number)]
        camera = json.loads((folder / "scene_camera.json").read_text())[str(number)]
        ids = [i for i, e in enumerate(entries) if e["obj_id"] == selected]
        if len(ids) != 1:
            raise ValueError("Require one externally selected visible instance")
        mask = cv2.imread(str(folder / f"mask_visib/{number:06d}_{ids[0]:06d}.png"), 0)
        rgb = cv2.imread(str(folder / f"rgb/{number:06d}.png"))[..., ::-1]
        depth = (
            cv2.imread(str(folder / f"depth/{number:06d}.png"), cv2.IMREAD_UNCHANGED).astype(
                np.float32
            )
            * camera["depth_scale"]
            / 1000
        )
        k = np.array(camera["cam_K"]).reshape(3, 3)
        y, x = np.where(mask)
        box = [int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)]
        capture = f"tless_scene{scene:06d}_frame{number:06d}"
        detection = Detection("supplied_instance", tuple(box), source="published_visible_mask_bbox")
        report, fields = estimator.predict(rgb, k, [detection], extents, capture_id=capture)
        # Only the external rectangle is supplied to geometric verification.
        rectangle = np.zeros(mask.shape, np.uint8)
        rectangle[box[1] : box[3], box[0] : box[2]] = 1
        frame = Frame(rgb, depth, k, rectangle)
        start = time.perf_counter()
        if depth_refine:
            refine_depth(mesh, frame, report)
        report["seconds"]["depth_refinement"] = time.perf_counter() - start
        start = time.perf_counter()
        verify(mesh, frame, report)
        report["seconds"]["depth_verification"] = time.perf_counter() - start
        report["assistance"] = "Published visible-mask bounding rectangle, not automatic detection"
        export_review(output / capture, mesh, frame, report, fields)
        # GT is assembled only after inference/verification, never supplied to either.
        truth = []
        for entry in entries:
            if entry["obj_id"] == 6:
                pose = np.eye(4)
                pose[:3, :3] = np.array(entry["cam_R_m2c"]).reshape(3, 3)
                pose[:3, 3] = np.array(entry["cam_t_m2c"]) / 1000
                truth.append(pose)
        candidate = report["candidates"][0]
        pose = np.array(candidate["pose"]["source_to_target"])
        raw_pose = np.array(candidate.get("unrefined_pose", candidate["pose"])["source_to_target"])
        result = dict(
            capture_id=capture,
            accepted=report["accepted_candidates"] > 0,
            raw_pose=evaluate(raw_pose, truth, metadata),
            final_pose=evaluate(pose, truth, metadata),
            fit=candidate["fit"],
            depth_refinement=candidate.get("depth_refinement"),
            pose_gt_entered_runtime=False,
            network_seconds=report["seconds"]["network"],
        )
        (output / capture / "evaluation.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        summary.append(result)
        print(json.dumps(result), flush=True)
    (output / "summary.json").write_text(
        json.dumps(dict(provider=model.provenance, cases=summary), indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("models", type=Path)
    p.add_argument("new_output", type=Path)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--depth-refine", action="store_true")
    a = p.parse_args()
    run(a.source, a.models, a.new_output, a.threads, a.depth_refine)


if __name__ == "__main__":
    main()
