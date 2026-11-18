"""Fixed-setting GDRNPP raw/local-ICP evaluation on a preselected real scene extension."""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
from opencv_simple_foundation_pose.render import fit_evidence
from opencv_simple_foundation_pose.specs import Frame
from replay_tless import evaluate

from opencv_simple_gdrnpp.estimate import Detection, Estimator
from opencv_simple_gdrnpp.refinement import _centered
from opencv_simple_gdrnpp.review import overlay, refine_depth, verify
from opencv_simple_gdrnpp.runtime import TorchModel
from opencv_simple_gdrnpp.setup import load_recipe


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )


def load_frame(source, row):
    folder = source / f"test_primesense/{row['scene']:06d}"
    camera = json.loads((folder / "scene_camera.json").read_bytes())[str(row["frame"])]
    rgb = cv2.imread(str(folder / f"rgb/{row['frame']:06d}.png"))[..., ::-1]
    depth = cv2.imread(str(folder / f"depth/{row['frame']:06d}.png"), cv2.IMREAD_UNCHANGED).astype(
        np.float32
    ) * (camera["depth_scale"] / 1000)
    x, y, w, h = row["bbox_xywh"]
    box = (x, y, x + w, y + h)
    k = np.array(camera["cam_K"]).reshape(3, 3)
    frame = Frame(rgb, depth, k, np.ones(depth.shape, np.uint8))
    return frame, Detection("supplied_instance", box, source="published_visible_bbox")


def aggregate(rows, mode):
    positives = [row for row in rows if row["selection"]["target_present"]]
    negatives = [row for row in rows if not row["selection"]["target_present"]]
    return dict(
        positives=len(positives),
        positive_depth_accepted=sum(row[mode]["accepted"] for row in positives),
        positive_pose_correct_5mm_5deg=sum(
            row[mode]["pose_error"]["correct_5mm_5deg"] for row in positives
        ),
        accepted_positive_pose_correct_5mm_5deg=sum(
            row[mode]["accepted"] and row[mode]["pose_error"]["correct_5mm_5deg"]
            for row in positives
        ),
        accepted_positive_pose_outside_5mm_5deg=sum(
            row[mode]["accepted"] and not row[mode]["pose_error"]["correct_5mm_5deg"]
            for row in positives
        ),
        target_absent_controls=len(negatives),
        target_absent_depth_accepted=sum(row[mode]["accepted"] for row in negatives),
    )


def run(source, models, recipe_path, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError("Choose a fresh validation result directory")
    provenance = json.loads((source / "provenance.json").read_bytes())
    for name, digest in provenance["sha256"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Pinned source changed: " + name)
    selection = json.loads((source / "selection.json").read_bytes())
    if not selection["settings_selected_before_inference"]:
        raise ValueError("Freeze selection before loading the model")
    output.mkdir(parents=True)
    recipe = load_recipe(recipe_path)
    settings = recipe.settings.fit_settings()
    model = TorchModel(models, threads=recipe.settings.threads)
    estimator = Estimator(model, batch_size=recipe.settings.batch_size)
    metadata = dict(symmetries_discrete=[])
    for symmetry in recipe.symmetries:
        raw = symmetry.copy()
        raw[:3, 3] *= 1000  # The existing BOP evaluator accepts publisher millimeters.
        metadata["symmetries_discrete"].append(raw.reshape(-1).tolist())
    rows = []
    for selected in selection["cases"]:
        started = time.perf_counter()
        name = f"scene{selected['scene']:06d}_frame{selected['frame']:06d}"
        frame, detection = load_frame(source, selected)
        raw, _ = estimator.predict(
            frame.rgb,
            frame.camera,
            [detection],
            np.ptp(recipe.mesh.vertices, axis=0),
            capture_id=name,
        )
        verify(recipe.mesh, frame, raw, settings)
        refined = copy.deepcopy(raw)
        tick = time.perf_counter()
        refine_depth(
            recipe.mesh,
            frame,
            refined,
            method="projective_icp_v1",
            maximum_correction_m=recipe.settings.maximum_correction_m,
            fit_settings=settings,
        )
        refine_seconds = time.perf_counter() - tick
        verify(recipe.mesh, frame, refined, settings)
        numeric_seconds = time.perf_counter() - started
        case = output / name
        case.mkdir()
        # Labels are assembled only after all inference, refinement and fit decisions.
        folder = source / f"test_primesense/{selected['scene']:06d}"
        entries = json.loads((folder / "scene_gt.json").read_bytes())[str(selected["frame"])]
        truth = []
        if selected["target_present"]:
            entry = entries[selected["instance_index"]]
            assert entry["obj_id"] == 6
            expected = np.eye(4)
            expected[:3, :3] = np.array(entry["cam_R_m2c"]).reshape(3, 3)
            expected[:3, 3] = np.array(entry["cam_t_m2c"]) / 1000
            truth.append(expected)
        else:
            assert not any(entry["obj_id"] == 6 for entry in entries)
        result = dict(
            case=name,
            selection=selected,
            pose_labels_entered_runtime=False,
            numeric_seconds=numeric_seconds,
            refinement_seconds=refine_seconds,
            network_seconds=raw["seconds"]["network"],
        )
        for mode, report in (("raw", raw), ("icp", refined)):
            c = report["candidates"][0]
            result[mode] = dict(
                accepted=bool(report["accepted_candidates"]),
                fit=c["fit"],
                pose_error=evaluate(np.array(c["pose"]["source_to_target"]), truth, metadata),
                refinement=c.get("depth_refinement"),
            )
            write_json(case / f"{mode}_report.json", report)
            cv2.imwrite(
                str(case / f"{mode}_overlay.png"), overlay(recipe.mesh, frame, report)[..., ::-1]
            )
        mask = np.zeros(frame.mask.shape, np.uint8)
        x0, y0, x1, y1 = map(int, detection.box)
        mask[y0:y1, x0:x1] = 1
        result["annotated_pose_depth_fit_evaluation_only"] = [
            fit_evidence(
                recipe.mesh,
                Frame(frame.rgb, frame.depth, frame.camera, mask),
                _centered(recipe.mesh, pose),
                settings,
            )
            for pose in truth
        ]
        write_json(case / "evaluation.json", result)
        rows.append(result)
        print(
            json.dumps(
                dict(
                    case=name,
                    target_present=selected["target_present"],
                    raw=result["raw"]["pose_error"],
                    icp=result["icp"]["pose_error"],
                    raw_accepted=result["raw"]["accepted"],
                    icp_accepted=result["icp"]["accepted"],
                )
            ),
            flush=True,
        )
    memory = psutil.Process().memory_info()
    summary = dict(
        scope=(
            "Preselected new-scene development cases; supplied rectangles, not detection or picking"
        ),
        recipe_archive_sha256=hashlib.sha256(Path(recipe_path).read_bytes()).hexdigest(),
        recipe_content_sha256=recipe.content_sha256,
        source_provenance=provenance,
        provider=model.provenance,
        settings=dict(
            distance_tolerance_m=settings.distance_tolerance,
            minimum_coverage=settings.minimum_coverage,
            maximum_contradiction=settings.maximum_contradiction,
        ),
        metrics={mode: aggregate(rows, mode) for mode in ("raw", "icp")},
        cases=rows,
        same_depth_used_for_refinement_and_verification=True,
        timing_scope=(
            "One sequential CPU replay; numeric excludes overlays and GT evaluation, "
            "includes frame decode, network, raw fit, ICP and refined fit; "
            "not a repeatable full-cycle benchmark"
        ),
        os_peak_working_set_bytes=getattr(memory, "peak_wset", None),
    )
    write_json(output / "summary.json", summary)
    print(json.dumps(summary["metrics"], indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("models", type=Path)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("new_output", type=Path)
    args = parser.parse_args()
    run(args.source, args.models, args.recipe, args.new_output)
