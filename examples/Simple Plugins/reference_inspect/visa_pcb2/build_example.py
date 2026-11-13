"""Build a posed VisA reference-inspection example with an untouched test split."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from opencv_geometric_pattern_match.dataflow import recipe_reference
from opencv_simple_reference_inspect import save_model, train
from opencv_simple_reference_inspect.dataflow import relative_fixture
from opencv_simple_reference_inspect.result_review import ResultReview

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def png(path, image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode an example image.")
    path.write_bytes(encoded.tobytes())


def build(source, target):
    source, target = Path(source), Path(target)
    manifest = json.loads(Path(__file__).with_name("source_manifest.json").read_bytes())
    rows = manifest["selection"]
    if [(row["split"], row["label"]) for row in rows] != (
        [("train", "normal")] * 8 + [("test", "normal")] * 4 + [("test", "anomaly")] * 4
    ):
        raise ValueError("Unexpected pinned split assignment.")
    for member, info in manifest["files"].items():
        if hashlib.sha256((source / member).read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"Source hash mismatch: {member}")
    target.mkdir(parents=True, exist_ok=False)
    assets = target / "assets"
    assets.mkdir()
    images, paths = [], []
    for row in rows:
        raw = (source / row["image"]).read_bytes()
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), 1)
        image = cv2.resize(image, (702, 535), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 94])
        if not ok:
            raise ValueError("Could not encode a VisA photograph.")
        relative = f"assets/{row['split']}-{row['label']}-{Path(row['image']).stem}.jpg"
        (target / relative).write_bytes(encoded.tobytes())
        images.append(cv2.cvtColor(cv2.imdecode(encoded, 1), cv2.COLOR_BGR2GRAY))
        paths.append(relative)
    source_matcher = Path(__file__).resolve().parents[2] / "pattern_matching/visa_pcb2/assets"
    for name in ("pcb2.npz", "pcb2.reference.png"):
        shutil.copyfile(source_matcher / name, assets / name)
    registry = builtin_registry()
    graph = Graph()
    poses, match_scores = [], []
    try:
        scene = graph.add(
            registry.create(
                "input.file",
                block_id="source",
                parameters={
                    "source_mode": "files",
                    "paths": paths,
                    "path_base": "project",
                },
            )
        )
        matcher = graph.add(
            registry.create(
                "pattern.ogpm_recipe",
                block_id="matcher",
                parameters=recipe_reference(target, "assets/pcb2.npz"),
            )
        )
        matcher.position = [360.0, 0.0]
        graph.connect(scene.id, "image", matcher.id, "image")
        graph.save(target / "project.json")
        for row in rows:
            graph.run()
            if matcher.state != "pass":
                raise ValueError(f"Cannot align {row['image']}: {matcher.error}")
            poses.append(np.asarray(matcher.last_result.outputs["pose"]["source_to_target"])[:2])
            match_scores.append(matcher.last_result.outputs["score"])
        transforms = [relative_fixture(pose, poses[0]) for pose in poses]
        region = np.zeros(images[0].shape, np.uint8)
        region[60:411, 70:591] = 255
        stable = np.zeros_like(region)
        stable[185:355, 100:550] = 255
        png(assets / "inspection-mask.png", region)
        png(assets / "normalization-mask.png", stable)
        model = train(
            images[:8],
            inspection_mask=region,
            normalization_mask=stable,
            reference_to_images=[None, *transforms[1:8]],
            reference_fixture=poses[0],
        )
        digest = save_model(assets / "reference.sri", model)
        evaluation = []
        for index in range(8, len(rows)):
            row, image = rows[index], images[index]
            result = model.inspect(image, reference_to_image=transforms[index])
            report = dict(
                image=paths[index],
                original_image=row["image"],
                label=row["label"],
                matching_score=match_scores[index],
                inspection=result.report,
                predicted_label=("normal" if result.passed else "anomaly")
                if result.report["evaluated"]
                else "not_evaluated",
            )
            if row["mask"]:
                mask = cv2.imdecode(np.frombuffer((source / row["mask"]).read_bytes(), np.uint8), 0)
                mask = cv2.resize(mask, image.shape[::-1], interpolation=cv2.INTER_NEAREST) != 0
                mask_name = f"assets/truth-anomaly-{Path(row['image']).stem}.png"
                png(target / mask_name, mask.astype(np.uint8) * 255)
                maps = result.source_maps()
                prediction = maps["defect_mask"] != 0
                true_positive = int(np.count_nonzero(prediction & mask))
                report.update(
                    ground_truth_mask=mask_name,
                    mask_metrics={
                        "truth_pixels": int(mask.sum()),
                        "predicted_pixels": int(prediction.sum()),
                        "true_positive_pixels": true_positive,
                        "covered_truth_pixels": int(
                            np.count_nonzero(mask & (maps["inspected_mask"] != 0))
                        ),
                        "precision": true_positive / max(1, int(prediction.sum())),
                        "recall": true_positive / max(1, int(mask.sum())),
                    },
                )
            ResultReview(image, model, result).save(
                target / "results" / f"{row['label']}-{Path(row['image']).stem}"
            )
            evaluation.append(report)
        inspection = graph.add(
            registry.create(
                "inspect.simple_reference",
                block_id="inspection",
                parameters={
                    "recipe_path": "assets/reference.sri",
                    "sha256": digest,
                    "render_assessment": True,
                },
            )
        )
        inspection.position = [720.0, 0.0]
        inspection.preview_output = "assessment"
        graph.connect(scene.id, "image", inspection.id, "image")
        graph.connect(matcher.id, "pose", inspection.id, "pose")
        scene.parameters["paths"] = paths[8:]
        scene.name = "Publisher-held-out normal and anomalous VisA photos"
        matcher.name = "SimplePatternMatch · align to the retained training fixture"
        note = graph.add(registry.create("layout.comment", block_id="credits"))
        note.name = "VisA PCB2 · CC BY 4.0 · retained hard case"
        note.description = (
            "Zou et al., VisA, ECCV 2022, CC BY 4.0. Eight publisher-training normal images "
            "teach the reference. Four separate normal and four anomalous test images "
            "are replayed. "
            "The matcher supplies pose; ground truth is evaluation-only. Default inspection "
            "limits are retained. Normal false alarms and missed defect pixels remain in "
            "evaluation.json; see README.md and SOURCES.md."
        )
        note.position = [0.0, -260.0]
        note.node_size = "large"
        graph.save(target / "project.json")
    finally:
        graph.close()
    (target / "evaluation.json").write_bytes((json.dumps(evaluation, indent=2) + "\n").encode())
    return evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Pinned original source members")
    parser.add_argument("output", type=Path, help="New directory")
    args = parser.parse_args()
    for row in build(args.source, args.output):
        print(row["image"], row["label"], row["predicted_label"])
