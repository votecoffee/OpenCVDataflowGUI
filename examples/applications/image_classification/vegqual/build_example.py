"""Build the small saved example from the original VegQual v4 ZIP."""

import argparse
import ast
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.simple_classifier.dataflow import ClassifyImage
from opencv_dataflow.simple_classifier.model import (
    SURFACE_TEXTURE,
    Example,
    stage_classifier,
    train_classifier,
)

ARCHIVE_MD5 = "de19cbfe7d9792c5db37b6c3af0458e1"
PREFIX = "VegQual/VegMining"
COUNTS = {"valid": 5, "test": 4, "train": 24}
CLASS_IDS = (12, 0, 9, 3)


def _crop(image, box):
    height, width = image.shape[:2]
    x, y, w, h = box
    left = max(0, round(width * (x - w / 2)))
    top = max(0, round(height * (y - h / 2)))
    right = min(width, round(width * (x + w / 2)))
    bottom = min(height, round(height * (y + h / 2)))
    crop = image[top:bottom, left:right]
    scale = 192 / max(crop.shape[:2])
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def select_images(archive, destination):
    labels = ast.literal_eval(
        next(line for line in archive.read(f"{PREFIX}/data.yaml").decode().splitlines()
             if line.startswith("names:")).removeprefix("names:").strip()
    )
    assignments = {labels[class_id]: [] for class_id in CLASS_IDS}
    samples = []
    seen_origins, seen_crops = set(), set()
    rng = random.Random(4765)
    for split, count in COUNTS.items():
        by_class = defaultdict(list)
        for label_path in sorted(archive.namelist()):
            if not label_path.startswith(f"{PREFIX}/{split}/labels/"):
                continue
            if not label_path.endswith(".txt"):
                continue
            rows = archive.read(label_path).decode().splitlines()
            if len(rows) != 1:
                continue
            fields = rows[0].split()
            if len(fields) != 5 or int(fields[0]) not in CLASS_IDS:
                continue
            source = label_path.replace("/labels/", "/images/").removesuffix(".txt") + ".jpg"
            by_class[int(fields[0])].append((source, label_path, list(map(float, fields[1:]))))
        for class_id in CLASS_IDS:
            candidates = by_class[class_id]
            rng.shuffle(candidates)
            label = labels[class_id]
            selected = 0
            for source, label_path, box in candidates:
                origin = Path(source).name.split(".rf.", 1)[0].casefold()
                if origin in seen_origins:
                    continue
                data = np.frombuffer(archive.read(source), np.uint8)
                image = cv2.imdecode(data, cv2.IMREAD_COLOR)
                crop = _crop(image, box)
                digest = hashlib.sha256(str(crop.shape).encode() + crop.tobytes()).hexdigest()
                if digest in seen_crops:
                    continue
                seen_origins.add(origin)
                seen_crops.add(digest)
                selected += 1
                path = Path("images") / split / label / f"{selected:02d}.jpg"
                output = destination / path
                output.parent.mkdir(parents=True, exist_ok=True)
                ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if not ok:
                    raise RuntimeError(f"Could not encode {source}")
                output.write_bytes(encoded.tobytes())
                if split != "test":
                    assignments[label].append(Example(output, validation=split == "valid"))
                samples.append({
                    "image": path.as_posix(),
                    "label": label,
                    "source_image": source,
                    "source_annotation": label_path,
                    "yolo_xywh": box,
                    "source_split": split,
                })
                if selected == count:
                    break
            if selected != count:
                raise RuntimeError(f"Only {selected} unique {split} images for {label}")
    return assignments, samples


def build(archive_path, destination):
    with archive_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "md5").hexdigest()
    if digest != ARCHIVE_MD5:
        raise ValueError("Archive checksum does not match VegQual v4")
    destination.mkdir(parents=True, exist_ok=False)
    with ZipFile(archive_path) as archive:
        assignments, samples = select_images(archive, destination)
    cv2.setRNGSeed(4765)
    legacy, legacy_health = train_classifier(assignments)
    legacy_manifest = stage_classifier(
        destination, assignments, legacy, legacy_health, revision="vegqual-v4"
    )
    model, health = train_classifier(assignments, profile=SURFACE_TEXTURE)
    manifest = stage_classifier(
        destination, assignments, model, health, revision="vegqual-v4-surface"
    )
    (destination / "ground_truth.json").write_text(
        json.dumps(samples, indent=2) + "\n", encoding="utf-8"
    )

    test_images = [row["image"] for row in samples if row["source_split"] == "test"]
    write_project(destination / "project.json", test_images, manifest)
    write_project(destination / "color_and_layout.json", test_images, legacy_manifest)
    print(f"Training health {health.score:.1f}% on {sum(health.validation_counts)} images")


def write_project(path, test_images, manifest):
    registry = builtin_registry()
    graph = Graph()
    try:
        inputs = graph.add(registry.create("input.file", block_id="test_images", parameters={
            "path_base": "project",
            "source_mode": "files",
            "paths": test_images,
        }))
        inputs.name = "VegQual held-out test images"
        classifier = graph.add(registry.create(
            ClassifyImage.type_id, block_id="classifier", parameters={"model": manifest}
        ))
        classifier.position = [340.0, 0.0]
        graph.connect(inputs.id, "image", classifier.id, "image")
        graph.save(path)
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    build(args.archive.resolve(), args.destination.resolve())
