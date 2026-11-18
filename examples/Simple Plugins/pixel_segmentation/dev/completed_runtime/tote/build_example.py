"""Build a portable ARMBench tote foreground example from Amazon's tiny archive."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import cv2
import numpy as np
from opencv_defect_segmentation import PixelSegmenter, TrainingImage
from opencv_defect_segmentation.dataflow import pin_assets

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.graphics_layers import RESULT_GRAPHICS

ARCHIVE_SHA256 = "30f2539a2c8ae32ae25000ee9025657e2c49529cf9e513196597f9a94e7ca0d7"
SIZE = (612, 512)
MATRIX = [[.7, -.35, -.35, 128.]]


def save_image(path, image):
    options = [cv2.IMWRITE_JPEG_QUALITY, 93] if path.suffix == ".jpg" else []
    ok, encoded = cv2.imencode(path.suffix, image, options)
    if not ok:
        raise ValueError(f"Cannot encode {path.name}")
    path.write_bytes(encoded.tobytes())
    return encoded


def polygons_mask(polygons, scale):
    mask = np.zeros(SIZE[::-1], np.uint8)
    for polygon in polygons:
        vertices = np.rint(np.asarray(polygon) * scale).astype(np.int32)
        cv2.fillPoly(mask, [vertices], 255)
    return mask


def prepare_frames(archive, root, registry):
    labels = json.loads(Path(__file__).with_name("polygons.json").read_text(encoding="utf-8"))
    scale = np.asarray(SIZE) / labels["coordinate_grid"]
    left, top, right, bottom = labels["tote"]
    region = polygons_mask([[(left, top), (right, top), (right, bottom), (left, bottom)]], scale)
    save_image(root / "region.png", region)
    transform = registry.create("cv.channel_transform", parameters={"matrix": MATRIX})
    manifests = {"train": [], "validation": []}
    validation_images = []
    samples = []
    with tarfile.open(archive, mode="r:gz") as dataset:
        for frame in labels["frames"]:
            image_id = frame["id"]
            member = f"./armbench-segmentation-tiny/{image_id}.jpg"
            with dataset.extractfile(member) as original:
                pixels = cv2.imdecode(np.frombuffer(original.read(), np.uint8), cv2.IMREAD_COLOR)
            image = cv2.resize(pixels, SIZE, interpolation=cv2.INTER_AREA)
            image_path = f"assets/{image_id}.jpg"
            encoded = save_image(root / image_path, image)
            color = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            transform.execute({"image": color})
            signal = transform.last_result.outputs["image"]
            signal_path = f"assets/{image_id}-blue.png"
            save_image(root / signal_path, signal)
            foreground = polygons_mask(frame["polygons"], scale)
            label_map = np.where(region != 0, np.where(foreground != 0, 2, 1), 0).astype(np.uint8)
            if frame["split"] == "train":
                kernel = np.ones((5, 5), np.uint8)
                boundary = cv2.dilate(foreground, kernel) != cv2.erode(foreground, kernel)
                label_map[boundary] = 0
                samples.append(TrainingImage(signal, label_map))
            else:
                validation_images.append(image_path)
            label_path = f"assets/{image_id}-labels.png"
            save_image(root / label_path, label_map)
            manifests[frame["split"]].append({"image": signal_path, "labels": label_path})
    for split, filename in (("train", "training.json"), ("validation", "validation.json")):
        (root / filename).write_text(
            json.dumps(manifests[split], indent=2) + "\n", encoding="utf-8"
        )
    return samples, validation_images


def add_pick_candidate(graph, registry, segment, overlay):
    distance = graph.add(registry.create(
        "cv.distance_transform", block_id="distance", parameters={"mask_size": "precise"},
    ))
    distance.name = "Distance to foreground boundary"
    distance.position = [1100., 360.]
    distance.preview_output = "preview"
    eligible = graph.add(registry.create(
        "cv.in_range", block_id="eligible", parameters={"lower": [12.], "upper": [1e6]},
    ))
    eligible.name = "Minimum interior clearance (px)"
    eligible.position = [1460., 360.]
    peak = graph.add(registry.create(
        "cv.image_statistics", block_id="pick", parameters={"statistic": "maximum"},
    ))
    peak.name = "Best image-space pick candidate"
    peak.position = [1820., 360.]
    peak.exposed_outputs = ["value", "maximum_location", "selected_count", RESULT_GRAPHICS]
    composite = graph.add(registry.create(
        "cv.overlay_result_graphics", block_id="pick_overlay",
        parameters={"excluded_graphics": {"graphics_1": ["minimum"]}},
    ))
    composite.name = "Show candidate on the photo"
    composite.position = [2220., 0.]
    composite.exposed_inputs = ["image", "graphics_1"]
    composite.preview_output = "image"
    graph.connect(segment.id, "mask", distance.id, "image")
    graph.connect(distance.id, "distance", eligible.id, "image")
    graph.connect(distance.id, "distance", peak.id, "image")
    graph.connect(eligible.id, "image", peak.id, "mask")
    graph.connect(overlay.id, "overlay", composite.id, "image")
    graph.connect(peak.id, RESULT_GRAPHICS, composite.id, "graphics_1")


def write_project(root, registry, paths, *, pick_candidate=False):
    graph = Graph()
    try:
        source = graph.add(registry.create("input.file", block_id="source", parameters={
            "source_mode": "files", "paths": paths, "path_base": "project",
        }))
        source.name = "Held-out ARMBench tote photos"
        transform = graph.add(registry.create("cv.channel_transform", block_id="blue", parameters={
            "matrix": MATRIX,
        }))
        transform.name = "Measure blue dominance"
        transform.position = [370., 0.]
        segment = graph.add(registry.create(
            "inspection.ocv_segment_image", block_id="segment", parameters={
                **pin_assets(root, "model.yml", "region.png"),
                "min_area_px": 180, "workers": 4, "render_overlay": True, "emit_score_map": True,
            },
        ))
        segment.name = "SimpleSegmentation · Inventory foreground"
        segment.position = [740., 0.]
        segment.preview_output = "overlay"
        blobs = graph.add(registry.create(
            "inspect.blobs", block_id="regions", parameters={"min_area": 1},
        ))
        blobs.name = "Group foreground regions"
        blobs.position = [1100., 0.]
        overlay = graph.add(registry.create(
            "collection.draw", block_id="color_overlay",
            parameters={"draw_bounds": False, "draw_centers": False, "labels": "none"},
        ))
        overlay.name = "Show outlines on the photo"
        overlay.position = [1460., 0.]
        overlay.preview_output = "overlay"
        note = graph.add(registry.create("layout.comment", block_id="source_credits"))
        note.name = "ARMBench image credits"
        note.description = (
            "Amazon.com, Inc. · ARMBench Sparrow, Object Segmentation tiny archive · CC BY 4.0.\n"
            "Photos resized and re-encoded. Binary inventory labels drawn for this example.\n"
            "Citation and source archive details are in SOURCES.md."
        )
        note.position = [0., -245.]
        note.node_size = "large"
        graph.connect(source.id, "image", transform.id, "image")
        graph.connect(transform.id, "image", segment.id, "image")
        graph.connect(segment.id, "mask", blobs.id, "image")
        graph.connect(blobs.id, "collection", overlay.id, "collection")
        graph.connect(source.id, "image", overlay.id, "image")
        if pick_candidate:
            add_pick_candidate(graph, registry, segment, overlay)
        graph.save(root / ("pick_candidate.json" if pick_candidate else "project.json"))
    finally:
        graph.close()


def build_example(archive, root):
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError("Archive SHA-256 differs from Amazon's ARMBench tiny sample")
    root.mkdir(parents=True, exist_ok=False)
    (root / "assets").mkdir()
    registry = builtin_registry()
    samples, validation_images = prepare_frames(archive, root, registry)
    cv2.setRNGSeed(34)
    PixelSegmenter.train(samples, seed=34).save(root / "model.yml")
    write_project(root, registry, validation_images)
    write_project(root, registry, validation_images, pick_candidate=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Amazon ARMBench segmentation tiny archive")
    parser.add_argument("directory", type=Path, help="New output directory")
    args = parser.parse_args()
    build_example(args.archive, args.directory.resolve())
