"""Build the portable VisA PCB2 matching example from attributed source photographs."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from opencv_geometric_pattern_match import InspectionRecipe, Search, save_recipe, train
from opencv_geometric_pattern_match.dataflow import recipe_reference

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

SOURCE_ASSETS = Path(__file__).resolve().parents[3] / "assets"
SAMPLES = (
    (
        "0000", "b3f01fc1c1bd43df2c5a1afa63348097f7ecbcc9ebe2fd43bed6f2d074531a37",
        [(99, 172), (556, 178), (96, 360), (553, 365)],
    ),
    (
        "0621", "d258d60936443d62745e0c208a78ada21227df6b1405bda0967e105b045c4720",
        [(89, 172), (547, 184), (85, 360), (542, 372)],
    ),
)
ROI = (85, 161, 479, 209)
DATUM = (99., 172.)
IMAGE_SIZE = (702, 535)


def build_example(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    assets = root / "assets"
    assets.mkdir()
    frames = []
    for image_id, sha256, points in SAMPLES:
        original = (SOURCE_ASSETS / f"visa-pcb2-normal-{image_id}.jpg").read_bytes()
        if hashlib.sha256(original).hexdigest() != sha256:
            raise ValueError(f"VisA source {image_id} did not match its recorded hash")
        image = cv2.imdecode(np.frombuffer(original, np.uint8), cv2.IMREAD_COLOR)
        pixels = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, 94])
        if not ok:
            raise RuntimeError(f"Could not encode VisA image {image_id}")
        name = f"pcb2-{image_id}.jpg"
        (assets / name).write_bytes(encoded.tobytes())
        frames.append({"path": f"assets/{name}", "mounting_holes": points})

    reference_bytes = (root / frames[0]["path"]).read_bytes()
    reference = cv2.imdecode(np.frombuffer(reference_bytes, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    pattern = train(gray, roi=ROI, origin=DATUM)
    search = Search(
        angle=(-5., 5., 2.5), scale=(.96, 1.04, .04), accept=.55, coarse_accept=.27,
        max_results=1,
    )
    save_recipe(assets / "pcb2.npz", InspectionRecipe(pattern, search))
    (root / "landmarks.json").write_text(json.dumps(frames, indent=2) + "\n", encoding="utf-8")

    registry = builtin_registry()
    graph = Graph()
    try:
        source = graph.add(registry.create("input.file", block_id="scene", parameters={
            "source_mode": "files", "paths": [frame["path"] for frame in frames],
            "path_base": "project",
        }))
        source.name = "VisA PCB2 photographs"
        reference_source = graph.add(registry.create(
            "input.file", block_id="reference",
            parameters={"path": frames[0]["path"], "path_base": "project"},
        ))
        reference_source.name = "Training reference · VisA 0000"
        reference_source.position = [0., 245.]
        matcher = graph.add(registry.create(
            "pattern.ogpm_recipe", block_id="matcher",
            parameters=recipe_reference(root, "assets/pcb2.npz"),
        ))
        matcher.name = "SimplePatternMatch · Locate the PCB"
        matcher.position = [375., 0.]
        matcher.exposed_outputs.append("result_graphics")
        graph.connect(source.id, "image", matcher.id, "image")
        graph.save(root / "project.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="New output directory for the project and assets")
    build_example(parser.parse_args().directory)
