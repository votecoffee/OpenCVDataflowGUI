"""Original MIT recorded MIL fixtures, annotations, seeds and deliberate guard failures."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
CASES = ("translation", "template_reacquisition", "region_seed", "wrong_region", "work_limit")


def build():
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    rng = np.random.default_rng(156)
    background = rng.integers(15, 35, (128, 160), np.uint8)
    patch = rng.integers(70, 230, (32, 32), np.uint8)
    cv2.rectangle(patch, (1, 1), (30, 30), 250, 2)
    cv2.line(patch, (3, 10), (28, 24), 15, 3)
    images = {"template.png": patch, "blank.png": np.zeros_like(background)}
    annotations = []
    for index in range(7):
        image = background.copy()
        x, y = 44 + index * 2, 42
        image[y : y + 32, x : x + 32] = patch
        name = f"frame_{index:02d}.png"
        images[name] = image
        annotations.append({"file": name, "bounds": [x, y, 32, 32]})
    for name, image in images.items():
        ok, data = cv2.imencode(".png", image)
        assert ok
        (assets / name).write_bytes(data.tobytes())
    registry = builtin_registry()
    for case in CASES:
        graph = Graph()

        def add(kind, name, position, **parameters):
            block = graph.add(registry.create(kind, block_id=name, parameters=parameters))
            block.position = list(position)
            return block

        paths = [f"assets/{entry['file']}" for entry in annotations]
        if case == "template_reacquisition":
            paths = paths[:4] + ["assets/blank.png"] + ["assets/frame_00.png"] * 3
        add(
            "input.file",
            "source",
            (0, 0),
            source_mode="files",
            paths=paths,
            path_base="project",
            color_format="gray",
            repeat="stop",
        )
        add(
            "sequence.track_mil",
            "track",
            (720, 0),
            warmup=3,
            search_radius=8,
            max_misses=1,
            initial_bounds=[]
            if case in ("template_reacquisition", "region_seed")
            else [44, 42, 32, 32],
            maximum_work=1 if case == "work_limit" else 4_000_000,
        )
        graph.connect("source", "image", "track", "image")
        if case == "template_reacquisition":
            add(
                "input.file",
                "template",
                (0, 330),
                path="assets/template.png",
                path_base="project",
                color_format="gray",
            )
            add("inspect.match_template", "detect", (360, 200), min_score=0.95)
            graph.connect("source", "image", "detect", "image")
            graph.connect("template", "image", "detect", "template")
            graph.connect("detect", "location", "track", "seed")
        if case in ("region_seed", "wrong_region"):
            add(
                "region.shape",
                "selection",
                (360, 160),
                rectangle=([44, 42, 32, 32] if case == "region_seed" else [4, 4, 16, 16]),
            )
            graph.connect("source", "image", "selection", "image")
            graph.connect("selection", "region", "track", "region")
        note = add("layout.comment", "scope", (360, 550))
        note.name = "Original clip and MIL limitations"
        note.node_size = "large"
        note.description = (
            "Original synthetic frames, template, annotations and project: OpenCVDataflowGUI "
            "contributors, MIT. No third-party/customer images. The 32x32 target translates by "
            "two pixels/frame. Search radius=8; warmup=3 withholds the first two activations, "
            "including bootstrap "
            "estimates. MIL is approximate/stochastic, fixed-size and has no native probability. "
            "Pose angle zero follows image axes, not measured part rotation. "
            "Template reacquisition "
            "includes a deliberate blank/lost frame: the full replay fails for that "
            "observation, "
            "then a new track ID/readiness appears. Wrong region and work limit "
            "deliberately reject "
            "processing. Loss guards and software localization do not qualify robot picking."
        )
        graph.save(HERE / f"{case}.json")
        graph.close()
    truth = {
        "creator": "OpenCVDataflowGUI contributors",
        "license": "MIT",
        "cases": list(CASES),
        "annotations": annotations,
        "warmup": 3,
        "search_radius": 8,
        "accepted_center_error_limit_pixels": 8,
        "asset_sha256": {
            name: hashlib.sha256((assets / name).read_bytes()).hexdigest()
            for name in sorted(images)
        },
    }
    (HERE / "truth.json").write_bytes((json.dumps(truth, indent=2) + "\n").encode())


if __name__ == "__main__":
    build()
