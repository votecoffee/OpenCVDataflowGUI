"""Pinned native learned tracking on attributed artificial part-motion clips."""

import hashlib
import json
import runpy
import tempfile
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.learned_tracking import preprocessing
from opencv_dataflow.model_contracts import export_contract

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "monocular_depth/assets/tless_color.png"
WEIGHTS = HERE / "assets/vit.onnx"
PREPARATION = runpy.run_path(str(Path(__file__).with_name("prepare_model.py")))
CASES = {
    "translation": {},
    "loss_expiry": {"minimum_contrast": 0.0},
    "detector_reseed": {"initial_bounds": [], "max_misses": 1},
    "region_seed": {"initial_bounds": []},
    "score_alarm": {"tracking_score_threshold": 0.99, "minimum_contrast": 0.0},
    "jump_alarm": {"maximum_center_jump": 0.5},
    "resource_limit": {"maximum_patch_side": 64},
    "new_engine": {"engine": "new"},
    "grayscale": {},
}
INITIAL = [92, 133, 43, 43]


def asset(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != raw:
        raise FileExistsError("Different existing assets preserved")
    path.write_bytes(raw)


def fixtures():
    raw = SOURCE.read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "04b19c87d60aaabe98c6be9655a58c5cc620ac31dc356ede55e89dc02cd2a166"
    )
    base = cv2.resize(
        cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR),
        (360, 270),
        interpolation=cv2.INTER_AREA,
    )
    images = {"blank": np.zeros_like(base), "template": base[133:176, 92:135]}
    annotations = []
    for i in range(7):
        name = f"frame_{i:02d}"
        images[name] = cv2.warpAffine(
            base,
            np.array([[1, 0, 2 * i], [0, 1, i]], np.float32),
            (360, 270),
            flags=cv2.INTER_NEAREST,
        )
        annotations.append({"file": name + ".png", "bounds": [92 + 2 * i, 133 + i, 43, 43]})
    return images, annotations


def seal(block, raw, case):
    with tempfile.TemporaryDirectory(prefix="dataflow-vit-") as temporary:
        root = Path(temporary)
        model = root / "input.onnx"
        model.write_bytes(raw)
        block.parameters["model_path"], block.parameters["contract_path"] = str(model), ""
        export_contract(block, root / "bundle", preprocessing=preprocessing(block.parameters))
        for item in (root / "bundle").iterdir():
            asset(HERE / "models" / case / item.name, item.read_bytes())
    block.parameters["model_path"], block.parameters["contract_path"] = (
        "",
        f"models/{case}/contract.json",
    )


def build():
    raw = WEIGHTS.read_bytes()
    assert (
        len(raw) == PREPARATION["SIZE"] and hashlib.sha256(raw).hexdigest() == PREPARATION["SHA256"]
    )
    asset(HERE / "assets/vit.onnx", raw)
    images, annotations = fixtures()
    for name, image in images.items():
        ok, data = cv2.imencode(".png", image)
        assert ok
        asset(HERE / "assets" / (name + ".png"), data.tobytes())
    registry = builtin_registry()
    for case, changes in CASES.items():
        graph = Graph()
        try:
            graph.program["assets"] = ["SOURCES.md", "MODEL-LICENSE.txt"]

            def add(kind, identity, position, **parameters):
                block = graph.add(registry.create(kind, block_id=identity, parameters=parameters))
                block.position = list(position)
                return block

            paths = ["assets/" + a["file"] for a in annotations]
            if case == "loss_expiry":
                paths += ["assets/blank.png"] * 3 + ["assets/frame_00.png"] * 2
            elif case == "detector_reseed":
                paths = paths[:4] + ["assets/blank.png"] * 2 + ["assets/frame_00.png"] * 3
            add(
                "input.file",
                "source",
                (0, 0),
                source_mode="files",
                paths=paths,
                path_base="project",
                color_format="gray" if case == "grayscale" else "bgr",
            )
            p = {"initial_bounds": INITIAL, "warmup": 3, "max_misses": 2, **changes}
            track = add("sequence.track_vit", "track", (600, 0), **p)
            track.node_size = "large"
            seal(track, raw, case)
            graph.connect("source", "image", "track", "image")
            if case == "detector_reseed":
                add(
                    "input.file",
                    "template",
                    (0, 360),
                    path="assets/template.png",
                    path_base="project",
                )
                add("inspect.match_template", "detect", (300, 360), min_score=0.99)
                graph.connect("source", "image", "detect", "image")
                graph.connect("template", "image", "detect", "template")
                graph.connect("detect", "location", "track", "seed")
            if case == "region_seed":
                add("region.shape", "selection", (300, 360), rectangle=INITIAL)
                graph.connect("source", "image", "selection", "image")
                graph.connect("selection", "region", "track", "region")
            note = add("layout.comment", "attribution", (0, -550))
            note.node_size = "large"
            note.name = "Model, motion and score scope"
            note.description = (
                "OpenCV Zoo ViTTrack, Pengyu Liu / Realtime object tracking models, Apache-2.0. "
                "Unchanged model and exact notice included; see SOURCES.md. "
                "T-LESS scene 6/frame 2: T. Hodan et al., WACV 2017, CC BY 4.0, "
                "https://cmp.felk.cvut.cz/t-less/ . Images are resized and artificially "
                "translated with known boxes, not recorded part motion. The score is a native "
                "window-weighted response, not a probability. First activation initializes; "
                "warmup=3 withholds boxes for two activations. Lost/expired phases emit no "
                "current box/pose. Detector reseed explicitly creates a new track; pose angle "
                "follows image axes, not measured object rotation."
            )
            graph.save(HERE / (case + ".json"))
        finally:
            graph.close()
    truth = {
        "cases": list(CASES),
        "annotations": annotations,
        "initial_bounds": INITIAL,
        "warmup": 3,
        "minimum_iou": 0.8,
        "maximum_center_error": 3.0,
        "model_sha256": PREPARATION["SHA256"],
        "asset_sha256": {
            p.relative_to(HERE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in (HERE / "assets", HERE / "models")
            for p in sorted(folder.rglob("*"))
            if p.is_file()
        },
    }
    asset(HERE / "truth.json", (json.dumps(truth, indent=2) + "\n").encode())


if __name__ == "__main__":
    build()
