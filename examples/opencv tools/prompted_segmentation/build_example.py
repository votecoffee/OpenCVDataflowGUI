"""Original exact shape references and prompted segmentation of a credited photograph."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from prepare_prompt_model import PARAMETERS

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
SOURCE = HERE
CASES = {
    "coffee_point": {"points": [[300, 140]], "point_labels": [1]},
    "coffee_box": {"box": [170, 18, 245, 292]},
    "coffee_background": {"points": [[300, 140], [300, 180]], "point_labels": [1, 0]},
    "rectangle_box": {"box": [19, 19, 42, 52]},
    "rectangle_linear": {
        "points": [[40, 45]],
        "point_labels": [1],
        "resampling": "logits_linear",
        "emit_logits": True,
    },
    "quality_fail": {"points": [[300, 140]], "point_labels": [1], "minimum_quality": 2},
    "selected_out": {"points": [[300, 140]], "point_labels": [1]},
    "mismatched_prompts": {"points": [[300, 140]], "point_labels": [1]},
}


def build():
    if any((HERE / (name + ".json")).exists() for name in (*CASES, "truth")):
        raise FileExistsError("Existing prompted examples preserved; generate into a fresh folder")
    photo_hash = hashlib.sha256((SOURCE / "../../assets/coffee.png").read_bytes()).hexdigest()
    if photo_hash != "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7":
        raise ValueError("Photo changed; review attribution first")
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    image = np.full((96, 128, 3), 30, np.uint8)
    image[20:70, 20:60] = (40, 70, 220)
    cv2.circle(image, (95, 48), 18, (220, 80, 30), -1)
    mask = np.zeros((96, 128), np.uint8)
    mask[20:70, 20:60] = 255
    digests = {}
    for name, pixels in (("shapes", image), ("rectangle_truth", mask)):
        ok, raw = cv2.imencode(".png", pixels)
        assert ok
        path = assets / (name + ".png")
        if path.exists() and path.read_bytes() != raw.tobytes():
            raise FileExistsError("Different existing fixture preserved")
        path.write_bytes(raw.tobytes())
        digests[name] = hashlib.sha256(raw).hexdigest()
    registry = builtin_registry()
    for name, options in CASES.items():
        graph = Graph()
        try:
            graph.program["assets"] = ["SOURCES.md", "MODEL-LICENSE.txt"]

            def add(kind, identity, position, **parameters):
                tool = graph.add(registry.create(kind, block_id=identity, parameters=parameters))
                tool.position = list(position)
                return tool

            add(
                "input.file",
                "source",
                (0, 0),
                path="assets/shapes.png"
                if name.startswith("rectangle")
                else "../../assets/coffee.png",
                path_base="project",
            )
            prompt = add(
                "ml.segmentation_prompts",
                "prompts",
                (380, 150),
                **{k: v for k, v in options.items() if k in ("points", "point_labels", "box")},
            )
            prompt.exposed_outputs.append("result_graphics")
            add(
                "ml.efficientsam",
                "infer",
                (780, 0),
                **PARAMETERS,
                contract_path=".local/efficient_sam/contract.json",
            )
            decode = add(
                "ml.decode_prompt_masks",
                "decode",
                (1200, 0),
                **{
                    k: v
                    for k, v in options.items()
                    if k in ("resampling", "emit_logits", "minimum_quality")
                },
            )
            decode.exposed_outputs.append("result_graphics")
            decode.node_size = "large"
            preview = add("flow.select_entry", "candidate_view", (1200, 700), index=1)
            preview.name = "Inspect native candidate"
            preview.description = (
                "Choose index 0..2 to inspect an unthresholded 1024x1024 logit plane."
            )
            graph.connect("infer", "candidates", "candidate_view", "group")
            add("cv.overlay_result_graphics", "overlay", (1650, 0))
            add("output.filmstrip_image", "filmstrip", (2050, 0))
            for a, output, b, input_ in (
                ("source", "image", "prompts", "image"),
                ("source", "image", "infer", "image"),
                ("prompts", "prompts", "infer", "prompts"),
                ("infer", "image", "decode", "image"),
                ("infer", "candidates", "decode", "candidates"),
                ("infer", "qualities", "decode", "qualities"),
                ("infer", "info", "decode", "info"),
                ("decode", "image", "overlay", "image"),
                ("decode", "result_graphics", "overlay", "graphics_1"),
                ("prompts", "result_graphics", "overlay", "graphics_2"),
                ("overlay", "image", "filmstrip", "image"),
            ):
                graph.connect(a, output, b, input_)
            if name == "mismatched_prompts":
                add(
                    "ml.segmentation_prompts",
                    "wrong_prompts",
                    (800, 620),
                    points=[[301, 140]],
                    point_labels=[1],
                )
                graph.connect("source", "image", "wrong_prompts", "image")
                graph.connect("wrong_prompts", "prompts", "decode", "prompts")
            else:
                graph.connect("infer", "prompts", "decode", "prompts")
            if name == "selected_out":
                add("region.shape", "selection", (800, -300), rectangle=[0, 350, 20, 40])
                graph.connect("source", "image", "selection", "image")
                graph.connect("selection", "region", "decode", "region")
            note = add("layout.comment", "attribution", (0, -470))
            note.name, note.node_size = "Model, photograph and reference credits", "large"
            note.description = (
                "EfficientSAM / OpenCV Zoo contributors, Apache 2.0, revision "
                "48b10fa505aeb51fbd127fc60a5c660040e823b7, unchanged April 2025 Ti ONNX. "
                "Coffee: Rachel Michetti, courtesy Pikolo Espresso Bar, scikit-image CC0. "
                "Shapes, exact rectangle reference and pipelines: original MIT "
                "repository material. "
                "Point labels: 1 foreground, 0 background; box is a prompt, not output clipping. "
                "Three native masks remain inspectable; predicted quality is not measured IoU. "
                "Explicit preparation only, no runtime download. See README.md and SOURCES.md."
            )
            graph.save(HERE / (name + ".json"))
        finally:
            graph.close()
    (HERE / "truth.json").write_bytes(
        (
            json.dumps(
                {
                    "cases": CASES,
                    "photo_sha256": photo_hash,
                    "asset_sha256": digests,
                    "rectangle_xywh": [20, 20, 40, 50],
                    "rectangle_pixels": 2000,
                    "reference_method": (
                        "Independent exact construction mask; not sampled from model predictions"
                    ),
                },
                indent=2,
            )
            + "\n"
        ).encode()
    )


if __name__ == "__main__":
    build()
