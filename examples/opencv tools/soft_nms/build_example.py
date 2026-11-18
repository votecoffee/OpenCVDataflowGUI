"""Original image-dependent proposal controls; no external models or image downloads."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
BOXES = [
    [10.2, 10.1, 32.2, 30.1],
    [12.2, 10.1, 34.2, 30.1],
    [30.2, 10.1, 52.2, 30.1],
    [10.2, 10.1, 32.2, 30.1],
    [-4.2, 35.2, 16.2, 55.2],
    [42.2, 40.2, 54.2, 52.2],
]
BASE_SCORES = [0.9, 0.8, 0.7, 0.85, 0.6, 0.2]
CLASSES = [0, 0, 0, 1, 0, 0]
CASES = {
    "linear": {},
    "gaussian": {"method": "gaussian"},
    "hard": {"method": "hard"},
    "none": {"method": "none"},
    "agnostic": {"class_agnostic": True},
    "top_k": {"top_k": 2},
    "selected": {},
    "empty": {"score_threshold": 0.99},
    "count_failure": {"minimum_count": 7},
    "high_depth": {},
    "nearest": {"rounding": "nearest"},
    "visible_boxes": {"box_source": "bbox"},
    "details_off": {},
}


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def model():
    import onnx
    from onnx import TensorProto as T
    from onnx import helper as h
    from onnx import numpy_helper as n

    # Source red mean changes scores; boxes/classes remain explicit known proposals.
    initial = [
        n.from_array(np.float32(BOXES)[None], "boxes"),
        n.from_array(np.float32(BASE_SCORES)[None], "bias"),
        n.from_array(np.float32([[0.02] * 6, [0] * 6, [0] * 6]), "weights"),
        n.from_array(np.int64([1, 6, 1]), "column"),
        n.from_array(np.float32(CLASSES).reshape(1, 6, 1), "classes"),
    ]
    nodes = [
        h.make_node("GlobalAveragePool", ["image"], ["pool"]),
        h.make_node("Flatten", ["pool"], ["rgb"], axis=1),
        h.make_node("MatMul", ["rgb", "weights"], ["adjustment"]),
        h.make_node("Add", ["adjustment", "bias"], ["scores"]),
        h.make_node("Reshape", ["scores", "column"], ["confidence"]),
        h.make_node("Concat", ["boxes", "confidence", "classes"], ["rows"], axis=2),
    ]
    result = h.make_model(
        h.make_graph(
            nodes,
            "original_soft_nms_controls",
            [h.make_tensor_value_info("image", T.FLOAT, [1, 3, 64, 64])],
            [h.make_tensor_value_info("rows", T.FLOAT, [1, 6, 6])],
            initial,
        ),
        opset_imports=[h.make_opsetid("", 13)],
        ir_version=8,
    )
    onnx.checker.check_model(result)
    return result.SerializeToString()


def build(destination=HERE):
    destination = Path(destination)
    assets = destination / "assets"
    if assets.exists():
        raise FileExistsError("Existing Soft-NMS assets preserved")
    assets.mkdir(parents=True)
    (destination / ".gitattributes").write_bytes(b"*.json -text\n.gitattributes -text\n")
    (assets / "proposals.onnx").write_bytes(model())
    image = np.full((64, 64, 3), [45, 60, 75], np.uint8)
    for a, b, c in [
        ((10, 10), (31, 29), (100, 160, 210)),
        ((30, 10), (51, 29), (150, 130, 190)),
        ((0, 35), (15, 54), (120, 140, 180)),
        ((42, 40), (53, 51), (180, 160, 130)),
    ]:
        cv2.rectangle(image, a, b, c, -1)
        cv2.line(image, a, b, (230, 230, 230), 1)
    (assets / "parts.png").write_bytes(cv2.imencode(".png", image)[1].tobytes())
    registry = builtin_registry()
    for case, settings in CASES.items():
        graph = Graph()
        try:

            def add(kind, key, x, y=0, **params):
                b = graph.add(registry.create(kind, block_id=key, parameters=params))
                b.position = [x, y]
                return b

            add(
                "input.file",
                "source",
                0,
                path="assets/parts.png",
                path_base="project",
                pixel_depth="uint16" if case == "high_depth" else "unchanged",
            )
            add(
                "ml.dnn",
                "infer",
                360,
                220,
                model_path="assets/proposals.onnx",
                engine="new",
                input_name="image",
                width=64,
                height=64,
                scale=1 / 65535 if case == "high_depth" else 1 / 255,
            )
            add(
                "ml.decode_detections",
                "decode",
                720,
                format="xyxy_score_class",
                output_name="rows",
                labels=["part", "other class"],
                nms_enabled=False,
                confidence_threshold=0.01,
            )
            suppress = add(
                "ml.soft_nms",
                "suppress",
                1080,
                include_candidate_details=case != "details_off",
                **({"score_threshold": 0.1} | settings),
            )
            suppress.exposed_outputs.append("result_graphics")
            add("cv.overlay_result_graphics", "overlay", 1440)
            add("output.filmstrip_image", "filmstrip", 1800)
            for a, o, b, i in [
                ("source", "image", "infer", "image"),
                ("source", "image", "decode", "image"),
                ("infer", "tensors", "decode", "tensors"),
                ("infer", "preprocessing", "decode", "preprocessing"),
                ("source", "image", "suppress", "image"),
                ("decode", "collection", "suppress", "collection"),
                ("suppress", "image", "overlay", "image"),
                ("suppress", "result_graphics", "overlay", "graphics_1"),
                ("overlay", "image", "filmstrip", "image"),
            ]:
                graph.connect(a, o, b, i)
            if case == "selected":
                add("region.shape", "selection", 720, 500, rectangle=[28, 0, 36, 64])
                graph.connect("source", "image", "selection", "image")
                graph.connect("selection", "region", "suppress", "region")
            note = add("layout.comment", "attribution", 0, -460)
            note.name, note.node_size = "Original Soft-NMS controls - MIT", "large"
            note.description = (
                "Original OpenCVDataflowGUI synthetic image, image-dependent ONNX proposals "
                "and projects, MIT. Six known proposals include a shifted duplicate, an adjacent "
                "part, an overlapping second class and a border-crossing box. Integer native "
                "suppression boxes are recorded while original fractional geometry is preserved. "
                "This tests postprocessing, not learned part recognition. See README.md/SOURCES.md."
            )
            graph.save(destination / (case + ".json"))
        finally:
            graph.close()
    (destination / "truth.json").write_bytes(
        encoded(
            {
                "cases": list(CASES),
                "boxes_xyxy": BOXES,
                "base_scores": BASE_SCORES,
                "classes": CLASSES,
                "red_mean_score_factor": 0.02,
                "asset_sha256": {
                    p.relative_to(destination).as_posix(): hashlib.sha256(
                        p.read_bytes()
                    ).hexdigest()
                    for p in sorted(assets.iterdir())
                },
            }
        )
    )


if __name__ == "__main__":
    build()
