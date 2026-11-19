"""Build independent known proposal controls for native spatial voting."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import ProjectSession, validate_graph

HERE = Path(__file__).resolve().parent
WIDTH, HEIGHT = 112, 80
BOXES = [
    [10, 10, 30, 30],
    [11, 10, 31, 30],
    [10, 11, 30, 31],
    [32, 10, 52, 30],
    [33, 10, 53, 30],
    [10, 10, 30, 30],
    [10, 10, 30, 30],
    [65, 40, 80, 55],
]
CLASSES = [0, 0, 0, 0, 0, 1, 1, 0]
CASES = {
    "grouped": ({}, [[0, 1, 2], [3, 4], [5, 6]]),
    "keep_singletons": ({"singleton_policy": "keep"}, [[0, 1, 2], [3, 4], [7], [5, 6]]),
    "exact_duplicates": ({"epsilon": 0}, [[5, 6]]),
    "minimum_three": ({"minimum_votes": 3}, [[0, 1, 2]]),
    "data_only": ({}, [[0, 1, 2], [3, 4], [5, 6]]),
    "count_failure": ({"minimum_votes": 4}, []),
}


def model():
    import onnx
    from onnx import TensorProto as T
    from onnx import helper as h
    from onnx import numpy_helper as n

    count = len(BOXES)
    initial = [
        n.from_array(np.float32(BOXES)[None], "boxes"),
        n.from_array(np.linspace(0.8, 0.45, count, dtype=np.float32)[None], "bias"),
        n.from_array(np.float32([[0.02] * count, [0] * count, [0] * count]), "weights"),
        n.from_array(np.int64([1, count, 1]), "column"),
        n.from_array(np.float32(CLASSES).reshape(1, count, 1), "classes"),
    ]
    nodes = [
        h.make_node("GlobalAveragePool", ["image"], ["pool"]),
        h.make_node("Flatten", ["pool"], ["rgb"]),
        h.make_node("MatMul", ["rgb", "weights"], ["adjustment"]),
        h.make_node("Add", ["adjustment", "bias"], ["scores"]),
        h.make_node("Reshape", ["scores", "column"], ["confidence"]),
        h.make_node("Concat", ["boxes", "confidence", "classes"], ["rows"], axis=2),
    ]
    result = h.make_model(
        h.make_graph(
            nodes,
            "original_spatial_vote_controls",
            [h.make_tensor_value_info("image", T.FLOAT, [1, 3, HEIGHT, WIDTH])],
            [h.make_tensor_value_info("rows", T.FLOAT, [1, count, 6])],
            initial,
        ),
        opset_imports=[h.make_opsetid("", 13)],
        ir_version=8,
    )
    onnx.checker.check_model(result)
    return result.SerializeToString()


def add(kind, identity, **parameters):
    return {"op": "add", "type": kind, "id": identity, "parameters": parameters}


def connect(source, output, target, input_port):
    return {
        "op": "connect",
        "source": source,
        "output": output,
        "target": target,
        "input": input_port,
    }


def save(folder, name, parameters):
    operations = [
        add("input.file", "source", path="assets/scene.png", path_base="project"),
        add(
            "ml.dnn",
            "infer",
            model_path="assets/proposals.onnx",
            engine="new",
            width=WIDTH,
            height=HEIGHT,
            input_name="image",
        ),
        add(
            "ml.decode_detections",
            "decode",
            format="xyxy_score_class",
            output_name="rows",
            labels=["part", "fixture"],
            nms_enabled=False,
            confidence_threshold=0.0,
        ),
        add("ml.group_detection_votes", "vote", **parameters),
        add("layout.comment", "attribution"),
        {
            "op": "update",
            "id": "attribution",
            "fields": {
                "name": "Synthetic spatial votes (MIT)",
                "description": "Original synthetic scene and prescribed ONNX proposals. "
                "Two adjacent parts produce three and two votes. Two coincident "
                "fixture votes stay separate by class. One isolated vote is optional.",
            },
        },
        connect("source", "image", "infer", "image"),
        connect("source", "image", "decode", "image"),
        connect("infer", "tensors", "decode", "tensors"),
        connect("infer", "preprocessing", "decode", "preprocessing"),
        connect("decode", "collection", "vote", "collection"),
    ]
    if name != "data_only":
        operations.append(connect("source", "image", "vote", "image"))
    with ProjectSession(directory=folder) as session:
        session.apply([*operations, {"op": "arrange"}])
        validation = validate_graph(session.graph)
        if not validation["valid"]:
            raise ValueError(validation["issues"])
        session.save(folder / f"{name}.json")


def build(folder):
    assets = folder / "assets"
    assets.mkdir(parents=True)
    image = np.full((HEIGHT, WIDTH, 3), [38, 48, 58], np.uint8)
    for a, b, color in [
        ((10, 10), (29, 29), (90, 160, 200)),
        ((32, 10), (51, 29), (90, 160, 200)),
        ((65, 40), (79, 54), (140, 160, 90)),
    ]:
        cv2.rectangle(image, a, b, color, -1)
        cv2.line(image, a, b, (220, 220, 220), 1)
    assert cv2.imwrite(str(assets / "scene.png"), image)
    (assets / "proposals.onnx").write_bytes(model())
    results = {}
    for name, (parameters, members) in CASES.items():
        save(folder, name, parameters)
        with ProjectSession.open(folder / f"{name}.json") as session:
            outcome = session.run()
            assert outcome["passed"] == bool(members)
            block = session.graph.blocks["vote"]
            items = block.last_result.outputs["collection"]["items"]
            assert [item["source_ids"] for item in items] == members, name
            results[name] = {
                "state": block.state,
                "members": members,
                "bbox": [item["geometry"]["bbox"] for item in items],
                "vote_counts": [item["vote_count"] for item in items],
            }
    truth = {
        "schema": "spatial_vote_fixture_v1",
        "source_boxes_xyxy": BOXES,
        "classes": CLASSES,
        "results": results,
        "assets_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(assets.iterdir())
        },
    }
    (folder / "truth.json").write_text(
        json.dumps(truth, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({name: item["vote_counts"] for name, item in results.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    build(parser.parse_args().output.resolve())
