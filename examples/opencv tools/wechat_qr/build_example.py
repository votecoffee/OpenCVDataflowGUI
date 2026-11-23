"""Rebuild the original QR comparison images and projects."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from prepare_models import MODELS

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

CASES = (
    ("01_tiny.png", (300, 210), 1, 55, 0, [(55, 68, "LOT-A17")]),
    ("02_low_contrast.png", (460, 300), 2, 18, 0.65, [(55, 68, "LOT-A17"), (335, 197, "LOT-B24")]),
)


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    encoder = cv2.QRCodeEncoder_create()
    for name, (width, height), pitch, contrast, blur, labels in CASES:
        image = np.full((height, width), 182, np.uint8)
        for x, y, text in labels:
            code = cv2.resize(
                encoder.encode(text), None, fx=pitch, fy=pitch, interpolation=cv2.INTER_NEAREST
            )
            h, w = code.shape
            image[y : y + h, x : x + w] = np.where(code != 0, 182, 182 - contrast)
        if blur:
            image = cv2.GaussianBlur(image, (0, 0), blur)
        ok, data = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise ValueError(f"Could not encode {name}")
        (assets / name).write_bytes(data.tobytes())

    for use_models in (False, True):
        build_project(output, use_models)


def build_project(output, use_models):
    graph, registry = Graph(), builtin_registry()

    def add(type_id, key, name, x, y, **parameters):
        block = graph.add(registry.create(type_id, block_id=key, parameters=parameters))
        block.name, block.position = name, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Compare QR readers", 0, -265)
        note.node_size = "large"
        note.description = (
            "Run All replays a one-pixel label followed by two low-contrast labels. "
            "Compare each branch's count, decoded text and outline. Original MIT fixtures."
        )
        if use_models:
            note.description += " This project uses the two pinned OpenCV ONNX models."
        add(
            "input.file",
            "source",
            "Small QR labels",
            0,
            40,
            source_mode="files",
            paths=[f"assets/{case[0]}" for case in CASES],
            path_base="project",
            repeat="stop",
        )
        add("inspect.read_qr", "standard", "Standard QR multi", 400, -125, minimum_reads=0)
        add(
            "inspect.read_wechat_qr",
            "without_models",
            "WeChat QR without models",
            400,
            100,
            minimum_reads=0,
        )
        graph.connect("source", "image", "standard", "image")
        graph.connect("source", "image", "without_models", "image")
        if use_models:
            params = {}
            for role, (file, _, digest) in MODELS.items():
                params[f"{role}_model_path"] = f".local/models/{file}"
                params[f"{role}_sha256"] = digest
            add("inspect.read_wechat_qr", "cnn", "WeChat QR with ONNX", 400, 325, **params)
            graph.connect("source", "image", "cnn", "image")
        validation = validate_graph(graph)
        if not validation["valid"]:
            raise ValueError(validation["issues"])
        graph.save(output / ("with_models.json" if use_models else "qr_comparison.json"))
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
