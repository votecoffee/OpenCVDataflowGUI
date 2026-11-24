"""Build a landmark-verified spline warp of a synthetic inspection plate."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

SIZE = (241, 192)
PROBES = [[27, 29], [75, 63], [115, 115], [179, 40], [205, 155], [162, 133]]


def truth(x, y):
    theta = np.pi * x / (SIZE[0] - 1)
    phi = np.pi * y / (SIZE[1] - 1)
    dx = 10 * np.sin(theta) * np.sin(2 * phi)
    dy = 9 * np.sin(2 * theta) * np.sin(phi)
    return np.stack((np.clip(x + dx, 0, SIZE[0] - 1), np.clip(y + dy, 0, SIZE[1] - 1)), axis=-1)


def inspection_plate():
    width, height = SIZE
    yy, xx = np.indices((height, width), dtype=np.float64)
    gray = 24 + 8 * np.cos(xx / 19) + 7 * np.sin(yy / 13)
    image = np.dstack((gray + 6, gray, gray - 5)).astype(np.uint8)
    cv2.rectangle(image, (20, 16), (221, 175), (125, 141, 158), -1)
    cv2.rectangle(image, (24, 20), (217, 171), (181, 193, 195), 2)
    for x, y in ((34, 30), (206, 30), (34, 161), (206, 161)):
        cv2.circle(image, (x, y), 6, (54, 60, 69), -1)
        cv2.circle(image, (x, y), 9, (212, 219, 222), 1)
    for i, y in enumerate((55, 86, 117, 145)):
        for x in range(55 + i * 4, 204, 29):
            cv2.rectangle(image, (x, y - 4), (x + 9, y + 4), (66, 72, 82), -1)
            cv2.line(image, (x + 11, y), (x + 17, y), (185, 198, 211), 1)
    cv2.putText(image, "A17", (101, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (42, 70, 132), 1)
    return cv2.GaussianBlur(image, (0, 0), 0.65)


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    source = inspection_plate()
    yy, xx = np.indices(source.shape[:2], dtype=np.float64)
    target = cv2.remap(source, truth(xx, yy).astype(np.float32), None, cv2.INTER_LINEAR)
    for name, image in (("source", source), ("target", target)):
        valid, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not valid:
            raise OSError(f"Could not encode {name} image")
        (assets / f"{name}.png").write_bytes(encoded.tobytes())

    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Register a bent inspection plate", 0, -190)
        note.node_size = "large"
        note.description = (
            "Run Once. Fit the warp using 30 control pairs. On the registered image, "
            "green rings mark controls and larger yellow rings mark the six held-out "
            "probes. Compare the warp with the analytically deformed target. "
            "Both images are original MIT assets."
        )
        for name, title, y in (
            ("source", "Original plate", 35),
            ("target", "Known warped target", 275),
        ):
            add(
                "input.file",
                name,
                title,
                0,
                y,
                path=f"assets/{name}.png",
                path_base="project",
            )
        target_points = [[x, y] for y in (0, 48, 96, 144, 191) for x in (0, 48, 96, 144, 192, 240)]
        spline = add(
            "cv.thin_plate_spline",
            "spline",
            "Register plate",
            305,
            75,
            source_points=truth(*np.asarray(target_points).T).tolist(),
            target_points=target_points,
            probe_points=PROBES,
            expected_source_points=truth(*np.asarray(PROBES).T).tolist(),
        )
        spline.exposed_inputs = ["image", "target"]
        graph.connect("source", "image", spline.id, "image")
        graph.connect("target", "image", spline.id, "target")
        add("cv.absdiff", "difference", "Residual to known warp", 645, 75)
        graph.connect(spline.id, "image", "difference", "image_a")
        graph.connect("target", "image", "difference", "image_b")
        validation = validate_graph(graph)
        if not validation["valid"]:
            raise RuntimeError(validation["issues"])
        graph.save(output / "thin_plate_spline.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
