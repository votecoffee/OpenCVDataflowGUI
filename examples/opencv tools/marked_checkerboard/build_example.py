"""Render a three-dot checkerboard and independently positioned partial views."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

_SQUARE = 48
_COLUMNS, _ROWS = 18, 14
_MARKER = (8, 7)


def fixture():
    side = _SQUARE
    full = np.full(((_ROWS + 2) * side, (_COLUMNS + 2) * side), 255, np.uint8)
    for row in range(_ROWS):
        for col in range(_COLUMNS):
            x, y = (col + 1) * side, (row + 1) * side
            shade = 25 if (row + col) % 2 == 0 else 230
            full[y : y + side, x : x + side] = shade
            if (col, row) in (_MARKER, (8, 6), (9, 7)):
                cv2.circle(full, (x + side // 2, y + side // 2), 6, 255 - shade, -1, cv2.LINE_AA)

    crops = {
        "crop_a": (165, 80, 620, 560),
        "crop_b": (285, 155, 625, 530),
        "no_marker": (0, 80, 370, 525),
    }
    views = {"full_board": (full, np.eye(3))}
    for key, (x, y, width, height) in crops.items():
        window = full[y : y + height, x : x + width].copy()
        projection = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]], dtype=float)
        if key == "crop_b":
            projection = np.array([[0, 1, 0], [-1, 0, width - 1], [0, 0, 1]]) @ projection
            source_outline = np.array([[0, 0], [529, 0], [529, 624], [0, 624]], np.float32)
            view_outline = np.array([[17, 9], [516, 0], [522, 603], [0, 614]], np.float32)
            tilt = cv2.getPerspectiveTransform(source_outline, view_outline)
            projection = tilt @ projection
            window = cv2.warpPerspective(full, projection, (530, 625), borderValue=245)
        views[key] = (window, projection)
    origin = (np.array(_MARKER, dtype=float) + 1) * side - 0.5
    return views, origin, side


def build(output):
    views, _, _ = fixture()
    project = output / "marked_checkerboard.json"
    target_paths = {key: output / f"assets/{key}.png" for key in views}
    if project.exists() or any(path.exists() for path in target_paths.values()):
        raise FileExistsError("Use an empty output directory to preserve the example")
    (output / "assets").mkdir(parents=True, exist_ok=True)
    for key, (image, _) in views.items():
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise ValueError(f"Could not encode fixture {key}")
        target_paths[key].write_bytes(encoded.tobytes())

    registry = builtin_registry()
    graph = Graph()
    try:
        note = graph.add(registry.create("layout.comment", block_id="note"))
        note.name = "Anchor a partially visible checkerboard"
        note.position, note.node_size = [0, -240], "large"
        note.description = (
            "Synthetic targets by OpenCVDataflowGUI contributors (MIT). Compare IDs for "
            "the same physical corners in the full, cropped and rotated views. The last "
            "view intentionally crops out the dots and cannot assign physical IDs."
        )
        for row, key in enumerate(views):
            y = row * 228
            source = graph.add(
                registry.create(
                    "input.file",
                    block_id=f"file_{key}",
                    parameters={
                        "path": f"assets/{key}.png",
                        "path_base": "project",
                        "color_format": "gray",
                    },
                )
            )
            source.name, source.position = key.replace("_", " ").capitalize(), [0, y]
            params = {"target_id": "fixture_board", "spacing": 12.0}
            if key == "full_board":
                params.update(
                    allow_larger=False, minimum_columns=_COLUMNS - 1, minimum_rows=_ROWS - 1
                )
            detect = graph.add(
                registry.create("calibration.marked_checkerboard", block_id=key, parameters=params)
            )
            detect.position = [420, y]
            graph.connect(source.id, "image", detect.id, "image")
        report = validate_graph(graph)
        if not report["valid"]:
            raise ValueError(report["issues"])
        graph.save(project)
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
