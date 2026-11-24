"""Recreate a translated image burst with fine features and known high-resolution truth."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

SHIFTS = [[0.5, 0.0], [0.0, 0.5], [0.0, 0.0], [0.5, 0.5], [-0.5, 0.0]]


def fixture():
    rng = np.random.default_rng(8103)
    y, x = np.indices((144, 192), np.float32)
    truth = 52 + 14 * np.sin(x / 5) + 9 * np.cos(y / 6) + 8 * np.cos((x + y) / 8)
    truth = np.uint8(np.rint(truth))
    cv2.rectangle(truth, (56, 35), (141, 111), 184, -1)
    for index, col in enumerate(range(65, 138, 6)):
        cv2.line(truth, (col, 45 + 2 * (index % 3)), (col, 101 - 2 * (index % 2)), 34 if index % 2 == 0 else 67, 1)
    cv2.circle(truth, (32, 101), 13, 219, -1)
    cv2.circle(truth, (158, 24), 10, 201, -1)
    cv2.rectangle(truth, (88, 13), (126, 27), 123, -1)
    cv2.line(truth, (20, 58), (48, 19), 141, 3)
    cv2.line(truth, (144, 102), (178, 122), 218, 4)
    frames = []
    for dx, dy in SHIFTS:
        transform = np.float32([[1, 0, 2 * dx], [0, 1, 2 * dy]])
        shifted = cv2.warpAffine(
            truth, transform, (192, 144), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE,
        )
        pixels = cv2.resize(shifted, (96, 72), interpolation=cv2.INTER_AREA)
        faded = np.rint(pixels.astype(float) + rng.normal(0, 0.55, pixels.shape))
        frames.append(faded.clip(0, 255).astype(np.uint8))
    return truth, frames


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    truth, frames = fixture()
    paths = [f"assets/frame_{i:02}.png" for i in range(1, 6)]
    for relative, pixels in zip(["assets/truth.png", *paths], [truth, *frames], strict=True):
        ok, data = cv2.imencode(".png", pixels, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise OSError(f"Could not encode {relative}")
        (output / relative).write_bytes(data.tobytes())

    registry = builtin_registry()
    for mode in ("estimate", "provided"):
        graph = Graph()

        def add(kind, key, title, x, y, **parameters):
            block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
            block.name, block.position = title, [x, y]
            return block

        try:
            note = add("layout.comment", "note", "Recover fine marks from a short burst", 0, -223)
            note.node_size = "large"
            note.description = (
                "Choose Run all for five translated frames. Reconstruct 2× around frame 03 "
                "and compare to single-frame bicubic upscaling, using the original high-resolution "
                "target. The measurements exclude the outer eight pixels. Original MIT fixture."
            )
            add(
                "input.file", "source", "Replay five translated observations", 0, 45,
                path_base="project", source_mode="files", paths=paths,
                repeat="stop", filename_ticket=True, color_format="gray",
            )
            batch = add("flow.batch", "batch", "Collect five frames", 260, 60, size=5)
            batch.exposed_inputs = ["item_1"]
            add(
                "sequence.multi_frame_super_resolution", "reconstruct", "Reconstruct around frame 03",
                540, 65, alignment=mode, shifts=SHIFTS if mode == "provided" else [],
                minimum_phase_bins=4, minimum_frames=4,
            )
            add(
                "input.file", "truth", "Known high-resolution target", 0, 310,
                path="assets/truth.png", path_base="project", color_format="gray",
            )
            add(
                "region.shape", "area", "Common interior measurement", 265, 310,
                rectangle=[8, 8, 176, 128],
            )
            graph.connect("source", "image", "batch", "item_1")
            graph.connect("batch", "group", "reconstruct", "frames")
            graph.connect("truth", "image", "area", "image")
            for index, (port, name) in enumerate((("image", "Burst"), ("baseline", "Bicubic"))):
                tool = add(
                    "inspect.image_quality", port + "_rmse", f"{name} error against truth",
                    850, 80 + index * 235, metric="rmse", mask_alignment="pixel_grid",
                    check_threshold=(port == "image"), threshold=6.0,
                )
                graph.connect("reconstruct", port, tool.id, "image_a")
                graph.connect("truth", "image", tool.id, "image_b")
                graph.connect("area", "mask", tool.id, "mask")
            checked = validate_graph(graph)
            if not checked["valid"]:
                raise RuntimeError(checked["issues"])
            graph.save(output / f"super_resolution_{mode}.json")
        finally:
            graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
