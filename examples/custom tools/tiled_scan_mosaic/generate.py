"""Reproduce the original synthetic scan and its editable example project."""

from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def main():
    folder = Path(__file__).resolve().parent
    assets = folder / "assets"
    assets.mkdir(exist_ok=True)
    height, view_width, step, count = 144, 1800, 1550, 12
    width = view_width + (count - 1) * step
    rng = np.random.default_rng(1024)
    scene = cv2.GaussianBlur(rng.integers(0, 256, (height, width), np.uint8), (3, 3), 0)
    for x in range(10, width, 31):
        cv2.circle(scene, (x, 32 + x % 65), 5 + x % 7, 240, 2)
    names = []
    for i in range(count):
        name = f"tile-{i + 1:02d}.png"
        image = scene[:, i * step : i * step + view_width]
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("Unable to encode synthetic scan")
        (assets / name).write_bytes(encoded.tobytes())
        names.append(f"assets/{name}")
    registry = builtin_registry()
    graph = Graph()
    source = graph.add(
        registry.create(
            "input.file",
            block_id="source",
            parameters={
                "source_mode": "files",
                "paths": names,
                "path_base": "project",
                "repeat": "stop",
                "color_format": "gray",
                "pixel_depth": "uint8",
            },
        )
    )
    source.name = "Twelve overlapping inspection views"
    batch = graph.add(registry.create("flow.batch", block_id="batch", parameters={"size": count}))
    batch.position = [300.0, 0.0]
    register = graph.add(registry.create("cv.register_scan_layout", block_id="register"))
    register.position = [600.0, 0.0]
    graph.connect(source.id, "image", batch.id, "item_1")
    graph.connect(batch.id, "group", register.id, "views")
    for i, (name, index) in enumerate((("first_tile", 0), ("middle_tile", 18), ("last_tile", 36))):
        block = graph.add(
            registry.create(
                "cv.render_mosaic_tile",
                block_id=name,
                parameters={"tile_index": index, "interpolation": "nearest", "blend": "feather"},
            )
        )
        block.name = f"Inspect mosaic tile {index}"
        block.position = [930.0, i * 260.0]
        block.exposed_inputs = ["views", "layout"]
        block.exposed_outputs = ["image", "coverage", "overlap", "tile"]
        graph.connect(batch.id, "group", block.id, "views")
        graph.connect(register.id, "layout", block.id, "layout")
    note = graph.add(registry.create("layout.comment", block_id="fixture_note"))
    note.name = "Original synthetic fixture"
    note.description = (
        "Original synthetic image strip and project, provided under this repository's MIT "
        "license. Run all 12 views. Registration computes a layout wider than 16,384 pixels; "
        "each renderer produces one 512-pixel tile. Change Tile index to inspect other windows. "
        "The tile payload maps measurements back to the first source frame."
    )
    note.position = [0.0, -270.0]
    note.node_size = "large"
    graph.save(folder / "tiled_scan_mosaic.json")
    graph.close()


if __name__ == "__main__":
    main()
