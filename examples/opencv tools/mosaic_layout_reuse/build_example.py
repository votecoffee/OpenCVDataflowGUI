"""Original calibrated views and source-ID replay projects with fixed geometry truth."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.mosaic_layout import MODEL, seal_layout
from opencv_dataflow.mosaic_layout_io import layout_bytes
from opencv_dataflow.payloads import frame

HERE = Path(__file__).resolve().parent
IDS = ("left", "middle", "right")
CASES = {
    "full": [0, 1, 2],
    "reordered": [2, 0, 1],
    "missing_middle": [2, 0],
    "missing_anchor": [2, 1],
    "single_view": [2],
    "missing_error": [0, 2],
    "changed_grid": [0, 1, 2],
    "multiband_tile": [2, 0, 1],
    "native_registration": [0, 1, 2],
}


def images():
    rng = np.random.default_rng(164)
    base = cv2.GaussianBlur(rng.integers(40, 200, (128, 480), np.uint8), (3, 3), 0)
    for x in range(15, 460, 27):
        cv2.circle(base, (x, 20 + (x * 7) % 85), 7, 225, 2)
        cv2.putText(base, str(x), (x - 5, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 35, 1)
    return {
        name + ".png": base[:, i * 144 : i * 144 + 192].copy() + i * 4 for i, name in enumerate(IDS)
    }


def calibration():
    return seal_layout(
        {
            "model": MODEL,
            "image_size": [480, 128],
            "source_identity": {"mode": "explicit", "field": "source_id"},
            "views": [
                {
                    "index": i,
                    "label": name,
                    "source_id": name,
                    "source_size": [192, 128],
                    "source_dtype": "uint8",
                    "source_format": "gray",
                    "source_to_mosaic": [[1, 0, i * 144], [0, 1, 0], [0, 0, 1]],
                    "source_spatial": frame([192, 128], "capture:image"),
                    "source_ticket": None,
                }
                for i, name in enumerate(IDS)
            ],
        }
    )


def build():
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    for name, image in images().items():
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        (assets / name).write_bytes(encoded.tobytes())
    raw = layout_bytes(calibration())
    (assets / "calibration.json").write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    registry = builtin_registry()
    truth = {
        "creator": "OpenCVDataflowGUI contributors",
        "license": "MIT",
        "canvas_size": [480, 128],
        "view_size": [192, 128],
        "offsets": [0, 144, 288],
        "layout_id": calibration()["layout_id"],
        "calibration_sha256": digest,
        "cases": {},
        "asset_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(assets.glob("*.png"))
        },
    }
    for case, order in CASES.items():
        graph = Graph()
        source = graph.add(
            registry.create(
                "input.file",
                block_id="capture",
                parameters={
                    "source_mode": "files",
                    "paths": [f"assets/{IDS[i]}.png" for i in order],
                    "path_base": "project",
                    "color_format": "gray",
                    "pixel_depth": "unchanged",
                    "repeat": "stop",
                },
            )
        )
        source.position = [0, 0]
        batch = graph.add(
            registry.create("flow.batch", block_id="batch", parameters={"size": len(order)})
        )
        batch.position = [430, 0]
        if case != "changed_grid":
            graph.connect("capture", "image", "batch", "item_1")
        renderer = graph.add(
            registry.create(
                "cv.render_mosaic_tile",
                block_id="render",
                parameters={
                    "source_ids": [IDS[i] for i in order],
                    "interpolation": "nearest",
                    "missing_views": "fail"
                    if case == "missing_anchor"
                    else "error"
                    if case == "missing_error"
                    else "allow",
                    "blend": "multiband" if case == "multiband_tile" else "prefer_first",
                    "multiband_bands": 2,
                    "tile_size": 64 if case == "multiband_tile" else 512,
                    "tile_index": 3 if case == "multiband_tile" else 0,
                },
            )
        )
        renderer.position = [900, 0]
        graph.connect("batch", "group", "render", "views")
        if case == "native_registration":
            layout = graph.add(
                registry.create(
                    "cv.register_scan_layout",
                    block_id="layout",
                    parameters={
                        "source_identity": "explicit",
                        "source_ids": list(IDS),
                        "min_inliers": 6,
                    },
                )
            )
            graph.connect("batch", "group", "layout", "views")
        else:
            layout = graph.add(
                registry.create(
                    "cv.load_mosaic_layout",
                    block_id="layout",
                    parameters={"path": "assets/calibration.json", "expected_sha256": digest},
                )
            )
        layout.position = [450, 390]
        graph.connect("layout", "layout", "render", "layout")
        if case == "changed_grid":
            spatial = frame([192, 128], "capture:image")
            spatial["to_root"][0][2] = 1
            changed = graph.add(
                registry.create(
                    "core.attach_payload",
                    block_id="changed_grid",
                    parameters={"payload": {"spatial": spatial}},
                )
            )
            changed.position = [210, 190]
            graph.connect("capture", "image", changed.id, "value")
            graph.connect(changed.id, "value", "batch", "item_1")
        if case == "multiband_tile":
            region = graph.add(
                registry.create(
                    "region.shape",
                    block_id="selection",
                    parameters={
                        "shape": "circle",
                        "center": [96, 64],
                        "radius": 10,
                        "invert": True,
                    },
                )
            )
            region.position = [0, 410]
            masks = graph.add(
                registry.create("flow.batch", block_id="masks", parameters={"size": len(order)})
            )
            masks.position = [430, 730]
            graph.connect("capture", "image", region.id, "image")
            graph.connect(region.id, "mask", "masks", "item_1")
            graph.connect("masks", "group", "render", "masks")
        note = graph.add(registry.create("layout.comment", block_id="attribution"))
        note.name, note.position, note.node_size = (
            "Original fixed-grid replay fixture",
            [0, -430],
            "large",
        )
        note.description = (
            "Original MIT images, layout geometry and projects by OpenCVDataflowGUI contributors. "
            "No customer/third-party images. Three 192x128 views occupy known offsets 0/144/288 "
            "on a 480x128 canvas. IDs map received images to calibration records; replay order "
            "does not change source blend priority. Missing regions remain uncovered; missing "
            "the anchor does not move the canvas. Layout bytes and geometric identity are pinned. "
            "Missing-anchor is a deliberate failed partial, missing-error emits nothing, and "
            "changed-grid rejects a one-pixel crop-context change. Multiband renders a masked "
            "bounded tile. Native-registration fits a new layout as a separate workflow; frozen "
            "replay performs no feature matching. IDs/grid metadata assert identity, not physical "
            "camera correctness or exposure synchronization."
        )
        graph.save(HERE / f"{case}.json")
        graph.close()
        truth["cases"][case] = {
            "order": order,
            "runs": len(order),
            "expected_state": "fail"
            if case == "missing_anchor"
            else "error"
            if case in ("changed_grid", "missing_error")
            else "pass",
        }
    (HERE / "truth.json").write_bytes((json.dumps(truth, indent=2) + "\n").encode())


if __name__ == "__main__":
    build()
