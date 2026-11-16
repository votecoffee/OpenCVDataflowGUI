"""Original fixed-grid captures for independent frame replay and crop/support checks."""

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
    "union": {"crop": "union"},
    "intersection": {"crop": "intersection", "masks": True},
    "native_union": {"crop": "union", "backend": "native_detail"},
    "native_intersection": {"crop": "intersection", "backend": "native_detail", "masks": True},
    "reordered_missing": {"crop": "canvas", "order": [2, 1]},
    "depth16": {"crop": "intersection", "depth": True},
    "no_common": {"crop": "intersection", "disjoint": True, "expected_state": "fail"},
    "budget_guard": {"crop": "canvas", "maximum_workspace_mib": 1, "expected_state": "error"},
    "native_depth_guard": {
        "crop": "union",
        "backend": "native_detail",
        "depth": True,
        "expected_state": "error",
    },
}


def images():
    rng = np.random.default_rng(165)
    texture = cv2.GaussianBlur(rng.integers(25, 210, (128, 288), np.uint8), (3, 3), 0)
    for x in range(18, 270, 26):
        cv2.circle(texture, (x, 24 + x % 75), 7, 230, 2)
        cv2.putText(texture, str(x), (x - 5, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 30, 1)
    result = {}
    for i, name in enumerate(IDS):
        gray = texture[:, i * 48 : i * 48 + 192].copy() + i * 3
        result[name + ".png"] = np.dstack((gray, gray + 2, gray + 4))
        result[name + "16.png"] = gray.astype(np.uint16) * 230
    return result


def calibration(depth=False, disjoint=False):
    return seal_layout(
        {
            "model": MODEL,
            "image_size": [576 if disjoint else 288, 128],
            "source_identity": {"mode": "explicit", "field": "source_id"},
            "views": [
                {
                    "index": i,
                    "label": name,
                    "source_id": name,
                    "source_size": [192, 128],
                    "source_dtype": "uint16" if depth else "uint8",
                    "source_format": "gray" if depth else "bgr",
                    "source_to_mosaic": [
                        [1, 0, i * (192 if disjoint else 48)],
                        [0, 1, 0],
                        [0, 0, 1],
                    ],
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
    for name, layout in (
        ("calibration", calibration()),
        ("depth16", calibration(depth=True)),
        ("disjoint", calibration(disjoint=True)),
    ):
        (assets / (name + ".json")).write_bytes(layout_bytes(layout))
    registry = builtin_registry()
    truth = {
        "creator": "OpenCVDataflowGUI contributors",
        "license": "MIT",
        "view_size": [192, 128],
        "offsets": [0, 48, 96],
        "cases": {},
    }
    for case, settings in CASES.items():
        graph, order = Graph(), settings.get("order", [0, 1, 2])
        depth = settings.get("depth", False)
        names = [IDS[i] + ("16" if depth else "") + ".png" for i in order]
        source = graph.add(
            registry.create(
                "input.file",
                block_id="capture",
                parameters={
                    "source_mode": "files",
                    "paths": ["assets/" + name for name in names],
                    "path_base": "project",
                    "color_format": "gray" if depth else "bgr",
                    "pixel_depth": "unchanged",
                    "repeat": "stop",
                },
            )
        )
        source.position = [0, 0]
        batch = graph.add(
            registry.create("flow.batch", block_id="batch", parameters={"size": len(order)})
        )
        batch.position = [420, 0]
        graph.connect("capture", "image", "batch", "item_1")
        layout_name = (
            "disjoint" if settings.get("disjoint") else "depth16" if depth else "calibration"
        )
        layout_path = assets / (layout_name + ".json")
        layout = graph.add(
            registry.create(
                "cv.load_mosaic_layout",
                block_id="layout",
                parameters={
                    "path": "assets/" + layout_path.name,
                    "expected_sha256": hashlib.sha256(layout_path.read_bytes()).hexdigest(),
                },
            )
        )
        layout.position = [420, 370]
        replay = graph.add(
            registry.create(
                "cv.replay_registered_views",
                block_id="replay",
                parameters={
                    "source_ids": [IDS[i] for i in order],
                    "missing_views": "allow",
                    "crop": settings["crop"],
                    "backend": settings.get("backend", "preserve"),
                    "interpolation": "nearest",
                    "maximum_workspace_mib": settings.get("maximum_workspace_mib", 256),
                },
            )
        )
        replay.position = [900, 0]
        graph.connect("batch", "group", "replay", "views")
        graph.connect("layout", "layout", "replay", "layout")
        if settings.get("masks"):
            region = graph.add(
                registry.create(
                    "region.shape",
                    block_id="selection",
                    parameters={
                        "shape": "circle",
                        "center": [96, 64],
                        "radius": 14,
                        "invert": True,
                    },
                )
            )
            region.position = [0, 400]
            masks = graph.add(
                registry.create("flow.batch", block_id="masks", parameters={"size": len(order)})
            )
            masks.position = [420, 730]
            graph.connect("capture", "image", region.id, "image")
            graph.connect(region.id, "mask", masks.id, "item_1")
            graph.connect(masks.id, "group", replay.id, "masks")
        serial = graph.add(registry.create("flow.unbatch", block_id="serial"))
        serial.position = [1360, 0]
        graph.connect(replay.id, "frames", serial.id, "group")
        if settings["crop"] == "intersection":
            measure = graph.add(
                registry.create(
                    "cv.image_statistics",
                    block_id="measure",
                    parameters={"percentiles": [5, 50, 95]},
                )
            )
            measure.position = [1770, 0]
            graph.connect(serial.id, "value", measure.id, "image")
            graph.connect(replay.id, "common", measure.id, "mask")
        note = graph.add(registry.create("layout.comment", block_id="attribution"))
        note.name, note.position, note.node_size = (
            "Original registered replay fixture",
            [0, -420],
            "large",
        )
        note.description = (
            "Original MIT images, known transforms and projects by OpenCVDataflowGUI contributors; "
            "no customer/third-party images. Three 192x128 captures occupy known x offsets 0/48/96 "
            "on a 288x128 canvas. Replay keeps each view independent with its original filename "
            "ticket, support and coordinate mapping. Intersection uses the exact common-valid "
            "support, including source-mask holes; Unbatch sends each image to an inspection. "
            "Native detail uses the factory and exact uint8 placement with masks enforced by "
            "the wrapper. Depth16 demonstrates preserved high-depth samples; native-depth and "
            "budget guards reject before warping. No-common is a deliberate failed replay using "
            "disjoint saved geometry (no physical timing/calibration claim). Source IDs/metadata "
            "assert identity, not synchronized exposures or physical camera correctness."
        )
        graph.save(HERE / (case + ".json"))
        graph.close()
        truth["cases"][case] = {
            "order": order,
            "runs": len(order),
            "filenames": names,
            "layout": layout_path.name,
            "crop": settings["crop"],
            "backend": settings.get("backend", "preserve"),
            "masked": settings.get("masks", False),
            "expected_state": settings.get("expected_state", "pass"),
        }
    truth["asset_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(assets.iterdir())
        if p.is_file()
    }
    (HERE / "truth.json").write_bytes((json.dumps(truth, indent=2) + "\n").encode())


if __name__ == "__main__":
    build()
