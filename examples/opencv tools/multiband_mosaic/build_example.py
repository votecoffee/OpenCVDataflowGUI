"""Credited VisA PCB crops with controlled exposure/depth for multiband replay."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "assets/showcase/visa-pcb2-0000.png"
CREDIT = (
    "Photo: Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang and Onkar Dabeer, "
    "SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation, "
    "ECCV 2022. VisA PCB2 normal 0000, CC BY 4.0. "
    "https://registry.opendata.aws/visa/ ; https://creativecommons.org/licenses/by/4.0/ . "
    "OpenCVDataflowGUI contributors crop the bundled 480x320 resized/letterboxed photograph "
    "into overlapping views, apply known darkening gains and expand copies to uint16. "
    "The original capture is 8-bit; uint16 expansion adds no sensor information. "
    "This demonstrates numerical blending on real photographic texture, not separate "
    "physical acquisitions or calibrated illumination. See SOURCES.md and truth.json."
)


def build():
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    photo = cv2.imread(str(SOURCE), cv2.IMREAD_COLOR)
    assert photo.shape == (320, 480, 3)
    origins, gains = [0, 110, 220], [1.0, 0.85, 0.70]
    for i, (x, gain) in enumerate(zip(origins, gains, strict=True)):
        view = np.rint(photo[:, x : x + 260].astype(float) * gain).astype(np.uint8)
        for depth, image in ((8, view), (16, view.astype(np.uint16) * 257)):
            ok, encoded = cv2.imencode(".png", image)
            assert ok
            (assets / f"view-{i + 1}-{depth}.png").write_bytes(encoded.tobytes())
    registry = builtin_registry()
    cases = {
        "pair_float64": ("pair", 16, "float64"),
        "pair_native_detail": ("pair", 8, "native_detail"),
        "pair_native_int16_weights": ("pair", 8, "native_detail"),
        "scan_float64": ("scan", 16, "float64"),
        "scan_exposure_gain": ("scan", 8, "float64"),
        "tiles_float64": ("tile", 16, "float64"),
        "native_depth_guard": ("pair", 16, "native_detail"),
    }
    for case, (mode, depth, backend) in cases.items():
        graph = Graph()

        def add(kind, name, position, **parameters):
            block = graph.add(registry.create(kind, block_id=name, parameters=parameters))
            block.position = list(position)
            return block

        shared = {"blend": "multiband", "multiband_backend": backend, "multiband_bands": 3}
        if case.endswith("int16_weights"):
            shared["multiband_native_weights"] = "int16"
        if mode == "pair":
            for i, name in enumerate(("left", "right")):
                add(
                    "input.file",
                    name,
                    (0, i * 350),
                    path=f"assets/view-{i + 1}-{depth}.png",
                    path_base="project",
                    pixel_depth="unchanged",
                    color_format="bgr",
                )
            add(
                "cv.stitch_image_pair",
                "mosaic",
                (700, 80),
                transform_source="matrix",
                matrix=[[1, 0, 110], [0, 1, 0], [0, 0, 1]],
                ticket_policy="anchor_a",
                **shared,
            )
            graph.connect("left", "image", "mosaic", "image_a")
            graph.connect("right", "image", "mosaic", "image_b")
        else:
            add(
                "input.file",
                "source",
                (0, 0),
                source_mode="files",
                paths=[f"assets/view-{i + 1}-{depth}.png" for i in range(3)],
                repeat="stop",
                filename_ticket=True,
                pixel_depth="unchanged",
                color_format="bgr",
                path_base="project",
            )
            add("flow.batch", "batch", (330, 0), size=3)
            graph.connect("source", "image", "batch", "item_1")
            if mode == "scan":
                add(
                    "cv.stitch_scan_sequence",
                    "mosaic",
                    (700, 0),
                    exposure="gain" if case == "scan_exposure_gain" else "none",
                    **shared,
                )
                graph.connect("batch", "group", "mosaic", "views")
            else:
                add("cv.register_scan_layout", "layout", (660, -180))
                graph.connect("batch", "group", "layout", "views")
                for i, index in enumerate((0, 5, 11)):
                    name = f"tile_{index}"
                    add(
                        "cv.render_mosaic_tile",
                        name,
                        (1040, i * 320 - 300),
                        tile_size=128,
                        tile_index=index,
                        **shared,
                    )
                    graph.connect("batch", "group", name, "views")
                    graph.connect("layout", "layout", name, "layout")
        note = add("layout.comment", "scope", (0, -430))
        note.name, note.node_size = "Source and expected result", "large"
        note.description = CREDIT + (
            " EXPECT VALIDATION ERROR: native detail only accepts uint8; select Float64 "
            "to process these uint16 views. No silent conversion to eight bits is allowed."
            if case == "native_depth_guard"
            else " EXPECT PASS. Pair projects run once; scan/tile projects use Run all. "
            "Coverage and overlap remain authoritative. Native CPU arithmetic is approximate; "
            "Float64 preserves high-depth signal until final output rounding. View-only pixels "
            "are restored exactly. Tiled filtering uses a phase-aligned halo."
        )
        graph.save(HERE / f"{case}.json")
        graph.close()
    truth = {
        "source": "VisA PCB2 normal 0000; see SOURCES.md",
        "source_image_size": [480, 320],
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "view_xywh": [[x, 0, 260, 320] for x in origins],
        "view_gains": gains,
        "gain_formula": "round(uint8 crop * gain)",
        "uint16_formula": "uint8 derivative * 257",
        "nominal_pair_size": [370, 320],
        "nominal_scan_size": [480, 320],
        "registration": "Pair uses exact known translation; scans estimate from image features",
        "scope": "Software changes to one real photo, not physical scan/illumination ground truth",
        "derivative_license": "CC BY 4.0",
        "cases": {k: list(v) for k, v in cases.items()},
        "asset_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(assets.glob("*.png"))
        },
    }
    (HERE / "truth.json").write_bytes((json.dumps(truth, indent=2) + "\n").encode())


if __name__ == "__main__":
    build()
