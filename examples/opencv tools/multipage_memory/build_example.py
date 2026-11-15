"""Original MIT gray/integer/HDR fixtures for TIFF page transport, without file outputs."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
CASES = (
    "typed_roundtrip",
    "raw_roundtrip",
    "selected_pages",
    "pixel_budget_guard",
    "predictor_guard",
)


def reference_pages():
    y, x = np.indices((48, 72))
    gray = (x * 2 + y).astype(np.uint8)
    cv2.rectangle(gray, (20, 12), (45, 32), 240, -1)
    y, x = np.indices((40, 64))
    color = np.dstack((x * 733, y * 1237, (x + y) * 509)).astype(np.uint16)
    y, x = np.indices((32, 56))
    hdr = np.dstack((x / 17 - 1, y / 11 + 0.5, (x + y) / 13, np.full(x.shape, 0.625)))
    return gray, color, hdr.astype(np.float64)


def build():
    registry = builtin_registry()
    pages = reference_pages()
    ok, data = cv2.imencodemulti(".tiff", pages, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
    assert ok
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    path = assets / "mixed-pages.tiff"
    path.write_bytes(data.tobytes())
    for case in CASES:
        graph = Graph()

        def add(kind, name, position, **parameters):
            block = graph.add(registry.create(kind, block_id=name, parameters=parameters))
            block.position = list(position)
            return block

        add(
            "input.tiff_file", "source", (0, 0), path="assets/mixed-pages.tiff", path_base="project"
        )
        add("flow.batch", "batch", (340, 0), size=3)
        add(
            "cv.encode_pages",
            "encode",
            (680, 0),
            compression="deflate" if case != "predictor_guard" else "lzw",
            predictor="horizontal" if case == "predictor_guard" else "native",
            rows_per_strip=8,
        )
        selected = case == "selected_pages"
        add(
            "cv.decode_pages",
            "decode",
            (1040, 0),
            stride=2 if selected else 1,
            preview_page=1 if selected else 2,
            maximum_pixels=1 if case == "pixel_budget_guard" else 16_777_216,
        )
        graph.connect("source", "image", "batch", "item_1")
        graph.connect("batch", "group", "encode", "pages")
        graph.connect(
            "encode", "buffer" if case == "raw_roundtrip" else "encoded", "decode", "encoded"
        )
        if "guard" not in case:
            add("flow.select_entry", "reference", (680, 370), index=2)
            add(
                "cv.image_norm",
                "verify",
                (1420, 170),
                check_threshold=True,
                maximum=0.0,
                include_alpha=True,
                alignment="pixel_grid" if case == "raw_roundtrip" else "same_frame",
            )
            graph.connect("batch", "group", "reference", "group")
            graph.connect("reference", "value", "verify", "reference")
            graph.connect("decode", "image", "verify", "image")
            add("flow.unbatch", "pages", (1400, -170))
            graph.connect("decode", "pages", "pages", "group")
        note = add("layout.comment", "scope", (0, -440))
        note.name, note.node_size = "Original fixture and transport contract", "large"
        note.description = (
            "Original synthetic images and projects by OpenCVDataflowGUI contributors, MIT. "
            "No customer images. Run all acquires 3 pages with filename/page tickets: "
            "72x48 uint8 gray, 64x40 uint16 BGR and 56x32 float64 BGRA with signed HDR values. "
            "Typed Encoded preserves page labels, tickets and coordinate payloads; raw Buffer "
            "contains TIFF pixels only and the decoder creates new page frames. Selected pages "
            "uses stride 2: indexes 0 and 2, preview 1 is original page 2. Verification requires "
            "zero sample difference. Unbatch emits each selected image in order. "
            "Pixel budget guard intentionally rejects before pixel decoding; Predictor guard "
            "intentionally rejects horizontal prediction on floating pages. No files are written."
        )
        graph.save(HERE / f"{case}.json")
        graph.close()
    truth = {
        "creator": "OpenCVDataflowGUI contributors",
        "license": "MIT",
        "asset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "pages": [
            {
                "size": list(p.shape[1::-1]),
                "dtype": str(p.dtype),
                "channels": 1 if p.ndim == 2 else p.shape[2],
                "samples_sha256": hashlib.sha256(p.tobytes()).hexdigest(),
            }
            for p in pages
        ],
        "cases": list(CASES),
    }
    (HERE / "truth.json").write_bytes((json.dumps(truth, indent=2) + "\n").encode())


if __name__ == "__main__":
    build()
