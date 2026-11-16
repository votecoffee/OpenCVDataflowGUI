"""Original MIT header/inventory fixtures and no-write encoder registry checks."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
CASES = ("file_inventory", "encoder_support", "page_limit", "invalid_header", "unsupported_encoder")


def build():
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    y, x = np.indices((24, 32))
    image = ((x * 7 + y * 3) % 256).astype(np.uint8)
    ok, data = cv2.imencode(".png", image)
    assert ok
    (assets / "\u90e8\u54c1-\u00e9.png").write_bytes(data.tobytes())
    pages = [
        image,
        np.uint16(image) * 257,
        np.dstack((image, image // 2, 255 - image)).astype(np.float32) / 255,
    ]
    ok, data = cv2.imencodemulti(".tiff", pages, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
    assert ok
    (assets / "pages.tiff").write_bytes(data.tobytes())
    animation = cv2.Animation()
    animation.frames = [image, 255 - image, np.roll(image, 7, axis=1)]
    animation.durations = [80, 120, 200]
    ok, data = cv2.imencodeanimation(".png", animation)
    assert ok
    (assets / "animation.png").write_bytes(data.tobytes())
    (assets / "truncated.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    registry = builtin_registry()
    for case in CASES:
        graph = Graph()

        def add(name, position, **parameters):
            block = graph.add(
                registry.create("cv.probe_image_file", block_id=name, parameters=parameters)
            )
            block.position = list(position)
            return block

        if case == "file_inventory":
            for row, (name, path, count) in enumerate(
                (
                    ("still", "\u90e8\u54c1-\u00e9.png", 1),
                    ("pages", "pages.tiff", 3),
                    ("animation", "animation.png", 3),
                )
            ):
                add(
                    name,
                    (0, row * 320),
                    mode="file",
                    path="assets/" + path,
                    minimum_pages=count,
                    maximum_pages=count,
                )
        elif case == "encoder_support":
            for row, format in enumerate(("png", "tiff", "bmp")):
                add(format, (0, row * 320), mode="writer", writer_format=format)
        elif case == "page_limit":
            add("probe", (0, 0), mode="file", path="assets/pages.tiff", minimum_pages=4)
        elif case == "invalid_header":
            add("probe", (0, 0), mode="file", path="assets/truncated.png")
        else:
            add(
                "probe",
                (0, 0),
                mode="writer",
                writer_format="custom",
                custom_extension=".notacodec",
            )
        note = graph.add(registry.create("layout.comment", block_id="scope"))
        note.position = [500, -100]
        note.name = "Original fixtures and preflight limits"
        note.node_size = "large"
        note.description = (
            "Original fixtures/projects by OpenCVDataflowGUI contributors, MIT. No customer "
            "or third-party images. Run once. File inventory expects one Unicode-named PNG, three "
            "mixed-depth TIFF pages and three APNG frames. Encoders only query the registry: "
            "no destination image is written. Page limit deliberately expects at least 4 from a "
            "3-page TIFF; invalid header recognizes PNG but counts zero; the encoder checks "
            "the invented .notacodec extension. All are valid failed preflights with diagnostics. "
            "Passing headers/counts do not verify pixel integrity. Registered encoders do not prove "
            "support for particular sample types, settings or destination permissions. No image "
            "acquisition or source ticket is created. Snapshots are bounded; source files retained."
        )
        graph.save(HERE / f"{case}.json")
        graph.close()
    truth = {
        "creator": "OpenCVDataflowGUI contributors",
        "license": "MIT",
        "cases": list(CASES),
        "file_counts": {"still": 1, "pages": 3, "animation": 3},
        "asset_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(assets.iterdir())
        },
    }
    (HERE / "truth.json").write_bytes((json.dumps(truth, indent=2) + "\n").encode())


if __name__ == "__main__":
    build()
