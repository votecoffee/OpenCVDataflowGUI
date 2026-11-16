"""Original MIT mixed-depth stack and graph projects for bounded random page access."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
CASES = ("random_access", "converted_page", "no_cache", "page_outside_stack", "workspace_limit")
REQUESTS = [256, 3, 128, 256, 255, 3, 3, 0, 128, 128, 256]


def page(index):
    y, x = np.indices((96, 128))
    if index % 4 == 0:
        return ((x * 193 + y * 157 + index * 101) % 65536).astype(np.uint16)
    if index % 4 == 1:
        return np.dstack(((x + index) % 256, (y * 3) % 256, (x + y) % 256)).astype(np.uint8)
    if index % 4 == 2:
        return ((x - 64) / 8 + (y - 48) / 4 + index / 2).astype(np.float32)
    return ((x - y) * 0.125 + index).astype(np.float64)


def build():
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    pages = [page(i) for i in range(257)]
    ok, data = cv2.imencodemulti(".tiff", pages, [cv2.IMWRITE_TIFF_COMPRESSION, 8])
    assert ok
    (assets / "mixed-stack.tiff").write_bytes(data.tobytes())
    registry = builtin_registry()
    for case in CASES:
        graph = Graph()
        block = graph.add(
            registry.create(
                "input.tiff_page",
                block_id="page",
                parameters={
                    "path": "assets/mixed-stack.tiff",
                    "path_base": "project",
                    "page_index": 257
                    if case == "page_outside_stack"
                    else 129
                    if case == "converted_page"
                    else 256,
                    "cache_pages": 0 if case == "no_cache" else 2,
                    "cache_bytes": 512 * 1024,
                    "maximum_decoded_bytes": 1 if case == "workspace_limit" else 256 * 1024**2,
                    "color_format": "rgb" if case == "converted_page" else "unchanged",
                    "pixel_depth": "float32" if case == "converted_page" else "unchanged",
                },
            )
        )
        block.position = [0, 0]
        if case == "random_access":
            selector = graph.add(
                registry.create(
                    "input.counter", block_id="selector", parameters={"start": 256, "step": 0}
                )
            )
            selector.position = [-360, 0]
            graph.connect("selector", "value", "page", "page_index")
        note = graph.add(registry.create("layout.comment", block_id="scope"))
        note.position, note.node_size = [360, 0], "large"
        note.name = "Original 257-page stack and access scope"
        note.description = (
            "All pages, reference formulas, annotations and project content are original "
            "OpenCVDataflowGUI contributor work, MIT. No customer or third-party media. "
            "257 top-left TIFF pages alternate uint16 gray, uint8 BGR, float32 gray and "
            "float64 gray; negative/HDR values are deliberate. Select a zero-based Page index "
            "or wire the integer input. The wired counter uses Start as a selector and Step=0; "
            "change its Start to request another page without changing the TIFF block. "
            "Repeat Run once to observe cache hits and fresh tickets. "
            "Only requested pages are decoded; intermediate TIFF directories still need traversal. "
            "Two-page/512 KiB LRU bounds raw retained pages. On-change checks file identity/times; "
            "Verify content hashes each request; Pinned ignores changes until reset/path change. "
            "Page-outside-stack and workspace-limit cases deliberately emit no image. "
            "Content/page pixel grids do not establish camera or physical calibration."
        )
        graph.save(HERE / f"{case}.json")
        graph.close()
    truth = {
        "creator": "OpenCVDataflowGUI contributors",
        "license": "MIT",
        "page_count": len(pages),
        "size": [128, 96],
        "cases": list(CASES),
        "request_order": REQUESTS,
        "raw_stack_bytes": sum(image.nbytes for image in pages),
        "pages": [
            {
                "index": i,
                "dtype": image.dtype.name,
                "shape": list(image.shape),
                "bgr_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            }
            for i, image in enumerate(pages)
        ],
        "asset_sha256": {"mixed-stack.tiff": hashlib.sha256(data.tobytes()).hexdigest()},
    }
    (HERE / "truth.json").write_bytes((json.dumps(truth, indent=2) + "\n").encode())


if __name__ == "__main__":
    build()
