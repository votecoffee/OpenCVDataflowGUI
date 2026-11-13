"""Original equal-grayscale color patches with independent color-fitting context."""

import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent


def main():
    rng = np.random.default_rng(42)
    image = cv2.GaussianBlur(rng.integers(0, 256, (128, 192, 3), dtype=np.uint8), (0, 0), 3)
    a, b = [220, 70, 85], [25, 125, 54]
    image[20:105, 20:88], image[20:105, 104:172] = a, b
    assert cv2.imwrite(str(HERE / "patches.png"), image)
    degenerate = image.copy()
    degenerate[:] = a
    degenerate[20:105, 104:172] = b
    assert cv2.imwrite(str(HERE / "two_color_degenerate.png"), degenerate)
    truth = {
        "patch_a_bgr": a,
        "patch_b_bgr": b,
        "ordinary_gray_dn": 92,
        "patch_a_rectangle": [28, 28, 52, 68],
        "patch_b_rectangle": [112, 28, 52, 68],
        "minimum_decolor_separation_dn": 50,
        "scope": "Original synthetic equal-BT.601 patches; mapping depends on full image context",
    }
    (HERE / "truth.json").write_bytes((json.dumps(truth, indent=2) + "\n").encode())
    registry, graph = builtin_registry(), Graph()

    def add(kind, ident, name, position, parameters=None):
        block = graph.add(registry.create(kind, block_id=ident, parameters=parameters or {}))
        block.name, block.position = name, list(position)
        return block

    source = add(
        "input.file",
        "source",
        "Two equal-gray color patches",
        (0, 0),
        {"path": "patches.png", "path_base": "project"},
    )
    decolor = add("cv.decolor", "decolor", "Preserve color separation in gray", (320, 0))
    a_region = add(
        "region.shape",
        "a_region",
        "Measure patch A",
        (320, 600),
        {"rectangle": truth["patch_a_rectangle"]},
    )
    b_region = add(
        "region.shape",
        "b_region",
        "Measure patch B",
        (320, 900),
        {"rectangle": truth["patch_b_rectangle"]},
    )
    before = add(
        "inspect.region_contrast",
        "before",
        "Ordinary grayscale: equal means",
        (740, 550),
        {"metric": "absolute_difference"},
    )
    after = add(
        "inspect.region_contrast",
        "after",
        "Decolor: at least 50 DN apart",
        (740, 0),
        {
            "metric": "absolute_difference",
            "check_threshold": True,
            "minimum_value": truth["minimum_decolor_separation_dn"],
        },
    )
    note = add("layout.comment", "scope", "Source and fitting context", (0, -470))
    note.node_size = "large"
    note.description = (
        "Original project fixture and graph (MIT), generated with PCG64 seed 42 and Gaussian "
        "context. Both designed patches convert to ordinary 8-bit gray 92. Native decolor "
        "separates them with the whole image as fitting context. Gray levels and polarity "
        "can change with image content. No calibrated luminance or real-part accuracy is claimed. "
        "The two-color-only companion records a native collapsed mapping; its primary outputs "
        "are withheld rather than presenting black as a successful conversion."
    )
    graph.connect(source.id, "image", decolor.id, "image")
    for region in (a_region, b_region):
        graph.connect(source.id, "image", region.id, "image")
    for tool, output in ((before, "ordinary_gray"), (after, "image")):
        graph.connect(decolor.id, output, tool.id, "image")
        graph.connect(a_region.id, "region", tool.id, "background_region")
        graph.connect(b_region.id, "region", tool.id, "signal_region")
    graph.save(HERE / "contrast_preserving_gray.json")
    record = graph.to_dict()
    next(b for b in record["blocks"] if b["id"] == "source")["parameters"]["path"] = (
        "two_color_degenerate.png"
    )
    failed = Graph.from_dict(record, registry)
    failed.save(HERE / "degenerate_mapping.json")
    failed.close()
    graph.close()


if __name__ == "__main__":
    main()
