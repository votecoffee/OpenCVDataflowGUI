"""Original MIT blurred coupons and changed-part scenes; no external assets."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
CASES = {
    "project": {"sharp": "sharp", "target": "blur", "truth": "sharp"},
    "moving_parts": {"sharp": "moving_donor", "target": "moving_blur", "truth": "moving_truth"},
    "highdepth": {"sharp": "sharp16", "target": "blur16", "truth": "sharp16", "coverage": True},
    "gaussian": {
        "sharp": "sharp",
        "target": "gaussian",
        "truth": "sharp",
        "blur_model": "gaussian",
    },
    "unchanged": {"sharp": "sharp", "target": "sharp", "truth": "sharp", "unchanged": True},
    "no_neighbor_fail": {
        "sharp": "sharp",
        "target": "sharp",
        "truth": "sharp",
        "unchanged": True,
        "no_neighbor": "fail",
    },
}


def box_reference(image):
    """Independent horizontal seven-sample average, with reflect-101 edges."""
    pads = ((0, 0), (3, 3)) + (((0, 0),) if image.ndim == 3 else ())
    padded = np.pad(image.astype(np.float64), pads, mode="reflect")
    return np.rint(sum(padded[:, i : i + image.shape[1]] / 7 for i in range(7))).astype(image.dtype)


def fixtures():
    y, x = np.mgrid[:96, :160]
    values = np.rint(127 + 55 * np.sin(x * 0.6) + 35 * np.cos(y * 0.43)).astype(np.uint8)
    cv2.circle(values, (32, 24), 9, 20, 2)
    cv2.rectangle(values, (105, 60), (130, 76), 220, 2)
    high = np.dstack(
        (
            values.astype(np.uint16) * 257,
            values.astype(np.uint16) * 180,
            values.astype(np.uint16) * 120,
        )
    )
    current, previous = values.copy(), values.copy()
    current[32:49, 100:116] = 255
    previous[32:49, 40:56] = 255
    return {
        "sharp": values,
        "blur": box_reference(values),
        "gaussian": cv2.GaussianBlur(values, (7, 7), 1.5),
        "sharp16": high,
        "blur16": box_reference(high),
        "moving_truth": current,
        "moving_blur": box_reference(current),
        "moving_donor": previous,
    }


def build():
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    for name, image in fixtures().items():
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        (assets / (name + ".png")).write_bytes(encoded.tobytes())
    registry = builtin_registry()
    for case, settings in CASES.items():
        graph = Graph()
        try:
            paths = [settings["sharp"], settings["target"], settings["sharp"], settings["truth"]]
            for i, asset in enumerate(paths):
                source = graph.add(
                    registry.create(
                        "input.file",
                        block_id=f"frame{i}",
                        parameters={
                            "path": f"assets/{asset}.png",
                            "path_base": "project",
                            "color_format": "bgr" if case == "highdepth" else "gray",
                            "pixel_depth": "unchanged",
                        },
                    )
                )
                source.name = (
                    "Current blurred target"
                    if i == 1
                    else "Independent sharp truth"
                    if i == 3
                    else f"Sharp neighbor {i}"
                )
                source.position = [0, i * 290]
            group = graph.add(
                registry.create("flow.batch", block_id="frames", parameters={"size": 3})
            )
            group.position = [400, 0]
            for i in range(3):
                graph.connect(f"frame{i}", "image", "frames", f"item_{i + 1}")
            restored = graph.add(
                registry.create(
                    "sequence.restore_neighbors",
                    block_id="restore",
                    parameters={
                        "target_index": 1,
                        "alignment": "pixel_grid",
                        "maximum_model_error": 0.005,
                        "change_radius": 5,
                        "blur_model": settings.get("blur_model", "horizontal_box"),
                        "no_neighbor": settings.get("no_neighbor", "pass_through"),
                    },
                )
            )
            restored.position, restored.node_size = [850, 0], "large"
            graph.connect("frames", "group", "restore", "frames")
            if settings.get("coverage"):
                coverage = graph.add(
                    registry.create("flow.batch", block_id="coverage", parameters={"size": 3})
                )
                coverage.position = [650, 600]
                for i in range(3):
                    region = graph.add(
                        registry.create(
                            "region.shape",
                            block_id=f"support{i}",
                            parameters={"rectangle": [8, 8, 144, 80]},
                        )
                    )
                    region.position = [350, 1000 + i * 250]
                    graph.connect(f"frame{i}", "image", f"support{i}", "image")
                    graph.connect(f"support{i}", "mask", "coverage", f"item_{i + 1}")
                graph.connect("coverage", "group", "restore", "coverage")
                selected = graph.add(
                    registry.create(
                        "region.shape",
                        block_id="selection",
                        parameters={"rectangle": [18, 18, 124, 60]},
                    )
                )
                selected.position = [650, 350]
                graph.connect("frame1", "image", "selection", "image")
                graph.connect("selection", "region", "restore", "region")
            if not settings.get("unchanged"):
                for identifier, source in (("before", "frame1"), ("verify", "restore")):
                    tool = graph.add(
                        registry.create(
                            "inspect.image_quality",
                            block_id=identifier,
                            parameters={
                                "metric": "max_absolute_error",
                                "check_threshold": identifier == "verify",
                                "threshold": 0,
                            },
                        )
                    )
                    tool.position = [1300, 0 if identifier == "verify" else 350]
                    graph.connect(source, "image", identifier, "image_a")
                    graph.connect("frame3", "image", identifier, "image_b")
                    graph.connect("restore", "restored_mask", identifier, "mask")
            note = graph.add(registry.create("layout.comment", block_id="attribution"))
            note.name, note.position, note.node_size = (
                "Original restoration fixtures - MIT",
                [0, -400],
                "large",
            )
            note.description = (
                "Original MIT images, generator and projects by OpenCVDataflowGUI contributors. "
                "No customer or third-party data. Three ordered captures are sharp/blurred/sharp; "
                "the middle capture keeps its ticket. Horizontal blur uses an independent "
                "seven-sample "
                "average. In the moving scene, the target part is at x100, the donor part at x40; "
                "both changed areas retain original target pixels. Sharp truth is not "
                "reconstructed "
                "from the target. Pixel-grid mode asserts registration of these generated "
                "captures. Accepted detail must match truth. No-neighbor-fail intentionally fails. "
                "Model agreement cannot prove absence of low-contrast defects. "
                "OpenCV Sobel/filter reference: "
                "https://docs.opencv.org/5.0/main_modules/imgproc_filter.html"
            )
            graph.save(HERE / (case + ".json"))
        finally:
            graph.close()
    truth = {
        "license": "MIT - original repository fixtures",
        "size": [160, 96],
        "target_index": 1,
        "blur_length": 7,
        "moving_rectangles_xywh": [[40, 32, 16, 17], [100, 32, 16, 17]],
        "cases": {
            case: {
                "state": "fail" if case == "no_neighbor_fail" else "pass",
                "unchanged": settings.get("unchanged", False),
            }
            for case, settings in CASES.items()
        },
        "asset_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(assets.glob("*.png"))
        },
    }
    (HERE / "truth.json").write_bytes((json.dumps(truth, indent=2) + "\n").encode())


if __name__ == "__main__":
    build()
