"""Build a rectified disparity pair with a known depth edge and occluded matches."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

SHAPE = (72, 192)
EDGE = 96
SCALE = 32
INVALID = 65535
RIGHT_INVALID = -100.0
HOLE = (slice(16, 26), slice(39, 47))
RIGHT_HOLE = (slice(33, 45), slice(110, 118))


def maps():
    rng = np.random.default_rng(121605)
    ideal = np.full(SHAPE, 8, np.uint8)
    ideal[:, EDGE:] = 16
    guide = np.where(ideal == 8, 46, 206).astype(np.uint8)
    raw = np.uint16(np.rint((ideal + rng.normal(0, 0.46, SHAPE)) * SCALE))
    raw[:, :8] = INVALID
    raw[HOLE] = INVALID
    right = np.full(SHAPE, -8, np.float32)
    right[:, EDGE - 16 :] = -16
    right[:, -16:] = RIGHT_INVALID
    right[RIGHT_HOLE] = RIGHT_INVALID
    return guide, raw, right, ideal


def write_assets(assets):
    assets.mkdir(parents=True, exist_ok=True)
    for name, image in zip(
        ("guide.png", "left_q5.png", "right.tiff", "truth.png"), maps(), strict=True
    ):
        ok, encoded = cv2.imencode(Path(name).suffix, image)
        if not ok:
            raise ValueError(f"Could not encode {name}")
        (assets / name).write_bytes(encoded.tobytes())


def build(output):
    path = output / "external_disparity_wls.json"
    if path.exists():
        raise FileExistsError("Use an empty output directory")
    write_assets(output / "assets")
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, title, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = title, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Filter an imported disparity pair", 0, -260)
        note.node_size = "large"
        note.description = (
            "Original synthetic stereo fields by OpenCVDataflowGUI contributors (MIT). "
            "Compare masked smoothing with left-right confidence at a known depth step."
        )
        for i, (key, title, file) in enumerate(
            (
                ("left", "Noisy left map · 32 samples/px", "left_q5.png"),
                ("guide", "Aligned left guide", "guide.png"),
                ("right", "Right map · signed pixel disparity", "right.tiff"),
                ("truth", "Reference · 8 and 16 pixels", "truth.png"),
            )
        ):
            source = add(
                "input.file",
                key,
                title,
                0,
                i * 235,
                path=f"assets/{file}",
                path_base="project",
                pixel_depth="unchanged",
                color_format="gray",
            )
            source.ticket_settings = {"mode": "fixed", "value": "disparity-step"}
        for i, (key, title) in enumerate(
            (("guided", "Filter left disparity"), ("paired", "Filter and check right disparity"))
        ):
            add(
                "3d.filter_disparity_wls",
                key,
                title,
                415,
                70 + 360 * i,
                samples_per_pixel=SCALE,
                invalid_mode="also_use_code",
                invalid_code=INVALID,
                right_invalid_mode="also_use_code",
                right_invalid_code=RIGHT_INVALID,
                minimum_confidence=0.2 if key == "paired" else 0.0,
            )
            graph.connect("left", "image", key, "disparity")
            graph.connect("guide", "image", key, "guide")
            if key == "paired":
                graph.connect("right", "image", key, "right_disparity")
            compare = f"check_{key}"
            add(
                "3d.evaluate_disparity",
                compare,
                "Compare with known disparity",
                840,
                70 + 360 * i,
                alignment="pixel_grid",
                reference_samples_per_pixel=1,
                bad_threshold_px=0.35,
            )
            graph.connect("truth", "image", compare, "reference")
            graph.connect(key, "disparity", compare, "estimate")
            graph.connect(key, "valid_mask", compare, "estimate_valid")
            if key == "paired":
                graph.connect(key, "confidence", compare, "confidence")
        report = validate_graph(graph)
        if not report["valid"]:
            raise ValueError(report["issues"])
        graph.save(path)
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
