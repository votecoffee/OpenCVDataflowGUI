"""Original finite image sequences with exact temporal noise/covariance references."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
SIGNAL = [-3, -2, -1, 0, 1, 2, 3, 0]
ORTHOGONAL = [1, -1, 1, -1, 1, -1, 1, -1]
CASES = (
    "stationary_noise",
    "channel_covariance",
    "paired_sources",
    "exponential",
    "masked_history",
    "ticket_mismatch",
    "memory_limit",
)


def references():
    y, x = np.indices((48, 64))
    gain = np.where(x < 32, 1, 2)
    base_x = 10000 + (x * 13 + y * 7) % 1000
    base_y = 20000 + (x * 7 + y * 11) % 2000
    a = np.stack([base_x + value * gain for value in SIGNAL]).astype(np.uint16)
    b = np.stack(
        [
            base_y + (3 * value + noise) * gain
            for value, noise in zip(SIGNAL, ORTHOGONAL, strict=True)
        ]
    ).astype(np.uint16)
    color = np.stack(
        [
            np.dstack((40 + value * gain, 80 + (2 * value + noise) * gain, 120 - value * gain))
            for value, noise in zip(SIGNAL, ORTHOGONAL, strict=True)
        ]
    ).astype(np.uint8)
    return a, b, color


def build():
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    arrays = references()
    for prefix, stack in zip(("x", "y", "color"), arrays, strict=True):
        for i, image in enumerate(stack):
            ok, encoded = cv2.imencode(".png", image)
            assert ok
            (assets / f"{prefix}_{i:02d}.png").write_bytes(encoded.tobytes())
    registry = builtin_registry()
    for case in CASES:
        graph = Graph()
        source = graph.add(
            registry.create(
                "input.file",
                block_id="x",
                parameters={
                    "source_mode": "files",
                    "paths": [
                        f"assets/{'color' if case == 'channel_covariance' else 'x'}_{i:02d}.png"
                        for i in range(8)
                    ],
                    "path_base": "project",
                    "repeat": "stop",
                    "color_format": "unchanged",
                    "pixel_depth": "unchanged",
                },
            )
        )
        source.position = [0, 0]
        stats = graph.add(
            registry.create(
                "sequence.pixel_moments",
                block_id="statistics",
                parameters={
                    "channel_x": "blue" if case == "channel_covariance" else "gray",
                    "channel_y": "green" if case == "channel_covariance" else "gray",
                    "method": "exponential" if case == "exponential" else "cumulative",
                    "alpha": 0.25,
                    "alignment": "pixel_indices"
                    if case in ("paired_sources", "ticket_mismatch")
                    else "same_frame",
                    "ticket_policy": "require_match",
                    "preview_max": 5.0,
                    "max_bytes_mb": 0.01 if case == "memory_limit" else 256.0,
                },
            )
        )
        stats.position = [720, 0]
        graph.connect("x", "image", "statistics", "image")
        if case in ("paired_sources", "ticket_mismatch"):
            other = graph.add(
                registry.create(
                    "input.file",
                    block_id="y",
                    parameters={
                        "source_mode": "files",
                        "paths": [f"assets/y_{i:02d}.png" for i in range(8)],
                        "path_base": "project",
                        "repeat": "stop",
                        "color_format": "unchanged",
                        "pixel_depth": "unchanged",
                    },
                )
            )
            other.position = [0, 340]
            graph.connect("y", "image", "statistics", "other")
            if case == "paired_sources":
                counter = graph.add(registry.create("input.counter", block_id="part"))
                counter.position = [-360, 160]
                for block in (source, other):
                    block.ticket_settings = {"mode": "external", "value": 1}
                    graph.connect("part", "value", block.id, "ticket")
        if case == "masked_history":
            region = graph.add(
                registry.create(
                    "region.shape",
                    block_id="region",
                    parameters={"shape": "circle", "center": [32, 24], "radius": 16},
                )
            )
            region.position = [360, 230]
            graph.connect("x", "image", "region", "image")
            graph.connect("region", "region", "statistics", "region")
        note = graph.add(registry.create("layout.comment", block_id="scope"))
        note.position, note.node_size = [720, 430], "large"
        note.name = "Original temporal references and measurement scope"
        note.description = (
            "Images, signals, pixel references and project are original OpenCVDataflowGUI "
            "contributor work, MIT. No customer or third-party media. Eight stationary 64x48 "
            "frames use a zero-mean signal with sample variance 4 and gains 1 and 2 in left/right "
            "halves. "
            "Paired Y has 3x signal plus orthogonal noise; green has 2x signal plus that noise. "
            "Default gray self-pair has covariance equal to variance. Counts/validity are "
            "per pixel. "
            "Paired sources explicitly assert matching pixel indices and share an external ticket; "
            "this fixture does not prove physical registration or exposure synchronization. "
            "Exponential weights are .25*.75^age; sample normalization uses W-sum(w²)/W. "
            "Run all from reset for the stated references. Ticket-mismatch and memory-limit "
            "projects deliberately emit no statistics. Temporal variation can include scene "
            "movement/lighting drift and is not automatically pure camera noise."
        )
        graph.save(HERE / f"{case}.json")
        graph.close()
    truth = {
        "creator": "OpenCVDataflowGUI contributors",
        "license": "MIT",
        "cases": list(CASES),
        "signal": SIGNAL,
        "orthogonal": ORTHOGONAL,
        "size": [64, 48],
        "count": 8,
        "sample_variance_signal": 4.0,
        "gain_boundary_x": 32,
        "asset_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(assets.glob("*.png"))
        },
    }
    (HERE / "truth.json").write_bytes((json.dumps(truth, indent=2) + "\n").encode())


if __name__ == "__main__":
    build()
