"""Original centered motion controls; truth never comes from the estimator or its PSF helper."""

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
CASES = {
    "horizontal": {"length": 8.0, "angle": 0.0},
    "diagonal": {"length": 12.0, "angle": 35.0},
    "vertical": {"length": 16.0, "angle": 90.0},
    "mild_noise": {"length": 12.0, "angle": 35.0, "noise": 2.0},
    "highdepth": {"length": 12.0, "angle": 35.0, "highdepth": True},
    "masked": {"length": 8.0, "angle": 0.0, "mask": True},
    "ambiguous": {
        "length": 9.0,
        "angle": 0.0,
        "length_step": 2.0,
        "minimum_margin": 0.1,
        "passed": False,
        "status": "ambiguous",
    },
    "flat": {"passed": False, "status": "insufficient_contrast"},
    "sharp": {"passed": False, "status": "weak_fit"},
    "heavy_noise": {
        "length": 12.0,
        "angle": 35.0,
        "noise": 20.0,
        "passed": False,
        "status": "weak_fit",
    },
    "validation_mismatch": {"passed": False, "status": "validation_rejected"},
}


def line_blur(source, length, angle):
    """Independent trapezoidal integration of bilinear pulled samples, circular boundaries."""
    theta = math.radians(angle)
    weights = {}
    # A different quadrature and pixel-space implementation from the exported motion PSF.
    for j, distance in enumerate(np.linspace(-length / 2, length / 2, 257)):
        dx, dy = distance * math.cos(theta), distance * math.sin(theta)
        x, y = math.floor(dx), math.floor(dy)
        fx, fy = dx - x, dy - y
        weight = (0.5 if j in (0, 256) else 1.0) / 256
        for ox, wx in ((0, 1 - fx), (1, fx)):
            for oy, wy in ((0, 1 - fy), (1, fy)):
                key = (y + oy, x + ox)
                weights[key] = weights.get(key, 0.0) + weight * wx * wy
    return sum(
        weight * np.roll(source, offset, (0, 1)) for offset, weight in weights.items() if weight > 0
    )


def controls():
    sharp = np.clip(np.random.default_rng(84).normal(110, 25, (512, 512)), 0, 255)
    images = {"sharp": np.rint(sharp).astype(np.uint8)}
    sharp16 = np.dstack((sharp * 257, sharp * 180, sharp * 120))
    images["sharp16"] = np.rint(sharp16).astype(np.uint16)
    horizontal = line_blur(sharp, 8, 0)
    vertical = line_blur(sharp, 16, 90)
    for case, options in CASES.items():
        if case == "sharp":
            continue
        elif case == "flat":
            values = np.full(sharp.shape, 110.0)
        elif case == "validation_mismatch":
            values = horizontal.copy()
            for y in range(0, 512, 128):
                for x in range(128, 512, 256):
                    values[y : y + 128, x : x + 128] = vertical[y : y + 128, x : x + 128]
        else:
            values = line_blur(
                sharp16 if options.get("highdepth") else sharp, options["length"], options["angle"]
            )
            if options.get("noise"):
                values += np.random.default_rng(184).normal(0, options["noise"], values.shape)
        depth = np.uint16 if options.get("highdepth") else np.uint8
        images[case] = np.rint(values).clip(0, np.iinfo(depth).max).astype(depth)
    return images


def build():
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    images, hashes = controls(), {}
    for name, image in images.items():
        ok, data = cv2.imencode(".png", image)
        assert ok
        (assets / f"{name}.png").write_bytes(data.tobytes())
        hashes[name + ".png"] = hashlib.sha256(data).hexdigest()
    registry = builtin_registry()
    for case, options in CASES.items():
        graph = Graph()
        try:

            def add(kind, key, parameters=None, position=(0, 0)):
                b = graph.add(registry.create(kind, block_id=key, parameters=parameters or {}))
                b.position = list(position)
                return b

            encoding = "bgr" if options.get("highdepth") else "gray"
            for key, name in (
                ("capture", case),
                ("truth", "sharp16" if options.get("highdepth") else "sharp"),
            ):
                add(
                    "input.file",
                    key,
                    {
                        "path": f"assets/{name}.png",
                        "path_base": "project",
                        "color_format": encoding,
                        "pixel_depth": "unchanged",
                    },
                    (0, 0 if key == "capture" else 400),
                )
            settings = {k: options[k] for k in ("length_step", "minimum_margin") if k in options}
            estimator = add("inspect.estimate_motion_blur", "estimate", settings, (400, 0))
            estimator.node_size = "large"
            estimator.exposed_outputs.extend(["diagnostic", "result_graphics"])
            graph.connect("capture", "image", "estimate", "image")
            add(
                "cv.wiener_deblur",
                "restore",
                {
                    "boundary": "circular",
                    "noise_to_signal": 0.005 if options.get("noise") else 0.0001,
                },
                (800, 0),
            )
            graph.connect("capture", "image", "restore", "image")
            graph.connect("estimate", "psf", "restore", "psf")
            if options.get("mask"):
                add("region.shape", "selection", {"rectangle": [0, 0, 256, 512]}, (0, -350))
                graph.connect("capture", "image", "selection", "image")
                graph.connect("selection", "region", "estimate", "region")
                graph.connect("selection", "region", "restore", "region")
            for key, image, priority, y in (
                ("restored", "restore", 10, 0),
                ("candidate", "capture", 0, 350),
            ):
                overlay = key + "_overlay"
                add("cv.overlay_result_graphics", overlay, position=(1200, y))
                graph.connect(image, "image", overlay, "image")
                graph.connect("estimate", "result_graphics", overlay, "graphics_1")
                add("output.filmstrip_image", key, {"priority": priority}, (1600, y))
                graph.connect(overlay, "image", key, "image")
            note = add("layout.comment", "attribution", position=(0, -650))
            note.name, note.node_size = "Original motion-estimation controls - MIT", "large"
            note.description = (
                "Original MIT seeded texture, independent line-path integration, projects and "
                "truth by OpenCVDataflowGUI contributors. No customer or third-party images. "
                "The estimator sees only one degraded image and selected tiles; Sharp truth "
                "is a separate disconnected comparison image. Linear motion, isotropic texture "
                "and circular boundaries are controlled assumptions, not facts about real scenes. "
                "Fit and validation use disjoint tiles. Flat/sharp/noisy/mixed/ambiguous cases "
                "withhold PSF and stop restoration. Frequency floor is a nuisance parameter, "
                "not measured SNR. See README/truth.json for numerical limits and attribution."
            )
            graph.save(HERE / f"{case}.json")
        finally:
            graph.close()
    truth = {
        "license": "MIT - original fixtures",
        "cases": CASES,
        "asset_sha256": hashes,
        "length_tolerance_pixels": 1.0,
        "angle_tolerance_degrees": 5.0,
        "maximum_restored_mse_ratio": 0.8,
    }
    (HERE / "truth.json").write_bytes(
        (json.dumps(truth, indent=2, allow_nan=False) + "\n").encode()
    )


if __name__ == "__main__":
    build()
