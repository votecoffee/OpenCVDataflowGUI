"""Original timed radiance controls; truth comes from authored scenes, not HDR blocks."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
TIMES = [0.25, 0.5, 1.0]
CASES = ("static", "moving", "strict", "ignore_clipping", "missing", "aligned")


def controls(case):
    aligned = case == "aligned"
    if aligned:
        raw = cv2.resize(np.random.default_rng(14).integers(16, 112, (16, 24), np.uint8), (96, 80))
        codes = (raw // 4 * 4).astype(np.uint16)
    else:
        y, x = np.indices((96, 160))
        codes = (24 + 4 * ((x // 4 + y // 3) % 20)).astype(np.uint16)
        codes[5:25, 100:140] = 200 + 4 * ((x[5:25, 100:140] // 4 + y[5:25, 100:140] // 3) % 11)
    radiance = codes.astype(np.float64) / 64
    scenes = [radiance.copy() for _ in TIMES]
    changed = np.zeros(codes.shape, bool)
    moving = case in ("moving", "strict", "ignore_clipping")
    if moving:
        for i, scene in enumerate(scenes):
            x = 20 + 40 * i
            scene[54:64, x : x + 12] = 3.75
            changed[54:64, x : x + 12] = True
    missing = np.zeros(codes.shape, bool)
    if case == "missing":
        missing[10:18, 20:28] = True
        for scene in scenes:
            scene[missing] = 10
    images = [
        np.rint(np.clip(scene * 128 * time, 0, 255)).astype(np.uint8)
        for scene, time in zip(scenes, TIMES, strict=True)
    ]
    support = [np.ones(codes.shape, bool) for _ in TIMES]
    shifts = [(2, 1), (0, 0), (-2, -1)] if aligned else [(0, 0)] * 3
    if aligned:
        h, w = codes.shape
        y, x = np.indices(codes.shape)
        images = [
            cv2.warpAffine(
                image, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), flags=cv2.INTER_NEAREST
            )
            for image, (dx, dy) in zip(images, shifts, strict=True)
        ]
        support = [(x + dx >= 0) & (x + dx < w) & (y + dy >= 0) & (y + dy < h) for dx, dy in shifts]
    rejected = changed.copy()
    if case == "ignore_clipping":
        rejected[54:64, 100:112] = False
    # Full raw images on the reference grid are known independently of alignment.
    ideal = [
        np.rint(np.clip(scene * 128 * time, 0, 255)).astype(np.uint8)
        for scene, time in zip(scenes, TIMES, strict=True)
    ]
    counts = sum(
        ((image > 4) & (image < 251) & valid).astype(np.uint8)
        for image, valid in zip(ideal, support, strict=True)
    )
    valid = (counts >= 2) & ~rejected
    return images, radiance, valid, rejected, shifts


def build():
    assets = HERE / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    registry = builtin_registry()
    truth = {
        "license": "MIT - original repository fixtures",
        "times_seconds": TIMES,
        "cases": {},
        "asset_sha256": {},
    }
    for case in CASES:
        images, radiance, valid, rejected, shifts = controls(case)
        h, w = valid.shape
        paths = []
        for i, image in enumerate(images):
            name = f"{case}_{i}.png"
            ok, encoded = cv2.imencode(".png", image)
            assert ok
            (assets / name).write_bytes(encoded.tobytes())
            paths.append("assets/" + name)
            truth["asset_sha256"][name] = hashlib.sha256(encoded).hexdigest()
        truth["cases"][case] = {
            "size": [w, h],
            "expected_pass": case != "strict",
            "measured_pixels": int(valid.sum()),
            "inconsistent_pixels": int(rejected.sum()),
            "fallback_pixels": int((~valid).sum()),
            "coverage": float(valid.mean()),
            "selected_mean_radiance": float(radiance[valid].mean()),
            "valid_mask_sha256": hashlib.sha256(valid.astype(np.uint8) * 255).hexdigest(),
            "motion_mask_sha256": hashlib.sha256(rejected.astype(np.uint8) * 255).hexdigest(),
            "shifts_to_reference": [[-dx, -dy] for dx, dy in shifts],
        }
        graph = Graph()
        try:

            def add(kind, key, parameters=None, position=(0, 0)):
                b = graph.add(registry.create(kind, block_id=key, parameters=parameters or {}))
                b.position = list(position)
                return b

            for i, path in enumerate(paths):
                add(
                    "input.file",
                    f"exposure_{i}",
                    {"path": path, "path_base": "project", "color_format": "gray"},
                    (0, i * 300),
                )
            add("flow.batch", "batch", position=(380, 0))
            for i in range(3):
                graph.connect(f"exposure_{i}", "image", "batch", f"item_{i + 1}")
            upstream, port = "batch", "group"
            if case == "aligned":
                add(
                    "cv.align_exposures",
                    "align",
                    {
                        "reference_index": 1,
                        "max_bits": 3,
                        "exclude_range": 1,
                        "border_mode": "zero_pad",
                    },
                    (760, 300),
                )
                graph.connect("batch", "group", "align", "exposures")
                upstream, port = "align", "aligned"
            merge = add(
                "cv.merge_hdr_motion",
                "hdr",
                {
                    "reference_index": 1,
                    "exposure_times": TIMES,
                    "absolute_tolerance": 0.0,
                    "relative_tolerance": 0.0,
                    "grid_policy": "matching_frames"
                    if case == "aligned"
                    else "assert_pixel_alignment",
                    "minimum_coverage": 1.0 if case == "strict" else 0.8,
                    "clipping_evidence": "ignore" if case == "ignore_clipping" else "bounds",
                },
                (1140, 0),
            )
            merge.exposed_outputs.extend(
                ["motion_mask", "fallback_mask", "contributors", "result_graphics"]
            )
            merge.node_size = "large"
            merge.description = (
                "Only Image plus Valid mask is measured HDR; Display fills are review-only."
            )
            add(
                "cv.merge_debevec",
                "conventional",
                {"reference_index": 1, "exposure_times": TIMES},
                (1140, 650),
            )
            add("cv.image_statistics", "statistics", position=(1520, 650))
            add("cv.tonemap_hdr", "review", position=(1520, 0))
            add("cv.overlay_result_graphics", "overlay", position=(1900, 0))
            add("output.filmstrip_image", "filmstrip", position=(2280, 0))
            for wire in [
                (upstream, port, "hdr", "exposures"),
                (upstream, port, "conventional", "exposures"),
                ("hdr", "image", "statistics", "image"),
                ("hdr", "valid_mask", "statistics", "mask"),
                ("hdr", "display", "review", "image"),
                ("review", "image", "overlay", "image"),
                ("hdr", "result_graphics", "overlay", "graphics_1"),
                ("overlay", "image", "filmstrip", "image"),
            ]:
                graph.connect(*wire)
            note = add("layout.comment", "attribution", position=(0, -500))
            note.name = "Original timed radiance controls - MIT"
            note.node_size = "large"
            note.description = (
                "Original MIT synthetic radiance scenes, PNGs, generator, projects and independent "
                "support truth by OpenCVDataflowGUI contributors. No customer/third-party images. "
                "Linear codes are radiance*128*exposure seconds; durations are 0.25, 0.5, 1.0. "
                "Motion puts a 12x10 part at three positions. Long exposure clips the third part; "
                "default clipping bounds still detect the contradiction. Ignore clipping is an "
                "explicit weaker comparison. Missing saturates an 8x8 region in every frame. "
                "Aligned applies known shifts and native MTB restores the reference grid; support "
                "metadata excludes alignment padding. Image is zero outside measured support. "
                "Display/filmstrip fills are reference-only review, not measured HDR. Strict "
                "intentionally fails 100% coverage and keeps diagnostics. Units are arbitrary "
                "response per second, not absolute radiometry or physical motion proof. See README."
            )
            graph.save(HERE / f"{case}.json")
        finally:
            graph.close()
    (HERE / "truth.json").write_bytes(
        (json.dumps(truth, indent=2, allow_nan=False) + "\n").encode()
    )


if __name__ == "__main__":
    build()
