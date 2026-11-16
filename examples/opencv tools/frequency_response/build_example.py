"""Original image captures with independently known circular transfer functions."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
CASES = {
    "identity": ("reference16.png", "reference16.png", {}, "pass"),
    "binomial_blur": ("blur16.png", "reference16.png", {}, "pass"),
    "shift_phase_fail": (
        "shift16.png",
        "reference16.png",
        {"check_phase": True, "maximum_phase_degrees": 5},
        "fail",
    ),
    "gain_half_fail": (
        "half16.png",
        "reference16.png",
        {"minimum_mean_gain": 0.8, "maximum_mean_gain": 1.2},
        "fail",
    ),
    "horizontal_grating": (
        "grating16.png",
        "grating16.png",
        {"direction": "horizontal", "angular_tolerance_degrees": 0},
        "pass",
    ),
    "masked_identity": ("exterior16.png", "reference16.png", {"window": "hann"}, "pass"),
    "color_red": ("color_sample.png", "color_reference.png", {"channel": "R"}, "pass"),
    "no_reference_energy": ("reference16.png", "constant16.png", {}, "fail"),
    "budget_guard": ("reference16.png", "reference16.png", {"maximum_workspace_mib": 1}, "error"),
    "grid_guard": (
        "reference16.png",
        "reference16.png",
        {"reference_alignment": "same_frame"},
        "error",
    ),
}


def images():
    rng = np.random.default_rng(166)
    reference = rng.integers(200, 2800, (96, 128), dtype=np.uint16) * 16
    wide = reference.astype(np.uint32)
    row = (np.roll(wide, -1, 1) + 2 * wide + np.roll(wide, 1, 1)) // 4
    blur = ((np.roll(row, -1, 0) + 2 * row + np.roll(row, 1, 0)) // 4).astype(np.uint16)
    y, x = np.indices(reference.shape)
    outside = (x - 64) ** 2 + (y - 48) ** 2 > 32**2
    exterior = reference.copy()
    exterior[outside] += 8192
    color = rng.integers(25, 190, (96, 128, 3), np.uint8)
    color[..., 2] = color[..., 2] // 2 * 2
    sample = color.copy()
    sample[..., 2] //= 2
    return {
        "reference16.png": reference,
        "blur16.png": blur,
        "shift16.png": np.roll(reference, (2, -3), axis=(0, 1)),
        "half16.png": reference // 2,
        "exterior16.png": exterior,
        "grating16.png": np.tile(np.array([16000, 32000, 16000, 0], np.uint16), (96, 32)),
        "constant16.png": np.full(reference.shape, 31000, np.uint16),
        "color_reference.png": color,
        "color_sample.png": sample,
    }


def build():
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    for name, image in images().items():
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        (assets / name).write_bytes(encoded.tobytes())
    registry = builtin_registry()
    truth = {
        "creator": "OpenCVDataflowGUI contributors",
        "license": "MIT",
        "image_size": [128, 96],
        "shift": [-3, 2],
        "cases": {},
    }
    for case, (sample, reference, parameters, state) in CASES.items():
        graph = Graph()
        for name, path in (("sample", sample), ("reference", reference)):
            source = graph.add(
                registry.create(
                    "input.file",
                    block_id=name,
                    parameters={
                        "path": "assets/" + path,
                        "path_base": "project",
                        "color_format": "bgr" if case == "color_red" else "gray",
                        "pixel_depth": "unchanged",
                    },
                )
            )
            source.position = [0, 0 if name == "sample" else 370]
        measure = graph.add(
            registry.create(
                "inspect.frequency_response",
                block_id="measure",
                parameters={"radial_bins": 32, **parameters},
            )
        )
        measure.position = [620, 0]
        graph.connect("sample", "image", "measure", "image")
        graph.connect("reference", "image", "measure", "reference")
        if case == "masked_identity":
            region = graph.add(
                registry.create(
                    "region.shape",
                    block_id="selection",
                    parameters={"shape": "circle", "center": [64, 48], "radius": 32},
                )
            )
            region.position = [0, 760]
            graph.connect("sample", "image", "selection", "image")
            graph.connect("selection", "mask", "measure", "mask")
        note = graph.add(registry.create("layout.comment", block_id="attribution"))
        note.name, note.position, note.node_size = (
            "Original frequency-response fixture",
            [0, -430],
            "large",
        )
        note.description = (
            "Original MIT images, known transforms and projects by OpenCVDataflowGUI contributors; "
            "no customer/third-party images. Reference samples are multiples of 16, so periodic "
            "[1,2,1]/4 convolution on both axes remains exact uint16 without quantization. Known "
            "gain is (1+cos(2*pi*fx))*(1+cos(2*pi*fy))/4. Shift is (-3,+2) pixels with unit gain "
            "and known Fourier phase. Half gain deliberately fails its tolerance; a four-pixel "
            "grating excites only one conjugate pair. The circle excludes changed exterior pixels. "
            "Color selects Red. Constant reference has no centered energy and cannot pass. "
            "Budget and grid guards deliberately reject before FFT. Equal pixel indices are a "
            "caller assertion of alignment; this does not measure optical MTF certification or "
            "infer general camera calibration from arbitrary photographs."
        )
        graph.save(HERE / (case + ".json"))
        graph.close()
        truth["cases"][case] = {"sample": sample, "reference": reference, "expected_state": state}
    truth["asset_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(assets.glob("*.png"))
    }
    (HERE / "truth.json").write_bytes((json.dumps(truth, indent=2) + "\n").encode())


if __name__ == "__main__":
    build()
