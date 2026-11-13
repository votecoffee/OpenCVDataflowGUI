"""Original synthetic step/noise examples for native stack smoothing (MIT)."""

import json
from pathlib import Path

import cv2
import numpy as np
from compare_filters import compare, fixture, matched_sigma

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent


def main():
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    clean, noisy = fixture()
    for name, image in (
        ("clean.png", clean),
        ("noisy.png", noisy),
        ("noisy16.png", noisy.astype(np.uint16) * 257),
        ("small.png", noisy[:96, :128]),
    ):
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        (assets / name).write_bytes(encoded.tobytes())
    truth = {
        "license": "MIT",
        "creator": "OpenCVDataflowGUI contributors",
        "noise_seed": 159,
        "clean_levels_dn": [60, 180],
        "edge_center_px": 511.5,
        "source_width": 1024,
        "source_height": 768,
        "reference": compare(127, clean, noisy),
        "scope": "Known synthetic step and noise; software checks without optical qualification",
    }
    (HERE / "truth.json").write_bytes(
        (json.dumps(truth, indent=2, allow_nan=False) + "\n").encode()
    )
    registry = builtin_registry()
    for case in (
        "stack_vs_gaussian",
        "selected_smoothing",
        "uint16_smoothing",
        "unsafe_native",
        "kernel_clamp",
    ):
        graph = Graph()

        def add(type_id, block_id, position, parameters=None):
            block = graph.add(
                registry.create(type_id, block_id=block_id, parameters=parameters or {})
            )
            block.position = position
            return block

        path = (
            "small.png"
            if case == "kernel_clamp"
            else "noisy16.png"
            if case in ("uint16_smoothing", "unsafe_native")
            else "noisy.png"
        )
        add(
            "input.file",
            "source",
            [0, 100],
            {
                "path": "assets/" + path,
                "path_base": "project",
                "color_format": "unchanged",
                "pixel_depth": "float32" if case == "stack_vs_gaussian" else "unchanged",
            },
        )
        parameters = {"kernel": 127}
        if case == "unsafe_native":
            parameters["arithmetic"] = "native"
        if case == "kernel_clamp":
            parameters["kernel"] = 255
        add("cv.stack_blur", "stack", [450, 0], parameters)
        graph.connect("source", "image", "stack", "image")
        if case == "stack_vs_gaussian":
            sigma = matched_sigma(127)
            add(
                "cv.gaussian_blur",
                "gaussian",
                [450, 450],
                {
                    "kernel": 127,
                    "sigma": sigma,
                    "sigma_y": sigma,
                    "border": "replicate",
                    "algorithm_hint": "accurate",
                },
            )
            graph.connect("source", "image", "gaussian", "image")
            for name, y in (("stack", 0), ("gaussian", 450)):
                add("cv.intensity_profile", name + "_profile", [900, y])
                graph.connect(name, "image", name + "_profile", "image")
            add("cv.absdiff", "difference", [1350, 200])
            graph.connect("stack", "image", "difference", "image_a")
            graph.connect("gaussian", "image", "difference", "image_b")
        if case == "selected_smoothing":
            add(
                "region.shape",
                "selection",
                [0, 500],
                {"shape": "circle", "center": [512, 384], "radius": 220},
            )
            graph.connect("source", "image", "selection", "image")
            graph.connect("selection", "region", "stack", "region")
            add("cv.absdiff", "difference", [900, 0])
            graph.connect("source", "image", "difference", "image_a")
            graph.connect("stack", "image", "difference", "image_b")
        note = add("layout.comment", "attribution", [0, -420])
        note.name, note.node_size = "Original MIT fixture and smoothing limits", "large"
        note.description = (
            "Original MIT step/noise data and projects from build_example.py; no customer "
            "or third-party imagery. The 1024x768 target has levels 60/180 DN, an ideal edge at "
            "x=511.5 and fixed noise seed 159. Comparison uses the same 127x127 support and a "
            "Gaussian sigma chosen to match the stack kernel's variance. Both filters widen "
            "edges; neither is an edge-preserving denoiser. Profiles and raw differences show "
            "the result. Selection affects writes while full-source pixels supply neighborhoods. "
            "Float32 working arithmetic restores integer output depth by rounding/clipping and "
            "avoids observed high-depth native integer artifacts. Unsafe native mode is an "
            "intentional validation failure. Kernel clamp records 127x95 on the 128x96 input. "
            "See truth.json for independent design and noise/edge measurements. Benchmark results "
            "are separate local native-call timings, excluding graph and display overhead."
        )
        graph.save(HERE / (case + ".json"))
        graph.close()


if __name__ == "__main__":
    main()
