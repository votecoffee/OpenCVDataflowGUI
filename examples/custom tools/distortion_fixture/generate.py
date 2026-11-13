"""Original machined-part image and independently evaluated synthetic warps."""

import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

NAMES = ("shear_x", "shear_y", "perspective_x", "perspective_y", "k1", "k2", "p1", "p2")
CASES = {
    "shear": {"shear_x": 0.125, "center_y": 0.0},
    "perspective": {"perspective_x": 0.06, "perspective_y": -0.04},
    "lens": {"k1": 0.08, "k2": 0.015, "p1": 0.002, "p2": -0.003},
    "seeded": {
        "coefficient_mode": "uniform_bounds",
        "seed": 23,
        "shear_x": 0.08,
        "perspective_y": 0.02,
        "k1": 0.03,
    },
}


def truth_coordinates(points, size, parameters):
    """Independent homogeneous matrix evaluation; never imports the implementation."""
    c = {name: float(parameters.get(name, 0)) for name in NAMES}
    if parameters.get("coefficient_mode") == "uniform_bounds":
        rng = np.random.Generator(np.random.PCG64(parameters["seed"]))
        c = {name: float(rng.uniform(-value, value)) for name, value in c.items()}
    center = (np.array(size) - 1) * [
        parameters.get("center_x", 0.5),
        parameters.get("center_y", 0.5),
    ]
    radius = max(size) * parameters.get("radius_scale", 0.5)
    vectors = np.c_[(points - center) / radius, np.ones(len(points))]
    matrix = np.array(
        [[1, c["shear_x"], 0], [c["shear_y"], 1, 0], [c["perspective_x"], c["perspective_y"], 1]]
    )
    projected = vectors @ matrix.T
    a, b = (projected[:, :2] / projected[:, 2:]).T
    r2 = a * a + b * b
    radial = 1 + r2 * (c["k1"] + r2 * c["k2"])
    values = np.c_[
        a * radial + 2 * c["p1"] * a * b + c["p2"] * (3 * a * a + b * b),
        b * radial + c["p1"] * (a * a + 3 * b * b) + 2 * c["p2"] * a * b,
    ]
    return values * radius + center, c


def main():
    folder = Path(__file__).resolve().parent
    assets = folder / "assets"
    assets.mkdir(exist_ok=True)
    y, x = np.indices((128, 160), np.float64)
    gray = 3500 + 35 * x + 19 * y
    gray[20:108, 24:136] = 31000 + 50 * x[20:108, 24:136] + 15 * y[20:108, 24:136]
    cv2.circle(gray, (51, 51), 14, 6000, -1)
    cv2.circle(gray, (112, 82), 18, 7000, -1)
    cv2.rectangle(gray, (75, 32), (120, 44), 54000, -1)
    gray[80:96, 42:70] = 52000
    source = np.rint(np.dstack([gray, gray * 0.73, gray * 0.46])).astype(np.uint16)
    assert cv2.imwrite(str(assets / "source.png"), source[..., ::-1])
    points = np.array([[px, py] for py in (0, 63.5, 127) for px in (0, 79.5, 159)])
    all_points = np.c_[x.ravel(), y.ravel()]
    truth = {
        "origin": "Original 16-bit RGB part and independent matrix/polynomial references; MIT",
        "image_size": [160, 128],
        "direction": "output_to_input",
        "output_probes": points.tolist(),
        "cases": {},
    }
    for name, parameters in CASES.items():
        donor, coefficients = truth_coordinates(all_points, [160, 128], parameters)
        mx = donor[:, 0].reshape(128, 160).astype(np.float32)
        my = donor[:, 1].reshape(128, 160).astype(np.float32)
        golden = cv2.remap(source, mx, my, cv2.INTER_LINEAR)
        assert cv2.imwrite(str(assets / (name + ".png")), golden[..., ::-1])
        truth["cases"][name] = {
            "parameters": parameters,
            "coefficients": coefficients,
            "input_probes": truth_coordinates(points, [160, 128], parameters)[0].tolist(),
            "maximum_image_error_counts": 0,
            "maximum_probe_error_pixels": 1e-11,
        }
    (assets / "truth.json").write_text(
        json.dumps(truth, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    registry = builtin_registry()
    graph = Graph()
    graph.set_branch_workers(2)
    try:
        source_block = graph.add(
            registry.create(
                "input.file",
                block_id="source",
                parameters={
                    "path": "assets/source.png",
                    "path_base": "project",
                    "color_format": "rgb",
                    "pixel_depth": "unchanged",
                },
            )
        )
        source_block.position = [0, 350]
        source_block.node_size = "large"
        for index, (name, parameters) in enumerate(CASES.items()):
            row = index * 480
            block = graph.add(
                registry.create("cv.distortion_fixture", block_id=name, parameters=parameters)
            )
            block.name = name.title() + " fixture"
            block.position = [420, row]
            block.node_size = "large"
            block.exposed_outputs = ["image", "map_x", "map_y", "valid_mask"]
            graph.connect("source", "image", name, "image")
            reuse = graph.add(registry.create("cv.remap", block_id="reuse_" + name))
            reuse.name = "Reuse " + name + " maps"
            reuse.position = [860, row]
            graph.connect("source", "image", reuse.id, "image")
            for axis in ("x", "y"):
                graph.connect(name, "map_" + axis, reuse.id, "map_" + axis)
            golden = graph.add(
                registry.create(
                    "input.file",
                    block_id="golden_" + name,
                    parameters={
                        "path": "assets/" + name + ".png",
                        "path_base": "project",
                        "color_format": "rgb",
                        "pixel_depth": "unchanged",
                    },
                )
            )
            golden.position = [860, row + 250]
            golden.name = "Independent " + name + " reference"
            verify = graph.add(
                registry.create(
                    "inspect.image_quality",
                    block_id="verify_" + name,
                    parameters={
                        "metric": "max_absolute_error",
                        "check_threshold": True,
                        "threshold": 0,
                    },
                )
            )
            verify.name = "Require exact " + name + " pixels"
            verify.position = [1300, row]
            graph.connect(reuse.id, "image", verify.id, "image_a")
            graph.connect(golden.id, "image", verify.id, "image_b")
        note = graph.add(registry.create("layout.comment", block_id="about"))
        note.name = "Original distortion fixtures - MIT"
        note.position = [0, -260]
        note.node_size = "large"
        note.description = (
            "Original 16-bit RGB target, fixture equations, references and project under the "
            "repository's MIT license. Four branches generate shear, perspective, lens-like and "
            "seeded distortions, then reuse their typed maps through ordinary Remap blocks. "
            "Each comparison must have zero maximum pixel error. Truth maps are output-to-input "
            "sampling coordinates, not estimated registration or a guessed forward transform. "
            "Probes and effective coefficients are in assets/truth.json. Validity excludes source "
            "borders and selections conservatively for interpolation support. Fixed-point maps "
            "can introduce extra quantization; float64 cubic remap is rejected because the pinned "
            "native build corrupts boundary samples. This is a synthetic stress model, not a "
            "camera calibration. API: https://docs.opencv.org/5.0/main_modules/imgproc_transform.html"
        )
        graph.save(folder / "project.json")
    finally:
        graph.close()


if __name__ == "__main__":
    main()
