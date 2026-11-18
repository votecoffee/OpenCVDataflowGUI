"""Original known-detail images and native learned/interpolated comparison graphs."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
SHA256 = "366b33f0084c7b3f2bf6724f0a2c77bca94fcec9d7b6d72389d330073b380d5c"
CASES = (
    "geometry",
    "color",
    "alpha",
    "gray16",
    "selected_region",
    "empty_region",
    "fidelity_alarm",
    "noise",
)


def asset(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != raw:
        raise FileExistsError("Different existing assets preserved; build in a fresh directory")
    path.write_bytes(raw)


def fixtures():
    reference = np.full((192, 256), 180, np.uint8)
    cv2.circle(reference, (70, 88), 36, 50, -1)
    reference[40:66, 125:188] = 50
    reference[99:108, 132:237] = 50
    reference[99:108, 181:183] = 180
    for index, width in enumerate((1, 2, 3, 4)):
        x = 140 + index * 24
        reference[146:169, x : x + width] = 50
    yy, xx = np.indices(reference.shape)
    color = np.dstack(
        (
            reference,
            np.clip(reference.astype(int) + xx // 8, 0, 255),
            np.clip(reference.astype(int) - yy // 8, 0, 255),
        )
    ).astype(np.uint8)
    alpha = np.dstack((color, np.rint(xx / 255 * 255).astype(np.uint8)))
    rng = np.random.default_rng(435)
    noisy = np.rint(np.clip(reference + rng.normal(0, 8, reference.shape), 0, 255)).astype(np.uint8)
    return {
        "reference": reference,
        "color": color,
        "alpha": alpha,
        "gray16": reference.astype(np.uint16) * 257,
        "noise": noisy,
    }


def build():
    images = fixtures()
    for name, image in images.items():
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        asset(HERE / "assets" / (name + ".png"), encoded.tobytes())
    registry = builtin_registry()
    for name in CASES:
        graph = Graph()
        try:
            graph.program["assets"] = ["MODEL-LICENSE.txt", "SOURCES.md"]

            def add(kind, identity, position, **parameters):
                block = graph.add(registry.create(kind, block_id=identity, parameters=parameters))
                block.position = list(position)
                return block

            image = name if name in images else "reference"
            add(
                "input.file",
                "source",
                (0, 0),
                path=f"assets/{image}.png",
                path_base="project",
                color_format="unchanged",
                pixel_depth="unchanged",
            )
            add("cv.resize", "downsample", (400, 0), width=128, height=96, interpolation="area")
            sr = add(
                "ml.super_resolution",
                "learned",
                (800, 0),
                model_path="assets/FSRCNN_x2.pb",
                model_sha256=SHA256,
                depth_mode="convert_full_scale" if name == "gray16" else "uint8_only",
            )
            sr.node_size = "large"
            add(
                "inspect.image_quality",
                "learned_error",
                (1240, -160),
                metric="mae",
                check_threshold=name == "fidelity_alarm",
                threshold=0.0,
            )
            add("inspect.image_quality", "baseline_error", (1240, 220), metric="mae")
            add("output.filmstrip_image", "filmstrip", (1640, -20))
            for a, out, b, into in (
                ("source", "image", "downsample", "image"),
                ("downsample", "image", "learned", "image"),
                ("source", "image", "learned_error", "image_a"),
                ("learned", "image", "learned_error", "image_b"),
                ("source", "image", "baseline_error", "image_a"),
                ("learned", "baseline", "baseline_error", "image_b"),
                ("learned", "image", "filmstrip", "image"),
            ):
                graph.connect(a, out, b, into)
            if name in ("selected_region", "empty_region"):
                add(
                    "region.shape",
                    "region",
                    (400, 420),
                    shape="circle",
                    center=[35, 44],
                    radius=22 if name == "selected_region" else 1,
                    rectangle=[0, 0, 20, 20],
                )
                if name == "empty_region":
                    graph.blocks["region"].parameters["center"] = [-100, -100]
                graph.connect("downsample", "image", "region", "image")
                graph.connect("region", "region", "learned", "region")
            note = add("layout.comment", "attribution", (0, -590))
            note.name = "FSRCNN source / reconstruction limits"
            note.node_size = "large"
            note.description = (
                "Original MIT geometric/color/noise controls by OpenCVDataflowGUI contributors. "
                "Unmodified Apache-2.0 FSRCNN_x2.pb from Saafke/FSRCNN_Tensorflow, revision "
                "6a4812c4ef1c4f5947d79beafa32a05a6eb4a94d. See SOURCES.md/MODEL-LICENSE.txt. "
                "Each source is the native-resolution reference; area reduction removes detail. "
                "Compare learned and cubic errors, processed selection and retained benchmark. "
                "Reconstruction can invent/erase defects. Nominal grid mapping is not new "
                "measured resolution or calibrated geometry. Model execution passing does not "
                "mean inspection fidelity passes; fidelity_alarm intentionally fails its "
                "zero-error check."
            )
            graph.save(HERE / (name + ".json"))
        finally:
            graph.close()
    truth = {
        "cases": list(CASES),
        "model_sha256": SHA256,
        "scale": 2,
        "source_size": [256, 192],
        "reduced_size": [128, 96],
        "dark_threshold": 110,
        "circle_roi": [30, 45, 82, 86],
        "gap_roi": [170, 96, 24, 15],
        "detail_roi": [128, 138, 112, 38],
        "flat_roi": [4, 4, 32, 28],
        "native_circle": {"center": [70, 88], "radius": 36},
        "native_crack_widths": [1, 2, 3, 4],
        "native_bead_gap_width": 2,
        "asset_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((HERE / "assets").iterdir())
        },
    }
    asset(HERE / "truth.json", (json.dumps(truth, indent=2) + "\n").encode())


if __name__ == "__main__":
    build()
