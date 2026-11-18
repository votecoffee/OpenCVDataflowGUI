"""Known native-resolution detail and portable multi-scale LapSRN workflows."""

import hashlib
import importlib.util
import json
from pathlib import Path

import cv2

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
DIGEST = "a00211938b2b4c7d23691525a9267d9d0bbb345cdb9f38a7fc917414a5a1884f"
CASES = (
    "geometry",
    "color",
    "alpha",
    "gray16",
    "float",
    "selected_region",
    "empty_region",
    "reordered",
    "subset",
    "wrong_scale",
    "budget_error",
    "noise",
    "coffee",
)


def fixtures():
    path = HERE.parent / "super_resolution/build_example.py"
    spec = importlib.util.spec_from_file_location("original_superres_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fixtures()


def build(destination=HERE, model_source=None):
    destination = Path(destination)
    assets = destination / "assets"
    if assets.exists():
        raise FileExistsError("Existing multi-scale assets preserved")
    model = Path(model_source) if model_source is not None else HERE / "assets/LapSRN_x8.pb"
    raw = model.read_bytes()
    if len(raw) != 4069183 or hashlib.sha256(raw).hexdigest() != DIGEST:
        raise ValueError("LapSRN bytes differ from the pinned Apache-2 export")
    assets.mkdir(parents=True)
    (assets / "LapSRN_x8.pb").write_bytes(raw)
    (destination / ".gitattributes").write_bytes(
        b"*.json -text\nMODEL-LICENSE.txt -text\n.gitattributes -text\n"
    )
    for name in ("MODEL-LICENSE.txt", "SOURCES.md"):
        source = HERE / name
        if source.exists() and destination != HERE:
            (destination / name).write_bytes(source.read_bytes())
    for name, image in fixtures().items():
        (assets / (name + ".png")).write_bytes(cv2.imencode(".png", image)[1].tobytes())
    registry = builtin_registry()
    for case in CASES:
        graph = Graph()
        try:
            graph.program["assets"] = ["MODEL-LICENSE.txt", "SOURCES.md"]

            def add(kind, key, x, y=0, **params):
                b = graph.add(registry.create(kind, block_id=key, parameters=params))
                b.position = [x, y]
                return b

            name = case if case in fixtures() else "reference"
            add(
                "input.file",
                "source",
                0,
                path="../../assets/coffee.png" if case == "coffee" else f"assets/{name}.png",
                path_base="project",
                color_format="unchanged",
                pixel_depth="float32" if case == "float" else "unchanged",
            )
            add("cv.resize", "downsample", 380, width=32, height=24, interpolation="area")
            settings = {
                "model_path": "assets/LapSRN_x8.pb",
                "model_sha256": DIGEST,
                "depth_mode": "convert_full_scale" if case in ("gray16", "float") else "uint8_only",
            }
            if case == "reordered":
                settings.update(
                    output_scales=[8, 2, 4],
                    output_nodes=["NCHW_output_8x", "NCHW_output_2x", "NCHW_output_4x"],
                    preview_scale="2",
                )
            if case == "subset":
                settings.update(
                    output_scales=[2, 4], output_nodes=["NCHW_output_2x", "NCHW_output_4x"]
                )
            if case == "wrong_scale":
                settings["output_nodes"] = ["NCHW_output_4x", "NCHW_output_2x", "NCHW_output_8x"]
            if case == "budget_error":
                settings["maximum_output_pixels"] = 100
            sr = add("ml.super_resolution_multi", "learned", 760, **settings)
            sr.exposed_outputs.append("result_graphics")
            add("cv.overlay_result_graphics", "overlay", 1150)
            add("output.filmstrip_image", "filmstrip", 1530)
            for a, o, b, i in [
                ("source", "image", "downsample", "image"),
                ("downsample", "image", "learned", "image"),
                ("learned", "image", "overlay", "image"),
                ("learned", "result_graphics", "overlay", "graphics_1"),
                ("overlay", "image", "filmstrip", "image"),
            ]:
                graph.connect(a, o, b, i)
            if case in ("selected_region", "empty_region"):
                add(
                    "region.shape",
                    "selection",
                    380,
                    420,
                    rectangle=[0, 0, 16, 24] if case == "selected_region" else [80, 80, 1, 1],
                )
                graph.connect("downsample", "image", "selection", "image")
                graph.connect("selection", "region", "learned", "region")
            if case == "geometry":
                for i, scale in enumerate((2, 4, 8)):
                    sr.exposed_outputs.append(f"image_x{scale}")
                    add(
                        "cv.threshold",
                        f"threshold_{scale}",
                        1150,
                        420 + i * 260,
                        value=110,
                        invert=True,
                    )
                    add("inspect.blobs", f"blobs_{scale}", 1530, 420 + i * 260, min_area=1)
                    graph.connect("learned", f"image_x{scale}", f"threshold_{scale}", "image")
                    graph.connect(f"threshold_{scale}", "image", f"blobs_{scale}", "image")
            note = add("layout.comment", "attribution", 0, -460)
            note.name, note.node_size = "LapSRN multi-scale sources and limits", "large"
            note.description = (
                "Graph and synthetic fixture: original OpenCVDataflowGUI MIT work. "
                "Unmodified pretrained LapSRN x8 export by Fanny Monori / TF-LapSRN, Apache 2.0, "
                "revision fc51c90af1b5801a357abc919160d7ff4f24b997. One native named forward "
                "delivers requested scales. Pixel grids are nominal; "
                "learned detail is reconstruction "
                "and may erase or invent inspection features. Not calibrated metrology. "
                + (
                    "Coffee photograph by Rachel Michetti, courtesy Pikolo Espresso Bar, "
                    "CC0 via scikit-image. "
                    if case == "coffee"
                    else ""
                )
                + "See MODEL-LICENSE.txt and SOURCES.md, retained on Save As."
            )
            graph.save(destination / (case + ".json"))
        finally:
            graph.close()
    truth = {
        "cases": list(CASES),
        "model_sha256": DIGEST,
        "scales": [2, 4, 8],
        "reference_size": [256, 192],
        "reduced_size": [32, 24],
        "circle_center": [70, 88],
        "circle_roi": [30, 45, 82, 86],
        "dark_threshold": 110,
        "asset_sha256": {
            p.relative_to(destination).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(assets.iterdir())
        },
    }
    (destination / "truth.json").write_bytes(
        (json.dumps(truth, sort_keys=True, indent=2) + "\n").encode()
    )


if __name__ == "__main__":
    build()
