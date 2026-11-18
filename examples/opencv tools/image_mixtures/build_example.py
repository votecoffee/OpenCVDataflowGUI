"""Original controlled color/texture surfaces, native models and saved example graphs."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
CASES = (
    "train_color",
    "train_texture",
    "color_known",
    "color_defect",
    "density_disabled",
    "component_filter",
    "region_color",
    "color_rgb16",
    "spherical_color",
    "posterior_maps",
    "texture_known",
    "texture_defect",
    "texture_support",
)
COLOR_SETTINGS = {
    "feature_space": "bgr",
    "initialization": "manual",
    "initial_means": [[40 / 255, 70 / 255, 180 / 255], [180 / 255, 150 / 255, 40 / 255]],
}
TEXTURE_SETTINGS = {
    "feature_space": "gray_texture",
    "texture_window": 7,
    "initialization": "manual",
    "initial_means": [[120 / 255, 0.005], [120 / 255, 0.12]],
}


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def color(seed):
    image = np.empty((64, 96, 3), np.int16)
    image[:, :48] = [40, 70, 180]
    image[:, 48:] = [180, 150, 40]
    return np.uint8(image + np.random.default_rng(seed).integers(-5, 6, image.shape))


def texture(seed):
    y, x = np.indices((64, 96))
    values = np.full((64, 96), 120, np.int16)
    values[:, 48:] += (((x + y)[:, 48:] % 2) * 2 - 1) * 30
    return np.uint8(values + np.random.default_rng(seed).integers(-1, 2, values.shape))


def texture_mask():
    mask = np.zeros((64, 96), np.uint8)
    mask[4:60, 4:44] = 255
    mask[4:60, 52:92] = 255
    return mask


def build():
    assets = HERE / "assets"
    if assets.exists():
        raise FileExistsError("Existing image mixture assets preserved")
    registry = builtin_registry()
    query = color(21)
    defect = query.copy()
    defect[14:24, 18:30] = [5, 240, 10]
    color_truth = np.zeros((64, 96), np.uint8)
    color_truth[14:24, 18:30] = 255
    texture_query = texture(22)
    texture_defect = texture_query.copy()
    y, x = np.indices((16, 16))
    texture_defect[20:36, 64:80] = 120 + (((x + y) % 2) * 2 - 1) * 15
    texture_truth = np.zeros((64, 96), np.uint8)
    texture_truth[20:36, 64:80] = 255
    images = {
        "train-color.png": color(11),
        "query-color.png": query,
        "defect-color.png": defect,
        "query16.png": query.astype(np.uint16) * 257,
        "color-truth.png": color_truth,
        "train-texture.png": texture(12),
        "query-texture.png": texture_query,
        "defect-texture.png": texture_defect,
        "texture-truth.png": texture_truth,
    }
    files = {}
    for name, image in images.items():
        ok, data = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("Could not encode " + name)
        files[name] = data.tobytes()
    for name, settings in (
        ("color", COLOR_SETTINGS),
        ("spherical", {**COLOR_SETTINGS, "covariance_type": "spherical"}),
        ("texture", TEXTURE_SETTINGS),
    ):
        block = registry.create("ml.em_train_image", parameters=settings)
        try:
            inputs = {
                "image": images["train-texture.png" if name == "texture" else "train-color.png"]
            }
            if name == "texture":
                inputs["mask"] = texture_mask()
            result = block.execute(inputs)
            if result is None:
                raise RuntimeError(block.error)
            files[f"model-{name}.json"] = encoded(result.outputs["model"])
        finally:
            block.close()
    assets.mkdir()
    for name, raw in files.items():
        (assets / name).write_bytes(raw)
    cases = {}
    for case in CASES:
        graph = Graph()

        def add(kind, block_id, x, y, **parameters):
            block = graph.add(registry.create(kind, block_id=block_id, parameters=parameters))
            block.position = [x, y]
            return block

        try:
            training = case.startswith("train")
            is_texture = "texture" in case
            filename = (
                "train-texture.png"
                if case == "train_texture"
                else "train-color.png"
                if training
                else ("defect-texture.png" if case == "texture_defect" else "query-texture.png")
                if is_texture
                else (
                    "defect-color.png"
                    if case in ("color_defect", "density_disabled", "region_color")
                    else "query16.png"
                    if case == "color_rgb16"
                    else "query-color.png"
                )
            )
            add(
                "input.file",
                "source",
                0,
                0,
                path="assets/" + filename,
                path_base="project",
                color_format="gray" if is_texture else "rgb" if case == "color_rgb16" else "bgr",
                pixel_depth="uint16" if case == "color_rgb16" else "uint8",
            )
            if training:
                target = add(
                    "ml.em_train_image",
                    "train",
                    720,
                    0,
                    **(TEXTURE_SETTINGS if is_texture else COLOR_SETTINGS),
                )
                add("ml.em_save", "save", 1080, 0, path="trained-mixture.json")
                graph.connect("source", "image", "train", "image")
                graph.connect("train", "model", "save", "model")
                state, end = "pass", "save"
            else:
                mode = (
                    "texture"
                    if is_texture
                    else "spherical"
                    if case == "spherical_color"
                    else "color"
                )
                add("ml.em_load", "model", 360, 450, path=f"assets/model-{mode}.json")
                settings = {"min_log_likelihood": 2.0} if is_texture else {}
                if case == "density_disabled":
                    settings["check_log_likelihood"] = False
                if case == "component_filter":
                    settings["accepted_components"] = [0]
                if case == "texture_support":
                    settings["minimum_support_fraction"] = 1.0
                target = add("ml.em_classify_image", "inspect", 720, 0, **settings)
                target.exposed_outputs.append("result_graphics")
                if case == "posterior_maps":
                    target.exposed_outputs.append("posteriors")
                add("cv.overlay_result_graphics", "overlay", 1080, 0)
                add("output.filmstrip_image", "filmstrip", 1440, 0)
                for a, b, c, d in (
                    ("source", "image", "inspect", "image"),
                    ("model", "model", "inspect", "model"),
                    ("inspect", "image", "overlay", "image"),
                    ("inspect", "result_graphics", "overlay", "graphics_1"),
                    ("overlay", "image", "filmstrip", "image"),
                ):
                    graph.connect(a, b, c, d)
                state = (
                    "fail"
                    if case
                    in ("color_defect", "component_filter", "texture_defect", "texture_support")
                    else "pass"
                )
                end = "inspect"
            if is_texture:
                for block_id, rectangle, y_position in (
                    ("left", [4, 4, 40, 56], 400),
                    ("right", [52, 4, 40, 56], 760),
                ):
                    add("region.shape", block_id, 0, y_position, rectangle=rectangle)
                    graph.connect("source", "image", block_id, "image")
                add("region.combine", "support", 360, 820)
                graph.connect("source", "image", "support", "image")
                graph.connect("left", "region", "support", "region_a")
                graph.connect("right", "region", "support", "region_b")
                graph.connect("support", "mask", target.id, "mask")
            elif case == "region_color":
                add("region.shape", "selection", 360, 820, rectangle=[54, 6, 36, 50])
                graph.connect("source", "image", "selection", "image")
                graph.connect("selection", "region", "inspect", "region")
            note = add("layout.comment", "attribution", 0, -480)
            note.name, note.node_size = "Original image surface mixture controls", "large"
            note.description = (
                "Original MIT synthetic images, truth masks, models and graphs by "
                "OpenCVDataflowGUI contributors. Training and held-out images use separate "
                "deterministic noise; queries are excluded from fitting. The color anomaly is "
                "exactly 120 pixels. Texture decisions refer to supported neighborhood centers "
                "and can extend beyond the altered patch. Relative component probability is not "
                "calibrated novelty confidence. Explicit density, component and support gates "
                "decide pass/fail; regularized covariance is saved in the model. Spherical fits "
                "use equivalent diagonal prediction. No production accuracy or radiometric "
                "calibration claim. See README.md, benchmark.json and SOURCES.md."
            )
            graph.save(HERE / (case + ".json"))
            cases[case] = {"target": end, "state": state, "image": filename}
        finally:
            graph.close()
    (HERE / "truth.json").write_bytes(
        encoded(
            {
                "cases": cases,
                "asset_sha256": {
                    name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()
                },
                "color_defect_pixels": 120,
                "texture_patch_pixels": 256,
                "texture_selection": [[4, 4, 40, 56], [52, 4, 40, 56]],
            }
        )
    )


if __name__ == "__main__":
    build()
