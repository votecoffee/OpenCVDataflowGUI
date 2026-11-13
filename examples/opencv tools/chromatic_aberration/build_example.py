"""Regenerate the original calibration, images and three saved inspection graphs."""

import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 240, 160
COEFFICIENTS = [[-2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8", newline="\n"
    )


def build():
    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    y, x = np.indices((HEIGHT, WIDTH))
    plate = 10000 + 20 * x + 35 * y + ((x + 3 * y) % 9) * 110
    plate[30:132, 30:210] += 18000
    for cx in (58, 180):
        plate[(x - cx) ** 2 + (y - 80) ** 2 <= 17**2] = 5000
    plate[65:95, 105:135] = 7000
    ideal = np.repeat(plate[..., None], 3, axis=-1).astype(np.uint16)
    observed = np.empty_like(ideal)
    for channel, (dx, dy) in enumerate(((-2, 1), (0, 0), (3, -2))):
        observed[..., channel] = ideal[
            np.clip(y + dy, 0, HEIGHT - 1), np.clip(x + dx, 0, WIDTH - 1), channel
        ]
    assert cv2.imwrite(str(assets / "fringed_rgb16.png"), observed)
    assert cv2.imwrite(str(assets / "ideal_rgb16.png"), ideal)
    write_json(
        assets / "chromatic_coefficients.json",
        {
            "image_width": WIDTH,
            "image_height": HEIGHT,
            "blue_channel": {"coeffs_x": COEFFICIENTS[0], "coeffs_y": COEFFICIENTS[1]},
            "red_channel": {"coeffs_x": COEFFICIENTS[2], "coeffs_y": COEFFICIENTS[3]},
        },
    )
    write_json(
        ROOT / "expected.json",
        {
            "image_size": [WIDTH, HEIGHT],
            "native_bgr_displacements": [[-2, 1], [0, 0], [3, -2]],
            "valid_rectangle_xywh": [3, 1, WIDTH - 5, HEIGHT - 3],
            "valid_pixels": (WIDTH - 5) * (HEIGHT - 3),
            "total_pixels": WIDTH * HEIGHT,
            "corrected_max_absolute_error": 0,
            "before_max_absolute_error": int(
                np.abs(
                    observed[1 : HEIGHT - 2, 3 : WIDTH - 2].astype(np.int32)
                    - ideal[1 : HEIGHT - 2, 3 : WIDTH - 2].astype(np.int32)
                ).max()
            ),
            "source_dtype": "uint16",
            "failed_support_minimum": 1.0,
        },
    )
    registry, graph = builtin_registry(), Graph()

    def add(kind, ident, name, position, parameters=None, preview=""):
        block = registry.create(kind, block_id=ident, parameters=parameters or {})
        block.name, block.position, block.preview_output = name, list(position), preview
        graph.add(block)
        return block

    for ident, filename, pos in (
        ("source", "fringed_rgb16.png", (0, 180)),
        ("reference", "ideal_rgb16.png", (0, 520)),
    ):
        add(
            "input.file",
            ident,
            filename,
            pos,
            {
                "path": "assets/" + filename,
                "path_base": "project",
                "color_format": "rgb",
                "pixel_depth": "uint16",
                "filename_ticket": True,
            },
        )
    add(
        "input.chromatic_calibration",
        "calibration",
        "Native calibration file",
        (0, -160),
        {
            "path": "assets/chromatic_coefficients.json",
            "path_base": "project",
            "camera_label": "Original synthetic camera",
            "lens_label": "Known integer channel shifts",
        },
    )
    add("cv.correct_chromatic_aberration", "correct", "Align R/B with green", (340, 180))
    add(
        "inspect.image_quality",
        "before",
        "Fringe error before correction",
        (700, 490),
        {
            "metric": "max_absolute_error",
            "peak_value": 65535,
            "mask_alignment": "pixel_grid",
        },
    )
    add(
        "inspect.image_quality",
        "after",
        "Exact corrected pixels",
        (700, 160),
        {
            "metric": "max_absolute_error",
            "peak_value": 65535,
            "mask_alignment": "pixel_grid",
            "check_threshold": True,
            "threshold": 0,
        },
    )
    add("cv.overlay_result_graphics", "overlay", "Support diagnostic overlay", (1050, 160))
    add("output.filmstrip_image", "filmstrip", "Final corrected image", (1390, 160))
    note = add("layout.comment", "notes", "Calibrated color-fringe correction", (690, -200))
    note.node_size = "large"
    note.description = (
        "Original 16-bit neutral part has B(-2,+1) and R(+3,-2) pixel displacements. "
        "The correction uses green as the pixel grid and retains sample depth. "
        "Inspect valid_mask: only the independently known inner rectangle is supported. "
        "Before error is nonzero; after error must be exactly zero there. "
        "Original graph, fixture and calibration: OpenCVDataflowGUI (MIT). "
        "Native API reference: docs.opencv.org/5.0/main_modules/photo_ca_correction.html"
    )
    for args in (
        ("source", "image", "correct", "image"),
        ("calibration", "calibration", "correct", "calibration"),
        ("source", "image", "before", "image_a"),
        ("correct", "image", "after", "image_a"),
        ("reference", "image", "before", "image_b"),
        ("reference", "image", "after", "image_b"),
        ("correct", "valid_mask", "before", "mask"),
        ("correct", "valid_mask", "after", "mask"),
        ("correct", "image", "overlay", "image"),
        ("correct", "result_graphics", "overlay", "graphics_1"),
        ("overlay", "image", "filmstrip", "image"),
    ):
        source, output, target, terminal = args
        if output not in graph.blocks[source].exposed_outputs:
            graph.blocks[source].exposed_outputs.append(output)
        if terminal not in graph.blocks[target].exposed_inputs:
            graph.blocks[target].exposed_inputs.append(terminal)
        graph.connect(*args)
    definition = graph.to_dict()
    write_json(ROOT / "chromatic_aberration.json", definition)
    defined = json.loads(json.dumps(definition))
    for item in defined["blocks"]:
        if item["id"] == "calibration":
            block = registry.create(
                "calibration.chromatic_model",
                block_id="calibration",
                parameters={
                    "width": WIDTH,
                    "height": HEIGHT,
                    "degree": 1,
                    "coefficients": COEFFICIENTS,
                    "camera_label": "Original synthetic camera",
                    "lens_label": "Known integer channel shifts",
                },
            )
            block.position, block.name = item["position"], "Editable calibration coefficients"
            item.clear()
            item.update(block.to_project_item())
    write_json(ROOT / "defined_model.json", defined)
    for item in definition["blocks"]:
        if item["id"] == "correct":
            item["parameters"]["minimum_valid_fraction"] = 1.0
    write_json(ROOT / "insufficient_support.json", definition)
    graph.close()


if __name__ == "__main__":
    build()
