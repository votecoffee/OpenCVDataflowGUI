"""Original MIT controls: analytic Brown distortion and independent four-tap support."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
PROFILE = {
    "model": "opencv_pinhole_v1",
    "camera_matrix": [[110, 0, 79.5], [0, 115, 59.5], [0, 0, 1]],
    "distortion": [-0.3, 0.05, 0.003, -0.002, 0],
    "image_size": [160, 120],
    "rms_pixels": 0.1,
    "units": "mm",
}
CASES = ("lens", "coverage", "packed", "wide", "wrong_pair")


def analytic_maps(size, camera):
    """Destination pinhole rays -> Brown5 distortion -> original camera pixels."""
    yy, xx = np.indices(size[::-1], np.float64)
    x, y = (xx - camera[0][2]) / camera[0][0], (yy - camera[1][2]) / camera[1][1]
    r2 = x * x + y * y
    radial = 1 - 0.3 * r2 + 0.05 * r2**2
    dx = x * radial + 2 * 0.003 * x * y - 0.002 * (r2 + 2 * x * x)
    dy = y * radial + 0.003 * (r2 + 2 * y * y) - 2 * 0.002 * x * y
    return (110 * dx + 79.5).astype(np.float32), (115 * dy + 59.5).astype(np.float32)


def four_taps(source, x, y, *, packed=False):
    """No native sampling: float maps retain fractions; packed maps use 1/32 phases."""
    x, y = x.astype(np.float64), y.astype(np.float64)
    if packed:
        x, y = np.rint(x * 32) / 32, np.rint(y * 32) / 32
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    result = np.zeros(x.shape, np.float64)
    for dy, wy in ((0, 1 - (y - iy)), (1, y - iy)):
        for dx, wx in ((0, 1 - (x - ix)), (1, x - ix)):
            xx, yy = ix + dx, iy + dy
            valid = (xx >= 0) & (xx < source.shape[1]) & (yy >= 0) & (yy < source.shape[0])
            result[valid] += source[yy[valid], xx[valid]] * wx[valid] * wy[valid]
    return result


def write_json(path, value):
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def build():
    assets = HERE / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    yy, xx = np.indices((120, 160))
    image = np.stack(
        (1000 + 100 * xx + 10 * yy, 2000 + 35 * xx + 40 * yy, 500 + 60 * xx + 15 * yy), axis=2
    ).astype(np.uint16)
    ok, encoded = cv2.imencode(".png", image[..., ::-1])
    assert ok
    (assets / "ramp_rgb16.png").write_bytes(encoded.tobytes())
    write_json(assets / "lens.json", PROFILE)
    source_mask = (xx - 68) ** 2 + (yy - 56) ** 2 <= 36**2
    source_mask[46:66, 64:82] = False
    truth = {
        "license": "MIT - original repository fixtures",
        "cases": {},
        "source_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    registry = builtin_registry()
    for case in CASES:
        size = [240, 180] if case == "wide" else [160, 120]
        camera = (
            [[65, 0, 119.5], [0, 70, 89.5], [0, 0, 1]]
            if case == "wide"
            else PROFILE["camera_matrix"]
        )
        x, y = analytic_maps(size, camera)
        packed = case == "packed"
        coverage = four_taps(source_mask, x, y, packed=packed)
        support = four_taps(np.ones(source_mask.shape), x, y, packed=packed) >= 1 - 1e-6
        selected = (coverage >= (0.5 if case == "coverage" else 1 - 1e-6)) & support
        sy, sx = np.nonzero(selected)
        truth["cases"][case] = {
            "expected_error": case == "wrong_pair",
            "output_size": size,
            "selected_pixels": len(sx),
            "valid_pixels": int(support.sum()),
            "mask_sha256": hashlib.sha256(selected.astype(np.uint8) * 255).hexdigest(),
            "centroid": [float(sx.mean()), float(sy.mean())],
            "channel_means": [
                float(four_taps(image[..., c], x, y, packed=packed)[selected].mean())
                for c in range(3)
            ],
            "mean_absolute_tolerance": 1.0,
        }
        if case == "wrong_pair":
            truth["cases"][case] = {"expected_error": True, "reason": "mismatched lens-map pair"}
        graph = Graph()
        try:

            def add(kind, key, parameters=None, position=(0, 0)):
                block = graph.add(registry.create(kind, block_id=key, parameters=parameters or {}))
                block.position = list(position)
                return block

            add(
                "input.file",
                "source",
                {
                    "path": "assets/ramp_rgb16.png",
                    "path_base": "project",
                    "color_format": "rgb",
                    "pixel_depth": "uint16",
                },
            )
            add(
                "input.calibration",
                "profile",
                {"path": "assets/lens.json", "path_base": "project"},
                (0, 350),
            )
            add(
                "calibration.lens_maps",
                "maps",
                {
                    "width": size[0],
                    "height": size[1],
                    "camera_matrix_mode": "custom",
                    "new_camera_matrix": camera,
                },
                (380, 350),
            )
            add(
                "region.shape",
                "circle",
                {"shape": "circle", "center": [68, 56], "radius": 36},
                (380, 0),
            )
            add("region.shape", "hole", {"rectangle": [64, 46, 18, 20]}, (380, -350))
            add("region.combine", "selection", {"operation": "difference"}, (760, 0))
            remap = add(
                "cv.remap_selection_packed" if case == "packed" else "cv.remap_selection",
                "remap",
                {
                    "border": "replicate",
                    **({"selection_rule": "coverage_threshold"} if case == "coverage" else {}),
                },
                (1140, 0),
            )
            remap.node_size = "large"
            remap.exposed_outputs.append("result_graphics")
            add("region.measure", "measure", position=(1520, 350))
            add("cv.image_statistics", "statistics", position=(1520, 700))
            add("cv.overlay_result_graphics", "overlay", position=(1520, 0))
            add("output.filmstrip_image", "filmstrip", position=(1900, 0))
            wires = [
                ("profile", "profile", "maps", "calibration"),
                ("source", "image", "circle", "image"),
                ("source", "image", "hole", "image"),
                ("source", "image", "selection", "image"),
                ("circle", "region", "selection", "region_a"),
                ("hole", "region", "selection", "region_b"),
                ("source", "image", "remap", "image"),
                ("selection", "region", "remap", "region"),
                ("remap", "image", "measure", "image"),
                ("remap", "region", "measure", "region"),
                ("remap", "image", "statistics", "image"),
                ("remap", "mask", "statistics", "mask"),
                ("remap", "image", "overlay", "image"),
                ("remap", "result_graphics", "overlay", "graphics_1"),
                ("overlay", "image", "filmstrip", "image"),
            ]
            if case == "packed":
                add("cv.pack_remap_map", "pack", position=(760, 350))
                wires += [
                    ("maps", "map_x", "pack", "map_x"),
                    ("maps", "map_y", "pack", "map_y"),
                    ("pack", "packed_map", "remap", "packed_map"),
                ]
            else:
                other = "maps"
                if case == "wrong_pair":
                    add(
                        "calibration.lens_maps",
                        "other_maps",
                        {"rotation_vector": [0, 0.02, 0]},
                        (760, 700),
                    )
                    wires.append(("profile", "profile", "other_maps", "calibration"))
                    other = "other_maps"
                wires += [("maps", "map_x", "remap", "map_x"), (other, "map_y", "remap", "map_y")]
            for wire in wires:
                graph.connect(*wire)
            note = add("layout.comment", "attribution", position=(0, -500))
            note.name, note.node_size = "Original analytic lens controls - MIT", "large"
            note.description = (
                "Original MIT synthetic RGB16 ramp, Brown5 lens profile and projects by "
                "OpenCVDataflowGUI contributors. No customer or third-party images. Circle "
                "minus rectangle makes a selection with a hole. Image and selection share "
                "one nonlinear remap. truth.json uses analytic distortion and independent "
                "four-tap sampling, without executing the blocks. Areas are target pixels, "
                "not calibrated mm2. Wide demonstrates rejected border support even with "
                "replicated image pixels. Wrong pair intentionally rejects unrelated map "
                "components. See README for expected counts and supported semantics."
            )
            graph.save(HERE / f"{case}.json")
        finally:
            graph.close()
    write_json(HERE / "truth.json", truth)


if __name__ == "__main__":
    build()
