"""Original pinhole/cylinder QR fixtures with analytic surface and module references."""

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
TEXT, SIZE, SCALE, DISTANCE = "CURVED-0042", 600, 10, 700
CASES = {
    "mild": {
        "radius": 500,
        "pitch": 0,
        "version": 2,
        "angle": 0,
        "passed": True,
        "flat_reads": True,
        "maximum_module_errors": 0,
    },
    "wrapped": {
        "radius": 300,
        "pitch": 35,
        "version": 2,
        "angle": 0,
        "passed": True,
        "flat_reads": False,
        "maximum_module_errors": 0,
    },
    "vertical": {
        "radius": 300,
        "pitch": 35,
        "version": 2,
        "angle": 90,
        "passed": True,
        "flat_reads": False,
        "maximum_module_errors": 1,
    },
    "larger_symbol": {
        "radius": 500,
        "pitch": 20,
        "version": 3,
        "angle": 0,
        "passed": True,
        "flat_reads": False,
        "maximum_module_errors": 0,
    },
    "selected": {
        "radius": 300,
        "pitch": 35,
        "version": 2,
        "angle": 0,
        "passed": True,
        "flat_reads": False,
        "maximum_module_errors": 0,
    },
    "severe": {
        "radius": 200,
        "pitch": 35,
        "version": 2,
        "angle": 0,
        "passed": False,
        "flat_reads": False,
        "maximum_module_errors": None,
    },
    "rotated_failure": {
        "radius": 300,
        "pitch": 35,
        "version": 2,
        "angle": 27,
        "passed": False,
        "flat_reads": False,
        "maximum_module_errors": None,
    },
}


def project(uv, radius, pitch):
    """Independent forward 3D pinhole projection of label-plane arc/height positions."""
    beta = math.radians(pitch)
    theta = uv[:, 0] / radius
    x = radius * np.sin(theta)
    depth = radius * (1 - np.cos(theta))
    y = uv[:, 1] * math.cos(beta) - depth * math.sin(beta)
    z = DISTANCE + uv[:, 1] * math.sin(beta) + depth * math.cos(beta)
    return (SIZE - 1) / 2 + DISTANCE * np.column_stack((x / z, y / z))


def scene(options):
    params = cv2.QRCodeEncoder_Params()
    params.version = options["version"]
    params.correction_level = cv2.QRCodeEncoder_CORRECT_LEVEL_H
    modules = cv2.QRCodeEncoder_create(params).encode(TEXT)[2:-2, 2:-2]
    label = np.repeat(np.repeat(np.pad(modules, 4, constant_values=255), SCALE, 0), SCALE, 1)
    radius, pitch = options["radius"], options["pitch"]
    beta, center_pixel = math.radians(pitch), (SIZE - 1) / 2
    yy, xx = np.indices((SIZE, SIZE), dtype=float)
    rays = np.stack(
        ((xx - center_pixel) / DISTANCE, (yy - center_pixel) / DISTANCE, np.ones_like(xx)), axis=-1
    )
    axis = np.array([0, math.cos(beta), math.sin(beta)])
    center = np.array([0, -radius * math.sin(beta), DISTANCE + radius * math.cos(beta)])
    perpendicular_ray = rays - (rays @ axis)[..., None] * axis
    perpendicular_center = center - (center @ axis) * axis
    a = (perpendicular_ray**2).sum(axis=-1)
    b = -2 * (perpendicular_ray @ perpendicular_center)
    c = perpendicular_center @ perpendicular_center - radius**2
    discriminant = b * b - 4 * a * c
    t = (-b - np.sqrt(np.maximum(discriminant, 0))) / (2 * a)
    points = rays * t[..., None]
    x, y, z = points[..., 0], points[..., 1], points[..., 2] - DISTANCE
    v = y * math.cos(beta) + z * math.sin(beta)
    depth = -y * math.sin(beta) + z * math.cos(beta)
    u = radius * np.arctan2(x, radius - depth)
    half = (label.shape[0] - 1) / 2
    image = cv2.remap(
        label,
        (u + half).astype(np.float32),
        (v + half).astype(np.float32),
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    image[discriminant < 0] = 255
    low, high = 4 * SCALE, (4 + len(modules)) * SCALE - 1
    uv = np.array([[low, low], [high, low], [high, high], [low, high]], float) - half
    corners = project(uv, radius, pitch)
    if options["angle"]:
        transform = cv2.getRotationMatrix2D((center_pixel, center_pixel), options["angle"], 1)
        image = cv2.warpAffine(
            image, transform, (SIZE, SIZE), flags=cv2.INTER_NEAREST, borderValue=255
        )
        corners = np.column_stack((corners, np.ones(4))) @ transform.T
    return image, modules, corners


def build():
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    truth = {
        "license": "MIT",
        "creator": "OpenCVDataflowGUI contributors",
        "text": TEXT,
        "image_size": [SIZE, SIZE],
        "distance": DISTANCE,
        "focal_pixels": DISTANCE,
        "pixels_per_module": SCALE,
        "cases": {},
        "asset_sha256": {},
    }
    registry = builtin_registry()
    for name, options in CASES.items():
        image, modules, corners = scene(options)
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        destination = assets / (name + ".png")
        if destination.exists() and destination.read_bytes() != encoded.tobytes():
            raise ValueError("Existing fixture differs; generate in a fresh folder")
        destination.write_bytes(encoded.tobytes())
        truth["asset_sha256"][destination.name] = hashlib.sha256(encoded).hexdigest()
        truth["cases"][name] = {
            **options,
            "surface_corners": corners.tolist(),
            "module_size": len(modules),
            "modules": modules.tolist(),
            "module_sha256": hashlib.sha256(modules.tobytes()).hexdigest(),
            "anchor_corner_indices": [0, 1, 3],
            "maximum_anchor_error_pixels": 3,
        }
        graph = Graph()
        try:
            source = graph.add(
                registry.create(
                    "input.file",
                    block_id="source",
                    parameters={
                        "path": f"assets/{name}.png",
                        "path_base": "project",
                        "color_format": "gray",
                    },
                )
            )
            reader = graph.add(
                registry.create(
                    "inspect.read_qr",
                    block_id="read",
                    parameters={"backend": "standard", "mode": "curved"},
                )
            )
            reader.exposed_outputs.append("info")
            reader.position, reader.node_size = [440, 0], "large"
            filmstrip = graph.add(registry.create("output.filmstrip_image", block_id="filmstrip"))
            filmstrip.position = [880, 0]
            graph.connect(source.id, "image", reader.id, "image")
            graph.connect(reader.id, "overlay", filmstrip.id, "image")
            if name == "selected":
                region = graph.add(
                    registry.create(
                        "region.shape",
                        block_id="selection",
                        parameters={"shape": "rectangle", "rectangle": [90, 90, 420, 420]},
                    )
                )
                region.position = [0, 350]
                graph.connect(source.id, "image", region.id, "image")
                graph.connect(region.id, "region", reader.id, "region")
            note = graph.add(registry.create("layout.comment", block_id="attribution"))
            note.name, note.position, note.node_size = (
                "Original curved QR fixtures - MIT",
                [0, -490],
                "large",
            )
            note.description = (
                "Original MIT images, pinhole/cylinder renderer, analytic references and "
                "projects by OpenCVDataflowGUI contributors. No customer/third-party images. "
                "Native OpenCV QR encoding supplies known module bytes. Curved decoding "
                "reads five fixtures; severe curvature and the 27-degree case intentionally "
                "fail. The flat decoder reads only mild. Native corners are an approximate "
                "quadrilateral: the fourth can be far from the true curved surface corner. "
                "Axes are image-plane orientation, not robot pose. Rectified modules have "
                "their own grid with no native inverse surface map; sampled pixels can "
                "contain errors even when the text decodes correctly. See truth.json."
            )
            graph.save(HERE / (name + ".json"))
        finally:
            graph.close()
    (HERE / "truth.json").write_bytes((json.dumps(truth, indent=2) + "\n").encode())


if __name__ == "__main__":
    build()
