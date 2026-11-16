"""Original, deterministic QR scenes with independent placement/module references."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
CASES = ("cluttered", "rotated", "small", "multiple", "selected", "blank")


def place(scene, text, x, y, *, scale=6, angle=0):
    modules = cv2.QRCodeEncoder_create().encode(text)[2:-2, 2:-2]
    label = np.repeat(np.repeat(np.pad(modules, 4, constant_values=255), scale, 0), scale, 1)
    side = label.shape[0]
    transform = np.vstack((cv2.getRotationMatrix2D(((side - 1) / 2,) * 2, angle, 1), [0, 0, 1]))
    transform[:2, 2] += [x, y]
    output_size = scene.shape[1::-1]
    warped = cv2.warpPerspective(label, transform, output_size, flags=cv2.INTER_NEAREST)
    support = cv2.warpPerspective(
        np.full_like(label, 255), transform, output_size, flags=cv2.INTER_NEAREST
    )
    scene[support != 0] = warped[support != 0]
    low, high = 4 * scale, (4 + len(modules)) * scale - 1
    corners = np.array([[low, low], [high, low], [high, high], [low, high]], np.float64)
    expected = cv2.perspectiveTransform(corners[None], transform)[0]
    return {
        "text": text,
        "corners": expected.tolist(),
        "module_size": len(modules),
        "module_sha256": hashlib.sha256(modules.tobytes()).hexdigest(),
        "pixels_per_module": scale,
        "angle_degrees": angle,
    }


def scenes():
    result = {}
    for case in CASES:
        y, x = np.indices((440, 900))
        image = (210 + (x * 7 + y * 3) % 25).astype(np.uint8)
        for i in range(24):
            point = (int((i * 83 + 9) % 880), int((i * 37 + 7) % 420))
            cv2.rectangle(image, point, (point[0] + 10, point[1] + 8), 80, 1)
        if case == "rotated":
            codes = [place(image, "ROTATED-42", 270, 90, angle=27)]
        elif case == "small":
            codes = [place(image, "SMALL-07", 330, 160, scale=2)]
        elif case in ("multiple", "selected"):
            codes = [
                place(image, "LEFT-42", 40, 100),
                place(image, "CENTER-07", 350, 110, angle=90),
                place(image, "RIGHT-18", 660, 100, angle=-12),
            ]
        elif case == "blank":
            codes = []
        else:
            codes = [place(image, "CLUTTER-42", 270, 100)]
        result[case] = (image, codes)
    return result


def build():
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    registry = builtin_registry()
    truth = {
        "creator": "OpenCVDataflowGUI contributors",
        "license": "MIT",
        "cases": {},
        "asset_sha256": {},
    }
    for case, (image, codes) in scenes().items():
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        (assets / f"{case}.png").write_bytes(encoded.tobytes())
        truth["asset_sha256"][f"{case}.png"] = hashlib.sha256(encoded).hexdigest()
        truth["cases"][case] = {
            "codes": codes,
            "expected_texts": [
                code["text"] for code in codes[: 1 if case == "selected" else len(codes)]
            ],
        }
        truth["cases"][case]["decoded_by_backend"] = {
            backend: []
            if case == "small" and backend == "standard"
            else truth["cases"][case]["expected_texts"]
            for backend in ("standard", "aruco")
        }
        graph = Graph()
        source = graph.add(
            registry.create(
                "input.file",
                block_id="source",
                parameters={
                    "path": f"assets/{case}.png",
                    "path_base": "project",
                    "color_format": "gray",
                    "filename_ticket": True,
                },
            )
        )
        source.position = [0, 0]
        for index, backend in enumerate(("standard", "aruco")):
            reader = graph.add(
                registry.create(
                    "inspect.read_qr",
                    block_id=backend,
                    parameters={
                        "backend": backend,
                        "minimum_reads": len(truth["cases"][case]["expected_texts"]),
                        "qr_min_module_size": 2.0 if case == "small" else 4.0,
                    },
                )
            )
            reader.position = [460, index * 350]
            reader.name = f"{backend.title()} QR reader"
            graph.connect("source", "image", backend, "image")
        if case == "selected":
            region = graph.add(
                registry.create(
                    "region.shape",
                    block_id="selection",
                    parameters={"shape": "rectangle", "rectangle": [20, 70, 230, 240]},
                )
            )
            region.position = [0, 360]
            graph.connect("source", "image", "selection", "image")
            for backend in ("standard", "aruco"):
                graph.connect("selection", "region", backend, "region")
        note = graph.add(registry.create("layout.comment", block_id="attribution"))
        note.position, note.node_size = [0, -450], "large"
        note.name = "Original QR backend comparison"
        note.description = (
            "Original MIT scenes, payloads, placement references and projects by "
            "OpenCVDataflowGUI contributors. Generated by build_example.py using native "
            "OpenCV QRCodeEncoder; no third-party/customer images. Both reader branches "
            "inspect the same filename-ticketed scene. Compare codes and camera-space "
            "corners; rectified module images have their own grid and emit only for one "
            "decoded symbol. Small symbols use two-pixel modules. The selected project "
            "keeps only the left symbol; the blank project expects zero. These controlled "
            "fixtures do not establish production robustness or barcode quality grades."
        )
        graph.save(HERE / f"{case}.json")
        graph.close()
    (HERE / "truth.json").write_bytes((json.dumps(truth, indent=2) + "\n").encode())


if __name__ == "__main__":
    build()
