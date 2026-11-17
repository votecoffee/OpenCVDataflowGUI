"""Original DISK controls plus unchanged, attributed VisA PCB photographs."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
SOURCES = HERE.parents[1] / "Simple Plugins/pattern_matching/visa_pcb2"
HASHES = {
    "depth": "9c2ee4ded238892dfa51569941372601e35e4a74aa6f84ea80053d2ab1c07abe",
    "epipolar": "6e0cf5df4abab1fddbf0b62325647c9a6f387453b28c4b986542b7dabda972ee",
    "matcher": "b5b21d47ea24f2c5e501aec9c91b9716e4c8c3429a4dc1e615c133c4c9378335",
}
CASES = (
    "translation",
    "resized",
    "masked",
    "epipolar",
    "rgb16",
    "empty",
    "minimum_matches",
    "pose_region",
    "comparison",
    "pcb_photos",
)


def build():
    path = HERE / "assets"
    path.mkdir(parents=True, exist_ok=True)
    a = cv2.GaussianBlur(np.random.default_rng(4).integers(0, 255, (160, 200), np.uint8), (3, 3), 0)
    b = cv2.warpAffine(a, np.float32([[1, 0, 8], [0, 1, 6]]), (200, 160))
    images = {"reference": a, "query": b}
    images["large_reference"] = cv2.resize(a, (400, 320), interpolation=cv2.INTER_NEAREST)
    images["large_query"] = cv2.warpAffine(
        images["large_reference"], np.float32([[1, 0, 16], [0, 1, 12]]), (400, 320)
    )
    images["rgb16_reference"] = np.repeat(a[..., None], 3, axis=2).astype(np.uint16) * 257
    images["rgb16_query"] = np.repeat(b[..., None], 3, axis=2).astype(np.uint16) * 257
    files = {}
    for name, pixels in images.items():
        ok, encoded = cv2.imencode(".png", pixels)
        assert ok
        files[name + ".png"] = encoded.tobytes()
    for item in json.loads((SOURCES / "landmarks.json").read_bytes()):
        files[Path(item["path"]).name] = (SOURCES / item["path"]).read_bytes()
    if any((path / name).exists() for name in files):
        raise FileExistsError("Existing DISK example assets are preserved")
    for name, raw in files.items():
        (path / name).write_bytes(raw)
    registry = builtin_registry()
    truth = {}
    for case in CASES:
        graph = Graph()
        try:
            variant = "epipolar" if case == "epipolar" else "depth"
            for side in ("reference", "query"):
                name = (
                    f"large_{side}.png"
                    if case == "resized"
                    else f"rgb16_{side}.png"
                    if case == "rgb16"
                    else f"{side}.png"
                )
                if case == "pcb_photos":
                    name = "pcb2-0000.jpg" if side == "reference" else "pcb2-0621.jpg"
                source = graph.add(
                    registry.create(
                        "input.file",
                        block_id=side,
                        parameters={
                            "path": "assets/" + name,
                            "path_base": "project",
                            "pixel_depth": "unchanged",
                            "color_format": "rgb" if case == "rgb16" else "unchanged",
                        },
                    )
                )
                source.position = [0, 0 if side == "reference" else 430]
                feature = graph.add(
                    registry.create(
                        "ml.disk_features",
                        block_id=side + "_features",
                        parameters={
                            "weights_path": f"../../../.local/models/disk_{variant}.pth",
                            "weights_sha256": HASHES[variant],
                            "max_features": 512 if case == "pcb_photos" else 128,
                            "resize_long_edge": 200 if case == "resized" else 0,
                            "minimum_count": 0 if case == "empty" else 4,
                            "detection_threshold": 1e6 if case == "empty" else 0.0,
                        },
                    )
                )
                feature.position = [380, source.position[1]]
                feature.exposed_outputs.append("result_graphics")
                graph.connect(side, "image", feature.id, "image")
                if case == "masked":
                    region = graph.add(
                        registry.create(
                            "region.shape",
                            block_id=side + "_region",
                            parameters={
                                "shape": "rectangle",
                                "rectangle": [8 if side == "query" else 0, 0, 110, 160],
                            },
                        )
                    )
                    region.position = [0, source.position[1] + 220]
                    graph.connect(side, "image", region.id, "image")
                    graph.connect(region.id, "region", feature.id, "region")
            match = graph.add(
                registry.create(
                    "ml.lightglue_disk",
                    block_id="match",
                    parameters={
                        "weights_path": "../../../.local/models/disk_lightglue.pth",
                        "weights_sha256": HASHES["matcher"],
                        "minimum_matches": 4096 if case == "minimum_matches" else 8,
                        "maximum_features": 1024,
                    },
                )
            )
            match.position = [800, 430]
            match.exposed_outputs.append("result_graphics")
            graph.connect("query", "image", "match", "image")
            for side in ("reference", "query"):
                graph.connect(side + "_features", "features", "match", side + "_features")
            overlay = graph.add(registry.create("cv.overlay_result_graphics", block_id="overlay"))
            film = graph.add(registry.create("output.filmstrip_image", block_id="filmstrip"))
            overlay.position, film.position = [1200, 430], [1600, 430]
            graph.connect("match", "image", "overlay", "image")
            graph.connect("match", "result_graphics", "overlay", "graphics_1")
            graph.connect("overlay", "image", "filmstrip", "image")
            if case == "pose_region":
                roi = graph.add(
                    registry.create(
                        "region.pose",
                        block_id="inspection_region",
                        parameters={"shape": "rectangle", "rectangle": [35, 35, 60, 50]},
                    )
                )
                roi.position = [800, 760]
                roi.exposed_outputs.append("result_graphics")
                graph.connect("query", "image", roi.id, "image")
                graph.connect("match", "pose", roi.id, "pose")
                graph.connect(roi.id, "result_graphics", "overlay", "graphics_2")
            if case == "comparison":
                sift = graph.add(
                    registry.create(
                        "ml.lightglue",
                        block_id="sift",
                        parameters={
                            "weights_path": "../../../.local/models/sift_lightglue.pth",
                            "max_features": 64,
                        },
                    )
                )
                sift.position = [800, 0]
                graph.connect("query", "image", "sift", "image")
                graph.connect("reference", "image", "sift", "reference")
            note = graph.add(registry.create("layout.comment", block_id="attribution"))
            note.name, note.node_size, note.position = (
                "DISK / LightGlue and image attribution",
                "large",
                [0, -500],
            )
            note.description = (
                "DISK: Tyszkiewicz, Fua and Trulls (NeurIPS 2020); Kornia and LightGlue "
                "(Lindenberger, Sarlin, Pollefeys, ICCV 2023), Apache-2.0; weights prepared "
                "separately. Original MIT synthetic images/graphs by OpenCVDataflowGUI "
                "contributors. PCB photos only: VisA, Zou, Jeong, Pemula, Zhang and Dabeer "
                "(ECCV 2022), CC BY 4.0, "
                "resized/JPEG prepared views copied unchanged. Real detector inference, not fixed "
                "feature rows. Empty/minimum_matches intentionally fail. See README and SOURCES."
            )
            graph.save(HERE / (case + ".json"))
            truth[case] = {
                "passed": case not in ("empty", "minimum_matches"),
                "translation": [16, 12] if case == "resized" else [8, 6],
                "pose_origin_tolerance": 1.0 if case == "masked" else 0.5,
                "reference_landmarks": [[99, 172], [556, 178], [96, 360], [553, 365]]
                if case == "pcb_photos"
                else [],
                "query_landmarks": [[89, 172], [547, 184], [85, 360], [542, 372]]
                if case == "pcb_photos"
                else [],
                "landmark_tolerance": 4.0,
            }
        finally:
            graph.close()
    (HERE / "truth.json").write_bytes(
        (
            json.dumps(
                {
                    "cases": truth,
                    "weights_sha256": HASHES,
                    "asset_sha256": {
                        name: hashlib.sha256(raw).hexdigest() for name, raw in sorted(files.items())
                    },
                },
                indent=2,
            )
            + "\n"
        ).encode()
    )


if __name__ == "__main__":
    build()
