"""ALIKED image-pair controls using openly attributed, byte-pinned existing images."""

import hashlib
import json
from pathlib import Path

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "disk_matching"
HASHES = {
    "n16": "5be8704840ed662d9d8c561bf7279c222092674e7eb05fd0feab94899e9d82f2",
    "n16rot": "ddf3abbf38e86f6a74540d214e1a9712c54b2d8551abc864542199f2347d7332",
    "n32": "055283fd1624d3ab46a05c3b617b1a79a3f28d1f551737eef2d653e8c8232890",
    "matcher": "d975e965b105311a6143194852297dff4f02aea5cc2e10cecfed966ca0e22503",
}
CASES = (
    "translation",
    "resized",
    "masked",
    "n16rot",
    "n32",
    "rgb16",
    "empty",
    "minimum_matches",
    "pose_region",
    "comparison",
    "pcb_photos",
    "threshold_fallback",
    "strict_score",
)


def build():
    path = HERE / "assets"
    source_truth = json.loads((SOURCE / "truth.json").read_bytes())
    files = {}
    for name, digest in source_truth["asset_sha256"].items():
        raw = (SOURCE / "assets" / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("Source image differs from its pinned public fixture: " + name)
        files[name] = raw
    path.mkdir(parents=True, exist_ok=True)
    if any((path / name).exists() for name in files):
        raise FileExistsError("Existing ALIKED example images preserved")
    for name, raw in files.items():
        (path / name).write_bytes(raw)
    registry = builtin_registry()
    truth = {}
    for case in CASES:
        graph = Graph()
        try:
            variant = case if case in ("n16rot", "n32") else "n16"
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
                        "ml.aliked_features",
                        block_id=side + "_features",
                        parameters={
                            "model_name": "aliked-" + variant,
                            "weights_path": f"../../../.local/models/aliked_{variant}.pth",
                            "weights_sha256": HASHES[variant],
                            "max_features": 512 if case == "pcb_photos" else 128,
                            "resize_long_edge": 200 if case == "resized" else 0,
                            "minimum_count": 0 if case in ("empty", "strict_score") else 4,
                            "detection_mode": "threshold"
                            if case in ("threshold_fallback", "strict_score")
                            else "top_k",
                            "detection_threshold": 1.0
                            if case in ("threshold_fallback", "strict_score")
                            else 0.2,
                            "minimum_score": 1.0 if case == "strict_score" else 0.0,
                        },
                    )
                )
                feature.position = [380, source.position[1]]
                feature.exposed_outputs.append("result_graphics")
                graph.connect(side, "image", feature.id, "image")
                if case in ("masked", "empty"):
                    roi = graph.add(
                        registry.create(
                            "region.shape",
                            block_id=side + "_region",
                            parameters={
                                "shape": "rectangle",
                                "rectangle": [-10, -10, 1, 1]
                                if case == "empty"
                                else [8 if side == "query" else 0, 0, 110, 160],
                            },
                        )
                    )
                    roi.position = [0, source.position[1] + 220]
                    graph.connect(side, "image", roi.id, "image")
                    graph.connect(roi.id, "region", feature.id, "region")
            match = graph.add(
                registry.create(
                    "ml.lightglue_aliked",
                    block_id="match",
                    parameters={
                        "weights_path": "../../../.local/models/aliked_lightglue.pth",
                        "weights_sha256": HASHES["matcher"],
                        "minimum_matches": 4096 if case == "minimum_matches" else 8,
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
                graph.connect("reference", "image", "sift", "reference")
                graph.connect("query", "image", "sift", "image")
            note = graph.add(registry.create("layout.comment", block_id="attribution"))
            note.name, note.node_size, note.position = (
                "ALIKED / LightGlue and image attribution",
                "large",
                [0, -500],
            )
            note.description = (
                "ALIKED: Xiaoming Zhao, Xingming Wu, Weihai Chen, Peter C.Y. Chen, Qingsong Xu "
                "and Zhengguo Li (IEEE TIM 2023), BSD-3-Clause; Kornia and LightGlue "
                "(Lindenberger, Sarlin, Pollefeys, ICCV 2023), Apache-2.0. Models prepared "
                "separately. Original MIT texture controls by OpenCVDataflowGUI contributors. "
                "PCB photos: VisA, Zou, Jeong, Pemula, Zhang, Dabeer (ECCV 2022), CC BY 4.0. "
                "Prepared public images copied unchanged. Native threshold mode can fall back to "
                "mean score; Minimum score is independent. Empty/minimum_matches/strict_score "
                "intentionally fail. See README and SOURCES for source links and limits."
            )
            graph.save(HERE / (case + ".json"))
            photo = source_truth["cases"]["pcb_photos"]
            truth[case] = {
                "passed": case not in ("empty", "minimum_matches", "strict_score"),
                "translation": [16, 12] if case == "resized" else [8, 6],
                "pose_origin_tolerance": 1.0 if case == "masked" else 0.5,
                "reference_landmarks": photo["reference_landmarks"] if case == "pcb_photos" else [],
                "query_landmarks": photo["query_landmarks"] if case == "pcb_photos" else [],
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
                    "asset_sha256": source_truth["asset_sha256"],
                },
                indent=2,
            )
            + "\n"
        ).encode()
    )


if __name__ == "__main__":
    build()
