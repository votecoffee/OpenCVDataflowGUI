"""Build portable cascade examples from pinned OpenCV XML and a NASA portrait."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
CASES = (
    "portrait",
    "neighbors",
    "boxes",
    "lbp",
    "two_faces",
    "selection",
    "high_depth",
    "equalized",
    "weight_fail",
    "count_fail",
    "blank",
    "bad_pin",
    "single_scale",
    "strict_size",
    "raw_windows",
)
REFERENCE_FACE = [175, 60, 105, 115]


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def build(destination, sources):
    destination, sources = Path(destination), Path(sources)
    assets = destination / "assets"
    if assets.exists():
        raise FileExistsError("Existing cascade assets preserved")
    portrait = cv2.imread(str(sources / "portrait.png"))
    if portrait is None or portrait.shape != (512, 512, 3):
        raise ValueError("Supply the unchanged 512x512 NASA portrait")
    two = np.full((512, 768, 3), 25, np.uint8)
    two[:, :512] = portrait
    two[100:356, 512:768] = cv2.resize(portrait, (256, 256), interpolation=cv2.INTER_AREA)
    images = {
        "portrait.png": portrait,
        "two-faces.png": two,
        "portrait16.png": portrait.astype(np.uint16) * 257,
        "blank.png": np.zeros_like(portrait),
    }
    files = {name: cv2.imencode(".png", image)[1].tobytes() for name, image in images.items()}
    # Preserve original portrait encoding and XML/license bytes exactly.
    files["portrait.png"] = (sources / "portrait.png").read_bytes()
    for name in ("haar.xml", "lbp.xml"):
        files[name] = (sources / name).read_bytes()
    assets.mkdir(parents=True)
    for name, raw in files.items():
        (assets / name).write_bytes(raw)
    registry, cases = builtin_registry(), {}
    for case in CASES:
        graph = Graph()

        def add(kind, name, x, y, **parameters):
            block = graph.add(registry.create(kind, block_id=name, parameters=parameters))
            block.position = [x, y]
            return block

        try:
            name = (
                "two-faces.png"
                if case in ("two_faces", "selection")
                else "portrait16.png"
                if case == "high_depth"
                else "blank.png"
                if case == "blank"
                else "portrait.png"
            )
            add(
                "input.file",
                "source",
                0,
                0,
                source_mode="files",
                paths=["assets/" + name],
                path_base="project",
                color_format="unchanged",
                pixel_depth="unchanged",
            )
            model = "lbp.xml" if case == "lbp" else "haar.xml"
            settings = {
                "path": "assets/" + model,
                "model_sha256": hashlib.sha256(files[model]).hexdigest(),
                "label": "face",
                "min_size": [40, 40] if case in ("two_faces", "selection") else [70, 70],
                "max_size": [150, 150],
                "min_neighbors": 0 if case == "raw_windows" else 3,
                "result_mode": case if case in ("neighbors", "boxes") else "weights",
                "equalize_histogram": case == "equalized",
                "minimum_weight_enabled": case == "weight_fail",
                "minimum_weight": 20.0,
                "minimum_count": 2 if case == "count_fail" else 1,
            }
            if case == "bad_pin":
                settings["model_sha256"] = "0" * 64
            if case in ("single_scale", "strict_size"):
                settings.update(
                    min_size=[97, 97], max_size=[97, 97], strict_size=case == "strict_size"
                )
            detector = add("ml.cascade_detect", "detect", 450, 0, **settings)
            detector.exposed_outputs.append("result_graphics")
            graph.connect("source", "image", "detect", "image")
            if case == "selection":
                add("region.shape", "selection", 0, -420, rectangle=[0, 0, 512, 512])
                graph.connect("source", "image", "selection", "image")
                graph.connect("selection", "region", "detect", "region")
            add("cv.overlay_result_graphics", "overlay", 850, 0)
            add("output.filmstrip_image", "filmstrip", 1250, 0)
            graph.connect("detect", "image", "overlay", "image")
            graph.connect("detect", "result_graphics", "overlay", "graphics_1")
            graph.connect("overlay", "image", "filmstrip", "image")
            note = add("layout.comment", "attribution", 0, -850)
            note.name, note.node_size = "NASA portrait and OpenCV cascade attribution", "large"
            note.description = (
                "NASA photograph of Eileen Collins, public domain via scikit-image. "
                "OpenCV 4.13.0 HAAR model by Rainer Lienhart retains its Intel license in XML; "
                "LBP model retains OpenCV Apache-2 terms in LICENSE.opencv. Original graphs and "
                "constructed two-view composition are MIT. Approximate manually marked face ROI "
                "is a smoke-test annotation, not a localization benchmark. Native stage weights "
                "and grouped neighbor counts are distinct and neither is probability. See "
                "README.md, SOURCES.md, truth.json and benchmark.json."
            )
            graph.save(destination / (case + ".json"))
            cases[case] = {
                "state": "error"
                if case == "bad_pin"
                else "fail"
                if case in ("weight_fail", "count_fail", "blank", "strict_size")
                else "pass",
                "image": name,
            }
        finally:
            graph.close()
    (destination / "truth.json").write_bytes(
        encoded(
            {
                "cases": cases,
                "reference_face": REFERENCE_FACE,
                "reference_scope": "Coarse manually marked portrait face ROI; constructed second "
                "ROI uses known scale/translation. Not an accuracy dataset.",
                "second_reference_face": [
                    512 + REFERENCE_FACE[0] / 2,
                    100 + REFERENCE_FACE[1] / 2,
                    REFERENCE_FACE[2] / 2,
                    REFERENCE_FACE[3] / 2,
                ],
                "asset_sha256": {
                    name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()
                },
            }
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path)
    parser.add_argument("--destination", type=Path, default=HERE)
    args = parser.parse_args()
    build(args.destination, args.sources)
