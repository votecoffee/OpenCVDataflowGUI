"""Original MIT industrial-style text labels and independently transformed word truth."""

import hashlib
import json
import runpy
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
PARAMETERS = runpy.run_path(str(Path(__file__).with_name("prepare_model.py")))["PARAMETERS"]
CASES = {
    "labels": {},
    "rotated": {"image": "rotated"},
    "selected_roi": {"roi": True},
    "empty": {"image": "empty", "passed": False},
    "count_failure": {"minimum_count": 4, "passed": False},
    "truncated": {"maximum_regions": 1, "passed": False},
    "joined_words": {"image": "joined"},
    "split_words": {"image": "split"},
}


def asset(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != raw:
        raise FileExistsError("Different existing asset preserved; generate in a fresh directory")
    path.write_bytes(raw)


def image_asset(name, image):
    ok, raw = cv2.imencode(".png", image)
    assert ok
    asset(HERE / "assets" / (name + ".png"), raw.tobytes())


def text_image(lines):
    image = np.full((320, 640, 3), 255, np.uint8)
    truth = []
    for value, x, y in lines:
        glyph = np.zeros(image.shape[:2], np.uint8)
        cv2.putText(glyph, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.7, 255, 3, cv2.LINE_AA)
        image[glyph > 0] = 255 - glyph[glyph > 0, None]
        xs, ys, w, h = cv2.boundingRect(cv2.findNonZero(glyph))
        truth.append(
            {
                "text": value,
                "corners": [[xs, ys], [xs + w - 1, ys], [xs + w - 1, ys + h - 1], [xs, ys + h - 1]],
            }
        )
    return image, truth


def build():
    labels, truth = text_image([("LOT2048", 60, 90), ("PNAX12", 60, 175), ("PASS", 60, 260)])
    rotation = cv2.getRotationMatrix2D((320, 160), 12, 1)
    rotated = cv2.warpAffine(labels, rotation, (640, 320), borderValue=(255, 255, 255))
    rotated_truth = [
        {
            **row,
            "corners": cv2.transform(np.asarray(row["corners"], np.float64)[None], rotation)[
                0
            ].tolist(),
        }
        for row in truth
    ]
    joined, joined_truth = text_image([("ABCD", 60, 175)])
    split, split_truth = text_image([("AB", 60, 175), ("CD", 330, 175)])
    references = {
        "labels": truth,
        "rotated": rotated_truth,
        "joined": joined_truth,
        "split": split_truth,
        "empty": [],
    }
    for name, image in (
        ("labels", labels),
        ("rotated", rotated),
        ("joined", joined),
        ("split", split),
        ("empty", np.full_like(labels, 255)),
    ):
        image_asset(name, image)
    registry = builtin_registry()
    for name, options in CASES.items():
        graph = Graph()
        try:
            graph.program["assets"] = ["MODEL-LICENSE.txt", "SOURCES.md"]

            def add(kind, identity, pos, **parameters):
                block = graph.add(registry.create(kind, block_id=identity, parameters=parameters))
                block.position = list(pos)
                return block

            source_name = options.get("image", "labels")
            add(
                "input.file",
                "source",
                (0, 0),
                path=f"assets/{source_name}.png",
                path_base="project",
            )
            add(
                "ml.dnn",
                "infer",
                (420, 0),
                **PARAMETERS,
                contract_path=".local/ppocr_en/contract.json",
            )
            decoder = add(
                "ml.text_regions",
                "decode",
                (850, 0),
                output_name="4",
                expansion="db_unclip",
                unclip_ratio=1.5,
                minimum_side=3,
                minimum_count=options.get("minimum_count", 1),
                maximum_regions=options.get("maximum_regions", 64),
            )
            decoder.node_size = "large"
            decoder.exposed_outputs.append("result_graphics")
            add("cv.overlay_result_graphics", "overlay", (1280, 0))
            add("output.filmstrip_image", "filmstrip", (1700, 0))
            for a, o, b, i in (
                ("source", "image", "infer", "image"),
                ("source", "image", "decode", "image"),
                ("infer", "tensors", "decode", "tensors"),
                ("infer", "preprocessing", "decode", "preprocessing"),
                ("decode", "image", "overlay", "image"),
                ("decode", "result_graphics", "overlay", "graphics_1"),
                ("overlay", "image", "filmstrip", "image"),
            ):
                graph.connect(a, o, b, i)
            if options.get("roi"):
                add("region.shape", "roi", (420, 450), rectangle=[45, 35, 260, 75])
                graph.connect("source", "image", "roi", "image")
                graph.connect("roi", "region", "decode", "region")
            note = add("layout.comment", "attribution", (0, -500))
            note.name, note.node_size = "PP-OCRv3 / original label attribution", "large"
            note.description = (
                "PP-OCRv3 English model: PaddlePaddle Authors / OpenCV Zoo, Apache 2.0, "
                "revision 48b10fa505aeb51fbd127fc60a5c660040e823b7; weights unchanged. "
                "Original MIT synthetic industrial-style labels, geometry truth and pipeline "
                "by OpenCVDataflowGUI contributors. No customer images. These are text regions, "
                "not OCR reads. Empty/count/truncation controls intentionally fail. "
                "Explicitly run prepare_model.py first. See SOURCES.md and README.md."
            )
            graph.save(HERE / (name + ".json"))
        finally:
            graph.close()
    asset(
        HERE / "truth.json",
        (
            json.dumps(
                {
                    "cases": CASES,
                    "annotations": references,
                    "annotation_method": (
                        "Known rendered glyph bounds; rotated by the independent affine matrix"
                    ),
                    "rotation_source_to_target": rotation.tolist(),
                    "asset_sha256": {
                        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted((HERE / "assets").iterdir())
                    },
                },
                indent=2,
            )
            + "\n"
        ).encode(),
    )


if __name__ == "__main__":
    build()
