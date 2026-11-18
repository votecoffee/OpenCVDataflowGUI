"""Original disjoint controlled views for portable SIFT/ORB image retrieval."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import FlowGroup, ValuePacket

HERE = Path(__file__).resolve().parent
LABELS = ("washers", "crosses", "tabs")
CASES = (
    "train_sift",
    "train_orb",
    "gallery_sift",
    "gallery_orb",
    "retrieve_sift",
    "retrieve_orb",
    "hellinger",
    "strict_rejection",
    "blank_query",
    "wrong_features",
    "wrong_histogram",
    "spatial_rearrangement",
)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def part(kind, angle, seed, gain=1):
    base = np.full((256, 256), 35, np.uint8)
    for y in (40, 98, 156, 214):
        for x in (40, 98, 156, 214):
            if kind == "washers":
                cv2.circle(base, (x, y), 19, 210, -1)
                cv2.circle(base, (x, y), 10, 50, -1)
                cv2.circle(base, (x + 11, y - 9), 3, 90, -1)
            elif kind == "crosses":
                cv2.line(base, (x - 18, y - 18), (x + 18, y + 18), 215, 7)
                cv2.line(base, (x - 18, y + 18), (x + 18, y - 18), 215, 7)
                cv2.circle(base, (x, y), 4, 35, -1)
            else:
                cv2.rectangle(base, (x - 17, y - 17), (x + 17, y + 17), 185, -1)
                cv2.rectangle(base, (x - 10, y - 10), (x + 17, y + 9), 35, -1)
                cv2.circle(base, (x - 11, y - 11), 3, 255, -1)
    matrix = cv2.getRotationMatrix2D((128, 128), angle, 1)
    base = cv2.warpAffine(base, matrix, (256, 256), borderValue=35)
    return np.uint8(
        np.clip(
            base.astype(float) * gain + np.random.default_rng(seed).integers(-2, 3, base.shape),
            0,
            255,
        )
    )


def build():
    assets = HERE / "assets"
    if assets.exists():
        raise FileExistsError("Existing visual-word assets preserved")
    registry = builtin_registry()

    def run(kind, inputs, **parameters):
        block = registry.create(kind, parameters=parameters)
        try:
            result = block.execute(inputs)
            if result is None:
                raise RuntimeError(block.error)
            return result
        finally:
            block.close()

    images, training, queries = {}, [], []
    for index, label in enumerate(LABELS):
        for view, angle in enumerate((-3, 3)):
            name = f"{label}-train-{view}.png"
            images[name] = part(label, angle, 10 + index * 2 + view)
            training.append(name)
        name = f"{label}-query.png"
        images[name] = part(label, 7, 50 + index, 0.95)
        queries.append(name)
    images["blank.png"] = np.full((256, 256), 35, np.uint8)
    original = images[queries[0]]
    images["rearranged.png"] = np.block(
        [[original[128:, 128:], original[:128, :128]], [original[128:, :128], original[:128, 128:]]]
    )
    files, results = {}, {}
    for name, image in images.items():
        ok, raw = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("Cannot encode " + name)
        files[name] = raw.tobytes()
    for method in ("sift", "orb"):
        records = [
            run(
                "ml.visual_features", {"image": images[name]}, method=method, max_features=200
            ).outputs["features"]
            for name in training
        ]
        group = FlowGroup(tuple(ValuePacket(value, {}) for value in records), tuple(training))
        model = run("ml.bow_train", {"features": group}).outputs["model"]
        files[f"codebook-{method}.json"] = encoded(model)
        for normalization in ("l1", "hellinger") if method == "sift" else ("l1",):
            signatures = []
            for i in (0, 2, 4):
                result = run(
                    "ml.bow_encode",
                    {"model": model, "features": records[i]},
                    normalization=normalization,
                )
                signatures.append(
                    ValuePacket(result.outputs["histogram"], result.payloads["histogram"])
                )
            gallery = run(
                "ml.bow_gallery", {"histograms": FlowGroup(tuple(signatures), LABELS)}
            ).outputs["model"]
            files[f"gallery-{method}-{normalization}.json"] = encoded(gallery)
            measures = []
            for index, name in enumerate(queries):
                features = run(
                    "ml.visual_features", {"image": images[name]}, method=method, max_features=200
                ).outputs["features"]
                result = run(
                    "ml.bow_encode",
                    {"model": model, "features": features},
                    normalization=normalization,
                )
                ranked = run(
                    "ml.bow_rank",
                    {
                        "gallery": gallery,
                        "query": ValuePacket(
                            result.outputs["histogram"], result.payloads["histogram"]
                        ),
                    },
                )
                if ranked.outputs["label"] != LABELS[index]:
                    raise AssertionError((method, name, ranked.outputs["results"]))
                measures.append(
                    {
                        "image": name,
                        "label": ranked.outputs["label"],
                        "distance": ranked.outputs["distance"],
                    }
                )
            results[f"{method}-{normalization}"] = measures
    assets.mkdir()
    for name, raw in files.items():
        (assets / name).write_bytes(raw)
    cases = {}
    for case in CASES:
        method = "orb" if case.endswith("orb") else "sift"
        normalization = (
            "hellinger" if case == "hellinger" else "l2" if case == "wrong_histogram" else "l1"
        )
        fitting, building = case.startswith("train_"), case.startswith("gallery_")
        names = (
            training
            if fitting
            else [training[i] for i in (0, 2, 4)]
            if building
            else ["blank.png"]
            if case == "blank_query"
            else ["rearranged.png"]
            if case == "spatial_rearrangement"
            else queries[:1]
            if case in ("strict_rejection", "wrong_features", "wrong_histogram")
            else queries
        )
        graph = Graph()

        def add(kind, block_id, x, y, **parameters):
            block = graph.add(registry.create(kind, block_id=block_id, parameters=parameters))
            block.position = [x, y]
            return block

        try:
            add(
                "input.file",
                "source",
                0,
                0,
                source_mode="files",
                path_base="project",
                paths=["assets/" + n for n in names],
                color_format="gray",
                pixel_depth="uint8",
            )
            add(
                "ml.visual_features",
                "features",
                360,
                0,
                method="orb" if case == "wrong_features" else method,
                max_features=200,
            )
            graph.connect("source", "image", "features", "image")
            if fitting:
                add("flow.batch", "batch", 720, 0, size=6)
                add("ml.bow_train", "fit", 1080, 0)
                add("ml.bow_save", "save", 1440, 0, path=f"trained-codebook-{method}.json")
                graph.connect("features", "features", "batch", "item_1")
                graph.connect("batch", "group", "fit", "features")
                graph.connect("fit", "model", "save", "model")
                target, states = "save", ["inactive"] * 5 + ["pass"]
            else:
                add("ml.bow_load", "codebook", 360, 420, path=f"assets/codebook-{method}.json")
                add("ml.bow_encode", "encode", 720, 0, normalization=normalization)
                graph.connect("features", "features", "encode", "features")
                graph.connect("source", "image", "encode", "image")
                graph.connect("codebook", "model", "encode", "model")
                if building:
                    add("flow.batch", "batch", 1080, 0, size=3)
                    add("ml.bow_gallery", "build", 1440, 0, labels=list(LABELS))
                    add("ml.bow_save", "save", 1800, 0, path=f"trained-gallery-{method}.json")
                    graph.connect("encode", "histogram", "batch", "item_1")
                    graph.connect("batch", "group", "build", "histograms")
                    graph.connect("build", "model", "save", "model")
                    target, states = "save", ["inactive", "inactive", "pass"]
                else:
                    gallery_mode = "hellinger" if case == "hellinger" else "l1"
                    add(
                        "ml.bow_load",
                        "gallery",
                        720,
                        420,
                        path=f"assets/gallery-{method}-{gallery_mode}.json",
                    )
                    rank = add(
                        "ml.bow_rank",
                        "rank",
                        1080,
                        0,
                        max_distance=0.001 if case == "strict_rejection" else -1,
                    )
                    rank.exposed_outputs.append("result_graphics")
                    graph.connect("encode", "histogram", "rank", "query")
                    graph.connect("source", "image", "rank", "image")
                    graph.connect("gallery", "model", "rank", "gallery")
                    add("cv.overlay_result_graphics", "overlay", 1440, 0)
                    add("output.filmstrip_image", "filmstrip", 1800, 0)
                    graph.connect("rank", "image", "overlay", "image")
                    graph.connect("rank", "result_graphics", "overlay", "graphics_1")
                    graph.connect("overlay", "image", "filmstrip", "image")
                    target = "encode" if case == "wrong_features" else "rank"
                    states = (
                        ["error"]
                        if case in ("wrong_features", "wrong_histogram")
                        else ["fail"]
                        if case in ("strict_rejection", "blank_query")
                        else ["pass"] * len(names)
                    )
            note = add("layout.comment", "attribution", 0, -460)
            note.name, note.node_size = "Original visual-word retrieval controls", "large"
            note.description = (
                "Original MIT synthetic images, models and graphs by "
                "OpenCVDataflowGUI contributors. Six training views use +/-3 degree rotation "
                "and separate noise. Held-out views use 7 degrees, different noise and a 0.95 "
                "gain. SIFT uses native L2 k-means; ORB words "
                "are actual binary descriptors selected by farthest-first Hamming distance. "
                "Image histograms rank a saved gallery, without proving geometry, pose, visibility "
                "or production accuracy. The rearranged query exposes lack of spatial "
                "layout checks. "
                "See README.md, SOURCES.md and benchmark.json."
            )
            graph.save(HERE / (case + ".json"))
            cases[case] = {"target": target, "states": states, "images": names}
        finally:
            graph.close()
    (HERE / "truth.json").write_bytes(
        encoded(
            {
                "cases": cases,
                "training": training,
                "queries": queries,
                "labels": LABELS,
                "rankings": results,
                "asset_sha256": {
                    name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()
                },
            }
        )
    )


if __name__ == "__main__":
    build()
