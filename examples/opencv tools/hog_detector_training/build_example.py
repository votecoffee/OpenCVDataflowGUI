"""Regenerate original HOG training, mining, and independent full-image controls."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import ProjectSession, validate_graph
from opencv_dataflow.classifier_common import canonical_json

HERE = Path(__file__).resolve().parent


def patch(mode, seed):
    image = np.full((32, 32), 30, np.uint8)
    if mode in ("part", "solid", "mirror"):
        cv2.rectangle(image, (8, 6), (23, 25), 205, -1)
        if mode == "part":
            cv2.rectangle(image, (14, 11), (23, 19), 30, -1)
        elif mode == "mirror":
            cv2.rectangle(image, (8, 11), (17, 19), 30, -1)
    elif mode == "ring":
        cv2.circle(image, (16, 16), 10, 200, 3)
    elif mode == "cross":
        cv2.line(image, (7, 7), (24, 24), 200, 3)
        cv2.line(image, (7, 24), (24, 7), 200, 3)
    return (image.astype(int) + np.random.default_rng(seed).integers(-2, 3, image.shape)).astype(
        np.uint8
    )


def scene(objects, seed):
    image = np.full((96, 144), 30, np.uint8)
    for i, (x, y, kind) in enumerate(objects):
        image[y : y + 32, x : x + 32] = patch(kind, seed + i)
    return image


def add(kind, identity, **parameters):
    return {"op": "add", "type": kind, "id": identity, "parameters": parameters}


def connect(source, output, target, input_port):
    return {
        "op": "connect",
        "source": source,
        "output": output,
        "target": target,
        "input": input_port,
    }


def image_source(identity, files):
    paths = [f"assets/{name}.png" for name in files]
    mode = {"path": paths[0]} if len(paths) == 1 else {"source_mode": "files", "paths": paths}
    return add("input.file", identity, path_base="project", color_format="gray", **mode)


def save(folder, name, operations, description):
    note = [
        add("layout.comment", "attribution"),
        {
            "op": "update",
            "id": "attribution",
            "fields": {
                "name": "Synthetic HOG training controls (MIT)",
                "description": description,
            },
        },
    ]
    with ProjectSession(directory=folder) as session:
        session.apply([*operations, *note, {"op": "arrange"}])
        checked = validate_graph(session.graph)
        if not checked["valid"]:
            raise ValueError(checked["issues"])
        session.save(folder / f"{name}.json")


def compare(folder, name, image_name, expected):
    ops = [
        image_source("scene", [image_name]),
        add("ml.hog_detector_load", "load_before", path="assets/baseline.json"),
        add("ml.hog_detector_load", "load_after", path="assets/detector.json"),
        add(
            "ml.hog_detect",
            "before",
            hit_threshold=0.5,
            pyramid_levels=1,
            minimum_count=0,
            nms_iou=0.2,
        ),
        add(
            "ml.hog_detect",
            "after",
            hit_threshold=0.5,
            pyramid_levels=1,
            minimum_count=len(expected),
            maximum_count=len(expected),
            nms_iou=0.2,
        ),
    ]
    for key in ("before", "after"):
        ops += [
            connect("scene", "image", key, "image"),
            connect(f"load_{key}", "model", key, "model"),
        ]
    save(
        folder,
        name,
        ops,
        "Original MIT bracket controls. The scene uses seeds and locations separate "
        "from crop training and mining. Both detectors use the same 0.5 threshold. "
        "Inspect the before and after boxes to compare false positives.",
    )
    with ProjectSession.open(folder / f"{name}.json") as session:
        assert session.run()["passed"]
        result = {}
        for key in ("before", "after"):
            items = session.graph.blocks[key].last_result.outputs["collection"]["items"]
            result[key] = sorted([item["geometry"]["bbox"] for item in items])
    assert result["after"] == sorted(expected), result
    assert set(map(tuple, expected)).issubset(map(tuple, result["before"])), result
    assert len(result["before"]) > len(result["after"]), result
    return result


def build(folder):
    folder.mkdir(exist_ok=True, parents=True)
    assets = folder / "assets"
    assets.mkdir()
    positives = [f"positive-{i}" for i in range(6)]
    negatives = [f"negative-{i}" for i in range(6)]
    images = {name: patch("part", 100 + i) for i, name in enumerate(positives)}
    images.update(
        {
            name: patch(("ring", "cross", "blank")[i % 3], 200 + i)
            for i, name in enumerate(negatives)
        }
    )
    images.update(
        {
            "mining-a": scene([(16, 16, "solid"), (88, 48, "mirror")], 300),
            "mining-b": scene([(64, 48, "solid"), (104, 16, "solid")], 400),
            "query-a": scene(
                [(16, 24, "part"), (96, 56, "part"), (72, 8, "solid"), (48, 56, "mirror")], 500
            ),
            "query-b": scene([(8, 48, "part"), (104, 16, "part"), (56, 32, "solid")], 600),
            "query-empty": scene([(48, 40, "solid"), (104, 16, "mirror")], 700),
        }
    )
    for name, image in images.items():
        assert cv2.imwrite(str(assets / (name + ".png")), image)
    save(
        folder,
        "train",
        [
            image_source("positive_source", positives),
            image_source("negative_source", negatives),
            image_source("mining_a", ["mining-a"]),
            image_source("mining_b", ["mining-b"]),
            add("flow.batch", "positive_batch", size=len(positives)),
            add("flow.batch", "negative_batch", size=len(negatives)),
            add("flow.batch", "background_batch", size=2),
            add("ml.hog_detector_train", "train", maximum_hard_negatives=20),
            connect("positive_source", "image", "positive_batch", "item_1"),
            connect("negative_source", "image", "negative_batch", "item_1"),
            connect("mining_a", "image", "background_batch", "item_1"),
            connect("mining_b", "image", "background_batch", "item_2"),
            connect("positive_batch", "group", "train", "positives"),
            connect("negative_batch", "group", "train", "negatives"),
            connect("background_batch", "group", "train", "backgrounds"),
        ],
        "Original MIT crops and scenes. Run all to collect six windows per class, "
        "mine two separate target-free frames, then refit and compare the models.",
    )
    with ProjectSession.open(folder / "train.json") as session:
        session.graph.set_run_profile("lean")
        outcome = session.run(all_images=True)
        block = session.graph.blocks["train"]
        assert outcome["passed"] and block.state == "pass", block.error
        results = block.last_result.outputs
        for filename, key in (("baseline", "baseline_model"), ("detector", "model")):
            (assets / f"{filename}.json").write_bytes(
                (canonical_json(results[key]) + "\n").encode()
            )
        training = results["report"]
    expected = {
        "heldout_parts": [[16, 24, 32, 32], [96, 56, 32, 32]],
        "heldout_second": [[8, 48, 32, 32], [104, 16, 32, 32]],
        "heldout_background": [],
    }
    observed = {
        name: compare(folder, name, image, expected[name])
        for name, image in zip(expected, ("query-a", "query-b", "query-empty"), strict=True)
    }
    receipt = {
        "schema": "hog_hard_negative_fixture_v1",
        "training": training,
        "heldout_boxes": expected,
        "observed_boxes": observed,
        "assets_sha256": {
            file.name: hashlib.sha256(file.read_bytes()).hexdigest()
            for file in sorted(assets.iterdir())
        },
    }
    (folder / "truth.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "mined": len(training["selected_mined_negatives"]),
                "mining_before": training["mining_candidates_before"],
                "mining_after": training["mining_candidates_after"],
                "heldout": {
                    key: {k: len(v) for k, v in item.items()} for key, item in observed.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    build(parser.parse_args().output.resolve())
