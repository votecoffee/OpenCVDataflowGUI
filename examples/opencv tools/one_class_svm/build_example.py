"""Build original known-good grooves and held-out one-class anomaly controls."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import ProjectSession, make_registry, validate_graph
from opencv_dataflow.classifier_common import model_revision
from opencv_dataflow.one_class_svm import model_bytes

HERE = Path(__file__).resolve().parent
TRAIN_COUNT = 20
GAMMA = 0.05


def surface(seed, tilt, *, defect=False):
    y, x = np.indices((32, 32))
    rng = np.random.default_rng(seed)
    phase = rng.uniform(-0.8, 0.8)
    image = 150 + 38 * np.cos((x + (y - 15.5) * tilt + phase) * np.pi / 4)
    image += rng.normal(0, 0.75, image.shape)
    if defect:
        image[14:18, 4:28] = 45
    return np.rint(image).astype(np.uint8)


def add(kind, ident, parameters=None):
    return {"op": "add", "type": kind, "id": ident, "parameters": parameters or {}}


def wire(source, output, target, input_port):
    return {
        "op": "connect",
        "source": source,
        "output": output,
        "target": target,
        "input": input_port,
    }


def write_project(registry, folder, name, operations):
    with ProjectSession(registry, directory=folder) as session:
        session.apply([*operations, {"op": "arrange"}])
        validation = validate_graph(session.graph)
        if not validation["valid"]:
            raise ValueError(validation["issues"])
        session.save(folder / f"{name}.json")


def source(parameters):
    return add(
        "input.file", "image", {**parameters, "path_base": "project", "color_format": "gray"}
    )


def inspect(registry, folder, name, image_name, model, *, alternate_hog=False):
    write_project(
        registry,
        folder,
        name,
        [
            source({"path": f"assets/{image_name}.png"}),
            add("inspect.hog", "hog", {"gamma_correction": False} if alternate_hog else None),
            add("ml.one_class_svm_load", "load", {"path": f"assets/{model}.json"}),
            add("ml.one_class_svm_predict", "inspect"),
            wire("image", "image", "hog", "image"),
            wire("image", "image", "inspect", "image"),
            wire("hog", "descriptor", "inspect", "samples"),
            wire("load", "model", "inspect", "model"),
        ],
    )


def build(folder):
    folder.mkdir(parents=True, exist_ok=True)
    assets = folder / "assets"
    assets.mkdir()
    images = {
        f"good-{i:02d}": surface(i, tilt) for i, tilt in enumerate(np.linspace(-0.14, 0.14, 20))
    }
    images.update(
        {
            "heldout-normal": surface(101, -0.03),
            "heldout-defect": surface(101, -0.03, defect=True),
            "heldout-tilted": surface(102, 0.30),
        }
    )
    for name, image in images.items():
        if not cv2.imwrite(str(assets / f"{name}.png"), image):
            raise OSError("Could not write " + name)

    registry = make_registry()
    write_project(
        registry,
        folder,
        "train",
        [
            source(
                {
                    "source_mode": "files",
                    "paths": [f"assets/good-{i:02d}.png" for i in range(TRAIN_COUNT)],
                }
            ),
            add("inspect.hog", "hog"),
            add("flow.batch", "batch", {"size": TRAIN_COUNT}),
            add("ml.one_class_svm_train", "train", {"gamma": GAMMA}),
            wire("image", "image", "hog", "image"),
            wire("hog", "descriptor", "batch", "item_1"),
            wire("batch", "group", "train", "samples"),
        ],
    )
    with ProjectSession.open(folder / "train.json", registry) as session:
        session.graph.set_run_profile("lean")
        receipt = session.run(all_images=True)
        if not receipt["passed"]:
            raise RuntimeError("Training project failed")
        fit = session.graph.blocks["train"].last_result
        if fit is None or fit.outputs["report"]["samples"] != TRAIN_COUNT:
            raise RuntimeError("Training batch did not complete")
        model = fit.outputs["model"]
    (assets / "reference.json").write_bytes(model_bytes(model))
    strict = copy.deepcopy(model)
    strict["settings"]["minimum_inlier_margin"] = 0.2
    strict["revision"] = model_revision(strict)
    (assets / "strict.json").write_bytes(model_bytes(strict))
    cases = (
        ("normal", "heldout-normal", "reference"),
        ("defect", "heldout-defect", "reference"),
        ("tilted_good", "heldout-tilted", "reference"),
        ("strict_cutoff", "heldout-normal", "strict"),
        ("wrong_features", "heldout-normal", "reference"),
    )
    decisions = {}
    for name, image_name, reference in cases:
        inspect(
            registry, folder, name, image_name, reference, alternate_hog=name == "wrong_features"
        )
        with ProjectSession.open(folder / f"{name}.json", registry) as session:
            session.run()
            block = session.graph.blocks["inspect"]
            decisions[name] = {
                "state": block.state,
                **(block.last_result.outputs["report"] if block.last_result else {}),
            }
    expected = {
        "normal": "pass",
        "defect": "fail",
        "tilted_good": "fail",
        "strict_cutoff": "fail",
        "wrong_features": "error",
    }
    assert {name: item["state"] for name, item in decisions.items()} == expected, decisions
    truth = {
        "schema": "one_class_svm_fixture_v1",
        "training_seeds": list(range(TRAIN_COUNT)),
        "heldout_seeds": [101, 102],
        "defect_xyxy": [4, 14, 28, 18],
        "assets_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(assets.iterdir())
        },
        "results": decisions,
    }
    (folder / "truth.json").write_text(
        json.dumps(truth, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(decisions, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    build(parser.parse_args().output.resolve())
