"""Native engine discrimination using original public embedded-weight ONNX fixtures."""

import hashlib
import json
from pathlib import Path

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parents[1]
SOURCE = HERE.parent / "dnn_detection"
CASES = ("classic_rgb", "new_rgb", "auto_rgb", "new_proposals", "classic_rejected")


def annotate(graph, registry, message):
    if "attribution" in graph.blocks:
        graph.remove("attribution")
    note = graph.add(registry.create("layout.comment", block_id="attribution"))
    note.name = "Original MIT engine dispatch controls"
    note.node_size, note.position = "large", [0, -510]
    note.description = (
        "Original MIT color image and embedded-weight ONNX fixtures by OpenCVDataflowGUI "
        "contributors; copied unchanged from examples/assets and dnn_detection. "
        "No outside images or weights. These demonstrate model importing, not trained "
        "recognition or accelerator accuracy. Classic uses a frozen temporary file to "
        "preserve explicit native dispatch; the file is removed before forward. "
        + message
        + " See README.md, SOURCES.md and truth.json."
    )


def build():
    assets = HERE / "assets"
    if assets.exists():
        raise FileExistsError("Existing engine fixture assets preserved")
    source_truth = json.loads((SOURCE / "truth.json").read_bytes())
    files = {"rgb_average.onnx": (EXAMPLES / "assets/rgb_average.onnx").read_bytes()}
    for name in ("color.png", "proposal_heads.onnx"):
        raw = (SOURCE / "assets" / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != source_truth["asset_sha256"][name]:
            raise ValueError("Original public fixture changed: " + name)
        files[name] = raw
    assets.mkdir()
    for name, raw in files.items():
        (assets / name).write_bytes(raw)
    registry = builtin_registry()
    for name in CASES:
        rgb = name.endswith("_rgb")
        graph = Graph() if rgb else Graph.load(SOURCE / "stretch.json", registry)
        try:
            graph.project_dir = HERE
            if rgb:
                graph.add(
                    registry.create(
                        "input.file",
                        block_id="source",
                        parameters={"path": "assets/color.png", "path_base": "project"},
                    )
                )
                infer = graph.add(
                    registry.create(
                        "ml.dnn",
                        block_id="infer",
                        parameters={
                            "model_path": "assets/rgb_average.onnx",
                            "width": 32,
                            "height": 32,
                            "engine": name.removesuffix("_rgb"),
                            "receipt_mode": "detailed",
                        },
                    )
                )
                infer.position = [380, 0]
                graph.connect("source", "image", "infer", "image")
                message = (
                    "Run and inspect Tensors/Receipt: expected RGB means 220/255, 180/255, 30/255."
                )
            else:
                graph.blocks["infer"].parameters["engine"] = (
                    "classic" if name == "classic_rejected" else "new"
                )
                graph.blocks["infer"].parameters["receipt_mode"] = "detailed"
                message = (
                    "Intentional classic importer error: no current tensor/receipt events. "
                    "The same model is accepted by New in new_proposals.json."
                    if name == "classic_rejected"
                    else "New engine accepts this fixture; three decoded proposals "
                    "and their overlays are visible."
                )
            graph.validation_sets = [
                {
                    "id": "engine",
                    "name": "Engine import control",
                    "source": "source",
                    "cases": [
                        {
                            "id": "color",
                            "path": "assets/color.png",
                            "expectation": "must_pass",
                            "image_sha256": hashlib.sha256(files["color.png"]).hexdigest(),
                        }
                    ],
                }
            ]
            annotate(graph, registry, message)
            graph.save(HERE / (name + ".json"))
        finally:
            graph.close()
    (HERE / "truth.json").write_bytes(
        (
            json.dumps(
                {
                    "asset_sha256": {
                        n: hashlib.sha256(raw).hexdigest() for n, raw in files.items()
                    },
                    "cases": {
                        name: {
                            "passed": name != "classic_rejected",
                            "engine": name.removesuffix("_rgb")
                            if name.endswith("_rgb")
                            else "classic"
                            if name == "classic_rejected"
                            else "new",
                        }
                        for name in CASES
                    },
                    "rgb_expected": [220 / 255, 180 / 255, 30 / 255],
                    "proposal_expected_count": 3,
                    "scope": "CPU native loader dispatch; "
                    "no accelerator or trained recognition claim",
                },
                indent=2,
            )
            + "\n"
        ).encode()
    )


if __name__ == "__main__":
    build()
