"""Original ONNX model/processing contracts, numerical controls and receipts."""

import hashlib
import json
import tempfile
from pathlib import Path

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.model_contracts import export_contract, image_preprocessing

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
CASES = {
    "pinned_single": {},
    "detailed_hashes": {},
    "receipts_off": {},
    "settings_mismatch": {"error": "settings differ"},
    "checksum_mismatch": {"error": "checksum differs"},
    "pinned_multi": {},
    "wrong_preprocessing": {"error": "preprocessing differs"},
    "pinned_batch": {},
    "cpu_fallback": {},
    "strict_target": {"error": "unavailable"},
}


def write_asset(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != raw:
        raise FileExistsError("Different existing assets preserved; regenerate in a fresh folder")
    path.write_bytes(raw)


def build():
    registry, truth = builtin_registry(), {}
    assets = []
    for i in range(3):
        target = HERE / f"assets/{i}.png"
        write_asset(target, (SOURCE / f"batch_inference/assets/{i}.png").read_bytes())
        assets.append(target)
    for case, expected in CASES.items():
        multi, batch = case in ("pinned_multi", "wrong_preprocessing"), case == "pinned_batch"
        if multi:
            graph = Graph.load(SOURCE / "multi_input_inference/paired_images.json", registry)
            graph.blocks["source"].parameters["path"] = "assets/0.png"
            model = graph.blocks["multi"]
            model.parameters["model_path"] = str(SOURCE / "multi_input_inference/assets/pair.onnx")
            preprocessing = {
                "left": image_preprocessing(graph.blocks["left"].parameters),
                "right": image_preprocessing(graph.blocks["right"].parameters),
            }
        elif batch:
            graph = Graph.load(SOURCE / "batch_inference/dynamic_tail.json", registry)
            for i in range(3):
                graph.blocks[f"source_{i}"].parameters["path"] = f"assets/{i}.png"
            model = graph.blocks["infer"]
            model.parameters["model_path"] = str(SOURCE / "batch_inference/assets/dynamic.onnx")
            preprocessing = None
        else:
            graph = Graph()
            graph.add(
                registry.create(
                    "input.file",
                    block_id="source",
                    parameters={"path": "assets/0.png", "path_base": "project"},
                )
            )
            model = graph.add(
                registry.create(
                    "ml.dnn",
                    block_id="infer",
                    parameters={
                        "model_path": str(SOURCE / "batch_inference/assets/dynamic.onnx"),
                        "width": 8,
                        "height": 8,
                        "input_name": "image",
                        "output_names": ["scores"],
                        "engine": "classic",
                    },
                )
            )
            graph.add(
                registry.create(
                    "ml.classify",
                    block_id="classify",
                    parameters={
                        "output_name": "scores",
                        "labels": ["red", "green", "blue"],
                        "accepted_labels": ["red"],
                    },
                )
            )
            graph.connect("source", "image", "infer", "image")
            graph.connect("infer", "tensors", "classify", "tensors")
            preprocessing = None
        try:
            graph.project_dir = HERE
            for block in graph.blocks.values():
                block.project_dir = HERE
            if case in ("cpu_fallback", "strict_target"):
                model.parameters.update(
                    backend="cuda",
                    target="cuda",
                    fallback_policy="cpu_if_unavailable" if case == "cpu_fallback" else "error",
                )
            folder = (
                "multi"
                if multi
                else "batch"
                if batch
                else case
                if case in ("checksum_mismatch", "cpu_fallback", "strict_target")
                else "single"
            )
            with tempfile.TemporaryDirectory() as temporary:
                source = Path(temporary) / "bundle"
                export_contract(model, source, preprocessing=preprocessing)
                for name in ("model.onnx", "contract.json"):
                    raw = (source / name).read_bytes()
                    if case == "checksum_mismatch" and name == "model.onnx":
                        raw = raw[:-1] + bytes([raw[-1] ^ 1])
                    target = HERE / "bundles" / folder / name
                    write_asset(target, raw)
                    assets.append(target)
            model.parameters.update(contract_path=f"bundles/{folder}/contract.json", model_path="")
            model.exposed_outputs.append("receipt")
            if case == "settings_mismatch":
                model.parameters["scale"] *= 2
            if case == "wrong_preprocessing":
                graph.blocks["left"].parameters["mean"] = [2, 0, 0]
            if case == "detailed_hashes":
                model.parameters["receipt_mode"] = "detailed"
            if case == "receipts_off":
                model.parameters["receipt_mode"] = "off"
            if not multi and not batch:
                graph.blocks["source"].position = [0, 0]
                model.position = [400, 0]
                graph.blocks["classify"].position = [900, 0]
            note = graph.blocks.get("attribution") or graph.add(
                registry.create("layout.comment", block_id="attribution")
            )
            note.name, note.node_size, note.position = (
                "Original ONNX contracts - MIT",
                "large",
                [0, -500],
            )
            note.description = (
                "Original MIT arithmetic ONNX models, color images, graphs and independent "
                "references by OpenCVDataflowGUI contributors. Reuses our batch/multi-input "
                "controls. No learned weights, customer or third-party assets. Contract pins "
                "model bytes and processing settings; Detailed adds tensor hashes. Off still "
                "checks the contract. Invalid controls emit no inference results. CPU fallback "
                "only applies when the requested CUDA target is unavailable. Native per-layer "
                "placement remains unverified. See README and truth.json."
            )
            graph.save(HERE / (case + ".json"))
            rgb = [200 / 255, 40 / 255, 20 / 255]
            gray = (0.299 * 200 + 0.587 * 40 + 0.114 * 20) / 255
            truth[case] = {
                **expected,
                "block": model.id,
                "scores": [v - gray for v in rgb] if multi else rgb,
                "label": "green" if batch else "red",
            }
        finally:
            graph.close()
    document = {
        "license": "MIT - original arithmetic controls",
        "cases": truth,
        "asset_sha256": {
            p.relative_to(HERE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(assets))
        },
    }
    (HERE / "truth.json").write_bytes((json.dumps(document, indent=2) + "\n").encode())


if __name__ == "__main__":
    build()
