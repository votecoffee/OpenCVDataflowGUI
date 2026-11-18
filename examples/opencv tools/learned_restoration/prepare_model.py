"""Explicitly prepare a pinned MIT DnCNN-25 export; no training or upstream code import."""

import argparse
import hashlib
import io
import json
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
URL = "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_25.pth"
WEIGHTS_SIZE = 2224903
WEIGHTS_SHA256 = "0451a70de9b672ae037270498fbb1c17a1c1c4403785df586ff65df5b858e5b0"
ONNX_SIZE = 2223887
ONNX_SHA256 = "6e9058f77ac18172ac201eeb9148fc57d5e839c1435e203eb1b03174878106c3"
PARAMETERS = {
    "engine": "classic",
    "color_format": "gray",
    "input_name": "input",
    "output_names": ["output"],
    "tile_width": 48,
    "tile_height": 40,
    "halo": 17,
}
VARIANTS = {
    "native_whole": {},
    "native_tiles": {"processing_mode": "halo_tiles"},
    "native_no_halo": {"processing_mode": "halo_tiles", "halo": 0},
    "native_low_noise": {},
}


def export_bytes(raw):
    import numpy as np
    import torch
    from onnx import TensorProto, checker, helper, numpy_helper

    if len(raw) != WEIGHTS_SIZE or hashlib.sha256(raw).hexdigest() != WEIGHTS_SHA256:
        raise ValueError("DnCNN source weights differ from the pinned artifact")
    state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    expected = {f"model.{2 * i}.{field}" for i in range(17) for field in ("weight", "bias")}
    if not isinstance(state, dict) or state.keys() != expected:
        raise ValueError("Unexpected DnCNN-25 state dictionary")
    nodes, initializers, current = [], [], "input"
    for index in range(17):
        channels = (64 if index < 16 else 1, 1 if index == 0 else 64)
        for field, shape in (("weight", (*channels, 3, 3)), ("bias", (channels[0],))):
            name = f"model.{2 * index}.{field}"
            tensor = state[name].detach().cpu().numpy()
            if tensor.dtype != np.float32 or tensor.shape != shape or not np.isfinite(tensor).all():
                raise ValueError("Unexpected DnCNN tensor shape/type/values")
            initializers.append(numpy_helper.from_array(tensor, name))
        out = f"conv_{index}"
        nodes.append(
            helper.make_node(
                "Conv",
                [current, f"model.{2 * index}.weight", f"model.{2 * index}.bias"],
                [out],
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
            )
        )
        current = out
        if index < 16:
            current = f"relu_{index}"
            nodes.append(helper.make_node("Relu", [out], [current]))
    nodes.append(helper.make_node("Sub", ["input", current], ["output"]))
    graph = helper.make_graph(
        nodes,
        "pinned_dncnn25",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, "height", "width"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, "height", "width"])],
        initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=8,
        producer_name="OpenCVDataflowGUI pinned DnCNN export",
    )
    model.doc_string = (
        "Unchanged MIT Kai Zhang DnCNN-25 weights; 17 fused convs/ReLU; input minus noise."
    )
    checker.check_model(model)
    converted = model.SerializeToString()
    if len(converted) != ONNX_SIZE or hashlib.sha256(converted).hexdigest() != ONNX_SHA256:
        raise ValueError("Converted ONNX differs from the pinned export; preserve and review")
    return converted


def prepare(weights=None, *, directory=None):
    from opencv_dataflow.learned_restoration import RestoreImage, preprocessing
    from opencv_dataflow.model_contracts import configuration, export_contract, read_contract

    directory = HERE / ".local" if directory is None else Path(directory)
    if weights is None:
        # Repeat preparation stays offline when a complete, matching bundle is present.
        if all((directory / case / "contract.json").is_file() for case in VARIANTS):
            raw = None
        else:
            with urllib.request.urlopen(URL, timeout=40) as stream:
                raw = stream.read(WEIGHTS_SIZE + 1)
    else:
        with Path(weights).open("rb") as stream:
            raw = stream.read(WEIGHTS_SIZE + 1)
    converted = export_bytes(raw) if raw is not None else None
    identities = []
    for case, changes in VARIANTS.items():
        block = RestoreImage(parameters={**PARAMETERS, **changes})
        destination = directory / case
        try:
            preparation = {"input": preprocessing(block.parameters)}
            if destination.exists():
                contract = read_contract(destination / "contract.json")
                model = destination / "model.onnx"
                identity = {
                    "file": "model.onnx",
                    "bytes": model.stat().st_size,
                    "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                }
                if (
                    contract["block_type"] != block.type_id
                    or identity != {"file": "model.onnx", "bytes": ONNX_SIZE, "sha256": ONNX_SHA256}
                    or contract["configuration"] != configuration(block)
                    or contract["preprocessing"] != preparation
                    or contract["artifact"] != identity
                    or converted is not None
                    and model.read_bytes() != converted
                ):
                    raise FileExistsError("Different existing DnCNN bundle preserved")
            else:
                with tempfile.TemporaryDirectory(prefix="dataflow-dncnn-") as temporary:
                    path = Path(temporary) / "model.onnx"
                    path.write_bytes(converted)
                    block.parameters["model_path"] = str(path)
                    contract = export_contract(block, destination, preprocessing=preparation)
            identities.append(contract["artifact"])
        finally:
            block.close()
    return identities


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, help="Reuse identical source .pth weights")
    print(json.dumps(prepare(parser.parse_args().weights), indent=2))
