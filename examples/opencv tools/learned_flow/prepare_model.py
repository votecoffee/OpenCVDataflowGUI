"""Explicit SHA-256 pinned OpenCV Zoo RAFT preparation, without training packages."""

import argparse
import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REVISION = "25f423d0e04c31a17254620e58febd7386da523b"
URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
    + REVISION
    + "/models/optical_flow_estimation_raft/optical_flow_estimation_raft_2023aug.onnx"
)
SIZE = 64119337
SHA256 = "3b479e0447894245b822bf0b7392155d31ccd22bb9b0ee55503066bf6f6f869a"
PARAMETERS = {
    "first_name": "0",
    "second_name": "1",
    "output_names": ["12006"],
    "std": [1 / 255, 1 / 255, 1 / 255],
    "engine": "classic",
}


def prepare(source=None, *, directory=None):
    from opencv_dataflow.learned_flow import LearnedFlow, preprocessing
    from opencv_dataflow.model_contracts import configuration, export_contract, read_contract

    target = HERE / ".local/raft" if directory is None else Path(directory)
    block = LearnedFlow(parameters=PARAMETERS)
    try:
        expected = preprocessing(block.parameters)
        identity = {"file": "model.onnx", "bytes": SIZE, "sha256": SHA256}
        if target.exists():
            contract = read_contract(target / "contract.json")
            model = target / "model.onnx"
            if (
                contract["block_type"] != block.type_id
                or contract["configuration"] != configuration(block)
                or contract["preprocessing"] != expected
                or contract["artifact"] != identity
                or model.stat().st_size != SIZE
            ):
                raise FileExistsError("Different existing RAFT bundle preserved")
            with model.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != SHA256:
                    raise FileExistsError("Different existing RAFT weights preserved")
            return target
        if source is None:
            with urllib.request.urlopen(URL, timeout=40) as stream:
                raw = stream.read(SIZE + 1)
        else:
            with Path(source).open("rb") as stream:
                raw = stream.read(SIZE + 1)
        if len(raw) != SIZE or hashlib.sha256(raw).hexdigest() != SHA256:
            raise ValueError("RAFT artifact differs from pinned native bytes")
        with tempfile.TemporaryDirectory(prefix="dataflow-raft-") as temporary:
            model = Path(temporary) / "model.onnx"
            model.write_bytes(raw)
            block.parameters["model_path"] = str(model)
            export_contract(block, target, preprocessing=expected)
        return target
    finally:
        block.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Reuse identical pinned ONNX")
    print(json.dumps(str(prepare(parser.parse_args().model))))
