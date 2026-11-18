"""Explicit pinned MIT MiDaS v2.1 Small preparation; unchanged native ONNX weights."""

import argparse
import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
URL = "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx"
SIZE = 66764249
SHA256 = "2d8c6cb8f415229daf1eb041024208e2608c9f98e17c81cc7c6ecb449c56fd58"
PARAMETERS = {
    "input_name": "0",
    "output_names": ["797"],
    "width": 256,
    "height": 256,
    "color_format": "rgb",
    "mean": [0.0, 0.0, 0.0],
    "std": [1.0, 1.0, 1.0],
    "engine": "classic",
}
VARIANTS = {"native_relative": {}, "native_letterbox": {"resize_mode": "letterbox"}}


def prepare(source=None, *, directory=None):
    from opencv_dataflow.model_contracts import configuration, export_contract, read_contract
    from opencv_dataflow.monocular_depth import MonocularDepth, preprocessing

    directory = HERE / ".local" if directory is None else Path(directory)
    raw = None
    result = []
    for name, changes in VARIANTS.items():
        target = directory / name
        block = MonocularDepth(parameters={**PARAMETERS, **changes})
        try:
            expected = {"0": preprocessing(block.parameters)}
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
                    raise FileExistsError("Different existing MiDaS bundle preserved")
                with model.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != SHA256:
                        raise FileExistsError("Different existing MiDaS weights preserved")
            else:
                if raw is None:
                    if source is None:
                        with urllib.request.urlopen(URL, timeout=40) as stream:
                            raw = stream.read(SIZE + 1)
                    else:
                        with Path(source).open("rb") as stream:
                            raw = stream.read(SIZE + 1)
                    if len(raw) != SIZE or hashlib.sha256(raw).hexdigest() != SHA256:
                        raise ValueError("MiDaS artifact differs from pinned bytes")
                with tempfile.TemporaryDirectory(prefix="dataflow-midas-") as temporary:
                    model = Path(temporary) / "model.onnx"
                    model.write_bytes(raw)
                    block.parameters["model_path"] = str(model)
                    export_contract(block, target, preprocessing=expected)
            result.append(target)
        finally:
            block.close()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Reuse identical pinned native ONNX weights")
    print(json.dumps([str(p) for p in prepare(parser.parse_args().model)], indent=2))
