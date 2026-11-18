"""Explicit pinned MobileNet download/sealing; graph opening and execution never download."""

import argparse
import hashlib
import importlib.util
import tempfile
import urllib.request
from pathlib import Path

from opencv_dataflow.dnn_blocks import DNNInference
from opencv_dataflow.model_contracts import configuration, export_contract, read_contract

HERE = Path(__file__).resolve().parent
REVISION = "c5cae0f1942c8d7727372c7b1f6b485aa39e4421"
MODELS = {
    "reference": (
        "mobilenetv2-12.onnx",
        13964571,
        "c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5",
    ),
    "qdq": (
        "mobilenetv2-12-qdq.onnx",
        3593903,
        "41a36090dafe98f4ad8f9b7fe0b218c56ac3c031e547f0367c30655d2702bffe",
    ),
    "qoperator": (
        "mobilenetv2-12-int8.onnx",
        3655033,
        "cc028fe6cae7bc11a4ff53cfc9b79c920e8be65ce33a904ec3e2a8f66d77f95f",
    ),
}


def preparation():
    spec = importlib.util.spec_from_file_location("quantized_example", HERE / "build_example.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.pretrained_parameters


def prepare(source=None, *, directory=None):
    destination = HERE / ".local" if directory is None else Path(directory)
    params = preparation()
    verified = []
    for kind, (filename, size, digest) in MODELS.items():
        pending = []
        for engine in ("classic", "new", "auto"):
            folder = destination / f"{kind}_{engine}"
            block = DNNInference(parameters=params(engine))
            if folder.exists():
                manifest = read_contract(folder / "contract.json")
                path = folder / "model.onnx"
                if (
                    manifest["artifact"] != {"file": "model.onnx", "bytes": size, "sha256": digest}
                    or manifest["configuration"] != configuration(block)
                    or manifest["preprocessing"] != {}
                    or path.stat().st_size != size
                    or hashlib.sha256(path.read_bytes()).hexdigest() != digest
                ):
                    raise FileExistsError("Different existing MobileNet bundle preserved")
                verified.append(folder)
            else:
                pending.append((folder, block))
        if not pending:
            continue
        if source is None:
            url = (
                f"https://media.githubusercontent.com/media/onnx/models/{REVISION}/"
                f"validated/vision/classification/mobilenet/model/{filename}"
            )
            with urllib.request.urlopen(url, timeout=45) as response:
                raw = response.read(size + 1)
        else:
            with (Path(source) / filename).open("rb") as stream:
                raw = stream.read(size + 1)
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("MobileNet size/hash differs from pinned upstream weights")
        with tempfile.TemporaryDirectory(prefix="dataflow-quantized-") as temporary:
            path = Path(temporary) / "model.onnx"
            path.write_bytes(raw)
            for folder, block in pending:
                block.parameters["model_path"] = str(path)
                export_contract(block, folder)
                verified.append(folder)
    return verified


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, help="Use an existing folder containing the three exact models"
    )
    folders = prepare(parser.parse_args().source)
    print(f"Verified {len(folders)} pinned MobileNet model/settings bundles (Apache 2.0)")
