"""Explicit pinned Apache HITNET ETH3D ONNX preparation; no framework install."""

import argparse
import hashlib
import json
import tarfile
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
URL = "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/142_HITNET/resources.tar.gz"
MEMBER = "eth3d/saved_model_480x640/model_float32.onnx"
SIZE = 3548840
SHA256 = "8c5505ab77f83c7e9aee6901b2a6647fa1dcfc57e5775c5a903c6b555c210cb2"
PARAMETERS = {
    "left_name": "input",
    "input_mode": "concatenated",
    "color_format": "gray",
    "mean": [0.0],
    "std": [1.0],
    "layout": "NHWC",
    "engine": "new",
    "output_names": ["reference_output_disparity"],
}


class BoundedStream:
    def __init__(self, stream):
        self.stream, self.remaining = stream, 500 * 1024**2

    def read(self, size=-1):
        raw = self.stream.read(min(size if size >= 0 else self.remaining + 1, self.remaining + 1))
        self.remaining -= len(raw)
        if self.remaining < 0:
            raise ValueError("HITNET archive exceeds 500 MiB compressed guard")
        return raw


def download():
    with urllib.request.urlopen(URL, timeout=40) as response:
        with tarfile.open(fileobj=BoundedStream(response), mode="r|gz") as archive:
            total = 0
            for index, member in enumerate(archive):
                total += member.size
                if index > 10000 or member.size < 0 or total > 2 * 1024**3:
                    raise ValueError("HITNET archive exceeds member/expanded-size guard")
                if member.name == MEMBER:
                    if not member.isfile() or member.size != SIZE:
                        raise ValueError("Pinned HITNET archive member differs")
                    return archive.extractfile(member).read(SIZE + 1)
    raise ValueError("Pinned HITNET ONNX member missing")


def prepare(source=None, *, directory=None):
    from opencv_dataflow.learned_stereo import LearnedStereo, preprocessing
    from opencv_dataflow.model_contracts import configuration, export_contract, read_contract

    target = HERE / ".local/hitnet" if directory is None else Path(directory)
    block = LearnedStereo(parameters=PARAMETERS)
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
                raise FileExistsError("Different existing HITNET bundle preserved")
            with model.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != SHA256:
                    raise FileExistsError("Different existing HITNET model preserved")
            return target
        if source is None:
            raw = download()
        else:
            with Path(source).open("rb") as stream:
                raw = stream.read(SIZE + 1)
        if len(raw) != SIZE or hashlib.sha256(raw).hexdigest() != SHA256:
            raise ValueError("HITNET bytes differ from pinned native ONNX")
        with tempfile.TemporaryDirectory(prefix="dataflow-hitnet-") as temporary:
            model = Path(temporary) / "model.onnx"
            model.write_bytes(raw)
            block.parameters["model_path"] = str(model)
            export_contract(block, target, preprocessing=expected)
        return target
    finally:
        block.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Reuse identical pinned native ONNX")
    print(json.dumps(str(prepare(parser.parse_args().model))))
