"""Explicit pinned ViT acquisition; shipped examples already contain these small weights."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REVISION = "25f423d0e04c31a17254620e58febd7386da523b"
URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
    + REVISION
    + "/models/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx"
)
SIZE = 714726
SHA256 = "2990f0b7cd44d92afa48cd97db6de7be113fc1d9594fddb74e2725c10478e91d"


def prepare(source=None, *, destination=None):
    target = HERE / "assets/vit.onnx" if destination is None else Path(destination)
    if target.exists():
        raw = target.read_bytes()
        if len(raw) != SIZE or hashlib.sha256(raw).hexdigest() != SHA256:
            raise FileExistsError("Different existing ViT weights preserved")
        return target
    if source is None:
        with urllib.request.urlopen(URL, timeout=40) as r:
            raw = r.read(SIZE + 1)
    else:
        with Path(source).open("rb") as r:
            raw = r.read(SIZE + 1)
    if len(raw) != SIZE or hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("ViT artifact differs from pinned native bytes")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as r:
        r.write(raw)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Reuse identical pinned ONNX")
    print(json.dumps(str(prepare(parser.parse_args().model))))
