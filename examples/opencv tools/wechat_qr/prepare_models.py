"""Fetch the OpenCV-published ONNX pair for the optional model comparison."""

import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
BASE = (
    "https://huggingface.co/opencv/opencv_contribution/resolve/"
    "18a5ee169b869025bf8d93d9ca4dc56894fee9dd/wechat_qr"
)
MODELS = {
    "detector": (
        "detect_2026april.onnx",
        985788,
        "a533c75ec6866759eb28ff8cfb11d30b4887f838124a9ff4ac477e25ad6c7f9f",
    ),
    "super_resolution": (
        "sr_2026april.onnx",
        26467,
        "125ab273d59717c5bf4c5dfb674aa9630d3df42e05b99d2d6fa5ffea32fe34f6",
    ),
}
LICENSE = (
    "LICENSE",
    20273,
    "f8549a71cfb8bfa34e6b7d6491fbfac8034673294e2fd80493389e460a496cb5",
)


def verified(data, length, digest):
    return len(data) == length and hashlib.sha256(data).hexdigest() == digest


def prepare():
    output = HERE / ".local" / "models"
    output.mkdir(parents=True, exist_ok=True)
    for name, length, digest in (*MODELS.values(), LICENSE):
        path = output / name
        if path.exists():
            with path.open("rb") as stream:
                data = stream.read(length + 1)
            if not verified(data, length, digest):
                raise ValueError(f"Existing file differs from the pinned artifact: {path}")
            print(f"Verified {path}")
            continue
        with urlopen(f"{BASE}/{name}", timeout=60) as response:
            data = response.read(length + 1)
        if not verified(data, length, digest):
            raise ValueError(f"Downloaded file differs from the pinned artifact: {name}")
        with NamedTemporaryFile(dir=output, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        try:
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Downloaded and verified {path}")


if __name__ == "__main__":
    prepare()
