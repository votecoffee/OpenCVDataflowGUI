"""Fetch and verify the optional OpenCV reference forest for this example."""

import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

from opencv_dataflow.structured_edges import MODEL_SHA256, MODEL_SIZE

URL = (
    "https://raw.githubusercontent.com/opencv/opencv_extra/"
    "3becd947dd6bf6340047de8ca4445a70271e6073/testdata/cv/ximgproc/model.yml.gz"
)


def verify(data):
    if len(data) != MODEL_SIZE or hashlib.sha256(data).hexdigest() != MODEL_SHA256:
        raise ValueError("Structured-edge model differs from the pinned OpenCV release")


def prepare():
    path = Path(__file__).resolve().parent / ".local" / "models" / "model.yml.gz"
    if path.exists():
        with path.open("rb") as stream:
            verify(stream.read(MODEL_SIZE + 1))
        print(f"Verified {path}")
        return
    with urlopen(URL, timeout=60) as response:
        data = response.read(MODEL_SIZE + 1)
    verify(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Downloaded and verified {path}")


if __name__ == "__main__":
    prepare()
