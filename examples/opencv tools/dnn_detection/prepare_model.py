"""Explicit pinned download into ignored local storage, never called by graph execution."""

import hashlib
import urllib.request
from pathlib import Path

REVISION = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
SHA256 = "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
SIZE = 35_858_002
URL = (
    f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{REVISION}/models/"
    "object_detection_yolox/object_detection_yolox_2022nov.onnx"
)


def prepare(path):
    if path.exists():
        if path.stat().st_size != SIZE:
            raise FileExistsError("Existing different weights preserved")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != SHA256:
                raise FileExistsError("Existing different weights preserved")
    else:
        with urllib.request.urlopen(URL, timeout=45) as response:
            data = response.read(SIZE + 1)
        if len(data) != SIZE or hashlib.sha256(data).hexdigest() != SHA256:
            raise ValueError("Downloaded model does not match the pinned size/digest")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    print(f"Verified OpenCV Zoo YOLOX-S ({SIZE} bytes, Apache 2.0): {path}")


if __name__ == "__main__":
    prepare(Path(__file__).resolve().parents[3] / ".local/models/yolox_s.onnx")
