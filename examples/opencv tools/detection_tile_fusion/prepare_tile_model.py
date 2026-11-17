"""Explicit pinned tile-detector YOLOX download and sealed local bundle."""

import argparse
import hashlib
import tempfile
import urllib.request
from pathlib import Path

REVISION = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
SHA256 = "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
SIZE = 35_858_002
URL = (
    f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{REVISION}/models/"
    "object_detection_yolox/object_detection_yolox_2022nov.onnx"
)
HERE = Path(__file__).resolve().parent
PARAMETERS = {
    "width": 640,
    "height": 640,
    "color_format": "rgb",
    "resize_mode": "letterbox",
    "padding_value": 114,
    "scale": 1,
    "mean": [0.0, 0.0, 0.0],
    "std": [1.0, 1.0, 1.0],
    "input_name": "images",
    "output_names": ["output"],
    "engine": "classic",
    "backend": "opencv",
    "target": "cpu",
    "receipt_mode": "summary",
}


def prepare(source=None, *, directory=None):
    from opencv_dataflow.dnn_blocks import DNNInference
    from opencv_dataflow.model_contracts import (
        configuration,
        export_contract,
        image_preprocessing,
        read_contract,
    )

    directory = HERE / ".local/yolox_raw" if directory is None else Path(directory)
    tool = DNNInference(parameters=PARAMETERS)
    preprocessing = {"images": image_preprocessing(tool.parameters)}
    if directory.exists():
        contract = read_contract(directory / "contract.json")
        model = directory / "model.onnx"
        if (
            contract["artifact"] != {"file": "model.onnx", "bytes": SIZE, "sha256": SHA256}
            or contract["configuration"] != configuration(tool)
            or contract["preprocessing"] != preprocessing
            or model.stat().st_size != SIZE
        ):
            raise FileExistsError("Different existing model bundle preserved")
        with model.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != SHA256:
                raise FileExistsError("Different existing weights preserved")
    else:
        if source is None:
            with urllib.request.urlopen(URL, timeout=45) as response:
                raw = response.read(SIZE + 1)
        else:
            with Path(source).open("rb") as stream:
                raw = stream.read(SIZE + 1)
        if len(raw) != SIZE or hashlib.sha256(raw).hexdigest() != SHA256:
            raise ValueError("Weights do not match the pinned size/digest")
        with tempfile.TemporaryDirectory(prefix="dataflow-tile-yolox-") as temporary:
            path = Path(temporary) / "model.onnx"
            path.write_bytes(raw)
            tool.parameters["model_path"] = str(path)
            export_contract(tool, directory, preprocessing=preprocessing)
    print(f"Verified OpenCV Zoo YOLOX-S (Apache 2.0): {directory}")
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Reuse identical pinned source weights")
    prepare(parser.parse_args().model)
