"""Explicit pinned SSD download and sealed local model bundle; no runtime downloads."""

import argparse
import hashlib
import tempfile
import urllib.request
from pathlib import Path

REVISION = "c5cae0f1942c8d7727372c7b1f6b485aa39e4421"
SHA256 = "5ec1e058e666ef980a6213ac40088877865aafb2b6676b9af204b3828e4bb9d5"
SIZE = 80_366_315
URL = (
    f"https://media.githubusercontent.com/media/onnx/models/{REVISION}/"
    "validated/vision/object_detection_segmentation/ssd/model/ssd-12.onnx"
)
HERE = Path(__file__).resolve().parent
PARAMETERS = {
    "width": 1200,
    "height": 1200,
    "color_format": "rgb",
    "resize_mode": "stretch",
    "mean": [123.675, 116.28, 103.53],
    "scale": 1 / 255,
    "std": [0.229, 0.224, 0.225],
    "input_name": "image",
    "output_names": ["bboxes", "labels", "scores"],
    "engine": "new",
    "backend": "opencv",
    "target": "cpu",
    "receipt_mode": "detailed",
}


def prepare(source=None, *, directory=None):
    from opencv_dataflow.dnn_blocks import DNNInference
    from opencv_dataflow.model_contracts import (
        configuration,
        export_contract,
        image_preprocessing,
        read_contract,
    )

    directory = HERE / ".local/ssd_resnet34" if directory is None else Path(directory)
    tool = DNNInference(parameters=PARAMETERS)
    preprocessing = {"image": image_preprocessing(tool.parameters)}
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
        with tempfile.TemporaryDirectory(prefix="dataflow-ssd-") as temporary:
            path = Path(temporary) / "model.onnx"
            path.write_bytes(raw)
            tool.parameters["model_path"] = str(path)
            export_contract(tool, directory, preprocessing=preprocessing)
    print(f"Verified ONNX Model Zoo SSD ResNet-34 (Apache 2.0): {directory}")
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, help="Reuse existing weights after exact verification"
    )
    prepare(parser.parse_args().model)
