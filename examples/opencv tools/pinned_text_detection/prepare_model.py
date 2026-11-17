"""Explicitly prepare the immutable Apache-2.0 PP-OCRv3 English detector."""

import argparse
import hashlib
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REVISION = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
SHA256 = "03f550c6b406fda8bf54bd8327815f6c7e2edd98cea02348c93d879254366587"
SIZE = 2_423_490
URL = (
    f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{REVISION}/"
    "models/text_detection_ppocr/text_detection_en_ppocrv3_2023may.onnx"
)
PARAMETERS = {
    "width": 736,
    "height": 736,
    "color_format": "bgr",
    "resize_mode": "letterbox",
    "padding_value": 255.0,
    "scale": 1 / 255,
    "mean": [123.675, 116.28, 103.53],
    "std": [0.229, 0.224, 0.225],
    "input_name": "x",
    "output_names": ["4"],
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

    directory = HERE / ".local/ppocr_en" if directory is None else Path(directory)
    tool = DNNInference(parameters=PARAMETERS)
    try:
        preprocessing = {"x": image_preprocessing(tool.parameters)}
        if directory.exists():
            contract = read_contract(directory / "contract.json")
            model = directory / "model.onnx"
            if (
                contract["artifact"] != {"file": "model.onnx", "bytes": SIZE, "sha256": SHA256}
                or contract["configuration"] != configuration(tool)
                or contract["preprocessing"] != preprocessing
                or model.stat().st_size != SIZE
            ):
                raise FileExistsError("Different existing PP-OCR bundle preserved")
            with model.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != SHA256:
                    raise FileExistsError("Different existing model weights preserved")
        else:
            if source is None:
                with urllib.request.urlopen(URL, timeout=60) as response:
                    raw = response.read(SIZE + 1)
            else:
                with Path(source).open("rb") as stream:
                    raw = stream.read(SIZE + 1)
            if len(raw) != SIZE or hashlib.sha256(raw).hexdigest() != SHA256:
                raise ValueError("PP-OCR weights do not match the pinned size/digest")
            with tempfile.TemporaryDirectory(prefix="dataflow-ppocr-") as temporary:
                path = Path(temporary) / "model.onnx"
                path.write_bytes(raw)
                tool.parameters["model_path"] = str(path)
                export_contract(tool, directory, preprocessing=preprocessing)
        return directory
    finally:
        tool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Reuse the identical already-downloaded weights")
    print(f"Verified pinned PP-OCRv3: {prepare(parser.parse_args().model)}")
