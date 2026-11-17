"""Explicit unchanged Apache CRNN baseline; its declared native batch is one."""

import argparse
import hashlib
import string
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REVISION = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
SIZE = 33_823_087
SHA256 = "a84b1f6e11a65c2d733cb0cc1f014aae3f99051e3f11447dc282faa678eee544"
URL = (
    f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{REVISION}/"
    "models/text_recognition_crnn/text_recognition_CRNN_EN_2021sep.onnx"
)
NATIVE_QUADS = [
    [[63, 54], [278, 54], [278, 90], [63, 90]],
    [[63, 139], [255, 139], [255, 174], [63, 174]],
    [[63, 225], [188, 225], [188, 260], [63, 260]],
]
PARAMETERS = {
    "engine": "classic",
    "batch_size": 1,
    "input_shape": [1, 1, 32, 100],
    "output_names": ["output"],
    "vocabulary": list(string.digits + string.ascii_lowercase),
    "quadrilaterals": NATIVE_QUADS,
}


def prepare(source=None, *, directory=None):
    from opencv_dataflow.model_contracts import (
        configuration,
        export_contract,
        image_preprocessing,
        read_contract,
    )
    from opencv_dataflow.ocr_batch import RecognizeTextBatch

    directory = HERE / ".local/crnn_en" if directory is None else Path(directory)
    block = RecognizeTextBatch(parameters=PARAMETERS)
    try:
        preprocessing = {"input": image_preprocessing(block.parameters)}
        if directory.exists():
            contract = read_contract(directory / "contract.json")
            model = directory / "model.onnx"
            if (
                contract["artifact"] != {"file": "model.onnx", "bytes": SIZE, "sha256": SHA256}
                or contract["configuration"] != configuration(block)
                or contract["preprocessing"] != preprocessing
                or model.stat().st_size != SIZE
            ):
                raise FileExistsError("Different existing CRNN bundle preserved")
            with model.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != SHA256:
                    raise FileExistsError("Different existing CRNN weights preserved")
        else:
            if source is None:
                with urllib.request.urlopen(URL, timeout=60) as response:
                    raw = response.read(SIZE + 1)
            else:
                with Path(source).open("rb") as stream:
                    raw = stream.read(SIZE + 1)
            if len(raw) != SIZE or hashlib.sha256(raw).hexdigest() != SHA256:
                raise ValueError("CRNN weights do not match the pinned artifact")
            with tempfile.TemporaryDirectory(prefix="dataflow-ocr-") as temporary:
                path = Path(temporary) / "model.onnx"
                path.write_bytes(raw)
                block.parameters["model_path"] = str(path)
                export_contract(block, directory, preprocessing=preprocessing)
        return directory
    finally:
        block.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Reuse identical pinned weights")
    print(prepare(parser.parse_args().model))
