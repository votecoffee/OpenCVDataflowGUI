"""Explicit pinned EfficientSAM download and sealed settings bundle; no runtime downloads."""

import argparse
import hashlib
import tempfile
import urllib.request
from pathlib import Path

REVISION = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
SHA256 = "4eb496e0a7259d435b49b66faf1754aa45a5c382a34558ddda9a8c6fe5915d77"
SIZE = 48_312_857
URL = (
    f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{REVISION}/models/"
    "image_segmentation_efficientsam/image_segmentation_efficientsam_ti_2025april.onnx"
)
HERE = Path(__file__).resolve().parent
PARAMETERS = {"engine": "new", "backend": "opencv", "target": "cpu", "receipt_mode": "summary"}


def prepare(source=None, *, directory=None):
    from opencv_dataflow.model_contracts import configuration, export_contract, read_contract
    from opencv_dataflow.prompt_segmentation import EfficientSAMInference, preprocessing_contract

    directory = HERE / ".local/efficient_sam" if directory is None else Path(directory)
    tool = EfficientSAMInference(parameters=PARAMETERS)
    preprocessing = preprocessing_contract()
    if directory.exists():
        contract = read_contract(directory / "contract.json")
        model = directory / "model.onnx"
        if (
            contract["artifact"] != {"file": "model.onnx", "bytes": SIZE, "sha256": SHA256}
            or contract["configuration"] != configuration(tool)
            or contract["preprocessing"] != preprocessing
            or model.stat().st_size != SIZE
        ):
            raise FileExistsError("Different existing EfficientSAM bundle preserved")
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
            raise ValueError("EfficientSAM weights do not match the pinned size/digest")
        with tempfile.TemporaryDirectory(prefix="dataflow-efficient-sam-") as temporary:
            path = Path(temporary) / "model.onnx"
            path.write_bytes(raw)
            tool.parameters["model_path"] = str(path)
            export_contract(tool, directory, preprocessing=preprocessing)
    print(f"Verified pinned EfficientSAM (Apache 2.0): {directory}")
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, help="Reuse identical weights after exact verification"
    )
    prepare(parser.parse_args().model)
