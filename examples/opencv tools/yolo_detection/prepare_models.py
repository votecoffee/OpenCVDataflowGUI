"""Explicit pinned YOLOX preparation, with a documented derived output-layout export."""

import argparse
import hashlib
import tempfile
import urllib.request
from pathlib import Path

REVISION = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
SHA256 = "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
SIZE = 35_858_002
DERIVED_SHA256 = "b02b06f1a60e2bccdab23400b2c52d62ced6a02baf856ac53aef42ddb92b0d5c"
DERIVED_SIZE = 35_960_492
URL = (
    f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{REVISION}/models/"
    "object_detection_yolox/object_detection_yolox_2022nov.onnx"
)
HERE = Path(__file__).resolve().parent
BASE_PARAMETERS = {
    "width": 640,
    "height": 640,
    "color_format": "rgb",
    "resize_mode": "letterbox",
    "padding_value": 114,
    "scale": 1,
    "mean": [0.0, 0.0, 0.0],
    "std": [1.0, 1.0, 1.0],
    "input_name": "images",
    "backend": "opencv",
    "target": "cpu",
    "receipt_mode": "detailed",
}
PARAMETERS = {
    "raw": {**BASE_PARAMETERS, "engine": "classic", "output_names": ["output"]},
    "decoded": {
        **BASE_PARAMETERS,
        "engine": "new",
        "output_names": ["decoded_rows", "decoded_channels"],
        "max_output_elements": 1_500_000,
    },
}


def decoded_export(raw):
    """Append grid/size decode, objectness multiplication and transpose; keep learned weights."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    if len(raw) != SIZE or hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("Source weights do not match the pinned YOLOX size/digest")
    model = onnx.load_model_from_string(raw)
    declaration = [
        (item.name, [d.dim_value for d in item.type.tensor_type.shape.dim])
        for item in [*model.graph.input, *model.graph.output]
    ]
    if declaration != [("images", [1, 3, 640, 640]), ("output", [1, 8400, 85])]:
        raise ValueError("Pinned input/output declaration changed")
    prefix = "dataflow_yolox_"
    if any(
        name.startswith(prefix) for node in model.graph.node for name in (*node.input, *node.output)
    ):
        raise ValueError("Derived output names already exist")
    grid, stride_values = [], []
    for stride in (8, 16, 32):
        for y in range(640 // stride):
            for x in range(640 // stride):
                grid.append((x, y))
                stride_values.append((stride,))
    initial = {
        "grid": np.asarray([grid], dtype="<f4"),
        "stride": np.asarray([stride_values], dtype="<f4"),
        "axis": np.array([2], dtype="<i8"),
    }
    for name, start, end in (("xy", 0, 2), ("wh", 2, 4), ("obj", 4, 5), ("class", 5, 85)):
        initial[name + "_start"] = np.array([start], dtype="<i8")
        initial[name + "_end"] = np.array([end], dtype="<i8")
        model.graph.node.append(
            helper.make_node(
                "Slice",
                ["output", prefix + name + "_start", prefix + name + "_end", prefix + "axis"],
                [prefix + name],
            )
        )
    model.graph.initializer.extend(
        numpy_helper.from_array(value, prefix + name) for name, value in initial.items()
    )
    for op, inputs, output in (
        ("Add", [prefix + "xy", prefix + "grid"], prefix + "offset_xy"),
        ("Mul", [prefix + "offset_xy", prefix + "stride"], prefix + "centers"),
        ("Exp", [prefix + "wh"], prefix + "exp_wh"),
        ("Mul", [prefix + "exp_wh", prefix + "stride"], prefix + "sizes"),
        ("Mul", [prefix + "obj", prefix + "class"], prefix + "scores"),
    ):
        model.graph.node.append(helper.make_node(op, inputs, [output]))
    model.graph.node.extend(
        [
            helper.make_node(
                "Concat",
                [prefix + "centers", prefix + "sizes", prefix + "scores"],
                ["decoded_rows"],
                axis=2,
            ),
            helper.make_node("Transpose", ["decoded_rows"], ["decoded_channels"], perm=[0, 2, 1]),
        ]
    )
    model.graph.output.extend(
        [
            helper.make_tensor_value_info("decoded_rows", TensorProto.FLOAT, [1, 8400, 84]),
            helper.make_tensor_value_info("decoded_channels", TensorProto.FLOAT, [1, 84, 8400]),
        ]
    )
    model.doc_string += (
        "\nModified by OpenCVDataflowGUI contributors: appended YOLOX grid/Exp size decode, "
        "objectness-times-class scores and row/channel transpose. Learned weights unchanged. "
        "This is a derived YOLOX export, not a YOLOv8 model."
    )
    onnx.checker.check_model(model)
    return model.SerializeToString(deterministic=True)


def prepare(source=None, *, root=None):
    from opencv_dataflow.dnn_blocks import DNNInference
    from opencv_dataflow.model_contracts import (
        configuration,
        export_contract,
        image_preprocessing,
        read_contract,
    )

    root = HERE / ".local" if root is None else Path(root)
    if source is None and (root / "yolox_raw/model.onnx").is_file():
        source = root / "yolox_raw/model.onnx"
    if source is None:
        with urllib.request.urlopen(URL, timeout=45) as response:
            raw = response.read(SIZE + 1)
    else:
        with Path(source).open("rb") as stream:
            raw = stream.read(SIZE + 1)
    if len(raw) != SIZE or hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("Source weights do not match the pinned YOLOX size/digest")
    derived = decoded_export(raw)
    if len(derived) != DERIVED_SIZE or hashlib.sha256(derived).hexdigest() != DERIVED_SHA256:
        raise ValueError("Derived export does not match its pinned size/digest")
    # Check every existing bundle before publishing any missing bundle.
    pending = []
    for kind, contents in (("raw", raw), ("decoded", derived)):
        directory = root / ("yolox_" + kind)
        tool = DNNInference(parameters=PARAMETERS[kind])
        preprocessing = {"images": image_preprocessing(tool.parameters)}
        identity = {
            "file": "model.onnx",
            "bytes": len(contents),
            "sha256": hashlib.sha256(contents).hexdigest(),
        }
        if directory.exists():
            contract = read_contract(directory / "contract.json")
            if (
                contract["artifact"] != identity
                or contract["configuration"] != configuration(tool)
                or contract["preprocessing"] != preprocessing
                or (directory / "model.onnx").read_bytes() != contents
            ):
                raise FileExistsError("Different existing YOLO model bundle preserved")
        else:
            pending.append((tool, directory, preprocessing, contents))
    for tool, directory, preprocessing, contents in pending:
        with tempfile.TemporaryDirectory(prefix="dataflow-yolo-") as temporary:
            path = Path(temporary) / "model.onnx"
            path.write_bytes(contents)
            tool.parameters["model_path"] = str(path)
            export_contract(tool, directory, preprocessing=preprocessing)
    print(f"Verified pinned raw and derived YOLOX model bundles (Apache 2.0): {root}")
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, help="Reuse existing source weights after verification"
    )
    prepare(parser.parse_args().model)
