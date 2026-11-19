"""Prepare the independently trained SuperPoint v6 as a dense-head ONNX model."""

import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen

REVISION = "1411bbd68c50163555d39c1b26e9e046ebd48f27"
SOURCE = f"https://raw.githubusercontent.com/rpautrat/SuperPoint/{REVISION}"
WEIGHTS_URL = f"{SOURCE}/weights/superpoint_v6_from_tf.pth"
WEIGHTS_SHA256 = "cd5d19a5061848e248c17728878ea166b66512076d43c77dbcf27f4a88a56084"
LICENSE_URL = f"{SOURCE}/LICENSE.txt"
MODEL_NAME = "superpoint-v6-dense.onnx"


def checkpoint_bytes(path):
    if path is not None:
        with path.open("rb") as stream:
            raw = stream.read(8_000_001)
    else:
        with urlopen(WEIGHTS_URL, timeout=60) as response:
            raw = response.read(8_000_001)
    if hashlib.sha256(raw).hexdigest() != WEIGHTS_SHA256:
        raise ValueError("The checkpoint does not match the pinned SuperPoint v6 SHA-256")
    return raw


def export_onnx(weights_bytes):
    import onnx
    import torch
    from onnx import TensorProto, helper, numpy_helper

    weights = torch.load(io.BytesIO(weights_bytes), map_location="cpu", weights_only=True)
    initializers, nodes = [], []

    def parameters(prefix, names):
        keys = [f"{prefix}.{name}" for name in names]
        initializers.extend(numpy_helper.from_array(weights[k].numpy(), k) for k in keys)
        return keys

    def layer(value, prefix, size, *, activate=True, output=None):
        conv = f"{prefix}.conv_out"
        padding = size // 2
        nodes.append(
            helper.make_node(
                "Conv",
                [value, *parameters(f"{prefix}.conv", ("weight", "bias"))],
                [conv],
                kernel_shape=[size, size],
                pads=[padding] * 4,
            )
        )
        value = conv
        if activate:
            value = f"{prefix}.relu"
            nodes.append(helper.make_node("Relu", [conv], [value]))
        normalized = output or f"{prefix}.bn_out"
        affine = parameters(f"{prefix}.bn", ("weight", "bias", "running_mean", "running_var"))
        nodes.append(
            helper.make_node("BatchNormalization", [value, *affine], [normalized], epsilon=0.001)
        )
        return normalized

    features = "image"
    for stage in range(4):
        features = layer(features, f"backbone.{stage}.0", 3)
        features = layer(features, f"backbone.{stage}.1", 3)
        if stage < 3:
            pooled = f"backbone.{stage}.pool"
            nodes.append(
                helper.make_node(
                    "MaxPool", [features], [pooled], kernel_shape=[2, 2], strides=[2, 2]
                )
            )
            features = pooled
    for head, name in (("detector", "logits"), ("descriptor", "descriptors")):
        value = layer(features, f"{head}.0", 3)
        layer(value, f"{head}.1", 1, activate=False, output=name)

    tensor = helper.make_tensor_value_info
    graph = helper.make_graph(
        nodes,
        "superpoint_v6_dense",
        [tensor("image", TensorProto.FLOAT, [1, 1, "height", "width"])],
        [
            tensor("logits", TensorProto.FLOAT, [1, 65, "grid_h", "grid_w"]),
            tensor("descriptors", TensorProto.FLOAT, [1, 256, "grid_h", "grid_w"]),
        ],
        initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=8,
        producer_name="OpenCVDataflowGUI",
    )
    helper.set_model_props(
        model,
        {
            "source": WEIGHTS_URL,
            "checkpoint_sha256": WEIGHTS_SHA256,
            "license": "MIT",
            "license_url": LICENSE_URL,
        },
    )
    onnx.checker.check_model(model)
    return model.SerializeToString(deterministic=True)


def write_artifact(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"Existing different artifact preserved: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(".local/models"))
    parser.add_argument(
        "--checkpoint", type=Path, help="Use an already downloaded pinned checkpoint"
    )
    args = parser.parse_args()
    model = export_onnx(checkpoint_bytes(args.checkpoint))
    license_text = (
        Path(__file__).resolve().parents[1] / "docs/third-party/superpoint-v6-MIT.txt"
    ).read_bytes()
    receipt = {
        "source": WEIGHTS_URL,
        "checkpoint_sha256": WEIGHTS_SHA256,
        "license": "MIT",
        "license_url": LICENSE_URL,
        "onnx_sha256": hashlib.sha256(model).hexdigest(),
        "onnx_bytes": len(model),
    }
    args.directory.mkdir(parents=True, exist_ok=True)
    write_artifact(args.directory / MODEL_NAME, model)
    write_artifact(args.directory / "superpoint-v6-dense.LICENSE.txt", license_text)
    write_artifact(
        args.directory / "superpoint-v6-dense.receipt.json",
        (json.dumps(receipt, indent=2) + "\n").encode(),
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
