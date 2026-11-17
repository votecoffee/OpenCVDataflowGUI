"""Explicit preparation of local ML examples. No downloads occur during graph execution."""

import argparse
import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path

LIGHTGLUE_URL = "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/sift_lightglue.pth"
SUPERPOINT_MATCHER_URL = (
    "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth"
)
SUPERPOINT_MATCHER_SHA256 = "6ff7040d0a497fc6639337946d7538dae07428c18f77a067a0b5a960e7cc551a"


def tiny_onnx(path):
    """A deterministic RGB average network, generated locally without pretrained weights."""
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError as exc:
        raise RuntimeError("Install the [ml-test] extra to generate the tiny ONNX example") from exc
    graph = helper.make_graph(
        [
            helper.make_node("GlobalAveragePool", ["image"], ["pooled"]),
            helper.make_node("Flatten", ["pooled"], ["scores"], axis=1),
        ],
        "rgb_average",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 32, 32])],
        [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    onnx.checker.check_model(model)
    data = model.SerializeToString()
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"Existing different model preserved: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(data)
    return path


def prepare_lightglue(path, *, url=LIGHTGLUE_URL, expected_sha256=None):
    if path.exists() or path.with_suffix(".receipt.json").exists():
        raise FileExistsError(f"Existing weights preserved: {path}")
    with tempfile.TemporaryFile() as temporary:
        with urllib.request.urlopen(url, timeout=60) as response:
            while chunk := response.read(1024 * 1024):
                temporary.write(chunk)
        temporary.seek(0)
        digest = hashlib.file_digest(temporary, "sha256").hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("Downloaded matcher SHA-256 differs from the pinned release")
        temporary.seek(0)
        with path.open("xb") as output:
            while chunk := temporary.read(1024 * 1024):
                output.write(chunk)
    receipt = {"source": url, "sha256": digest, "bytes": path.stat().st_size}
    with path.with_suffix(".receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(".local/models"))
    parser.add_argument(
        "--lightglue", action="store_true", help="Download official SIFT matcher weights"
    )
    parser.add_argument(
        "--superpoint-matcher",
        action="store_true",
        help="Download Apache-2.0 LightGlue matcher only, without the SuperPoint detector",
    )
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    print(tiny_onnx(args.directory / "rgb_average.onnx"))
    if args.lightglue:
        prepare_lightglue(args.directory / "sift_lightglue.pth")
    if args.superpoint_matcher:
        prepare_lightglue(
            args.directory / "superpoint_lightglue.pth",
            url=SUPERPOINT_MATCHER_URL,
            expected_sha256=SUPERPOINT_MATCHER_SHA256,
        )


if __name__ == "__main__":
    main()
