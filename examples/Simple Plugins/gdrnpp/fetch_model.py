"""Fetch one author-published T-LESS object-6 checkpoint with pinned bytes."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

MODEL_PATH = (
    "gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/6/model_final_wo_optim.pth"
)
MODEL_SHA256 = "b1dc54a1a6572e4db560f28a05a0821ed4dc05b18c07052f651f890753d8e606"
MODEL_BYTES = 410204745
SOURCE_COMMIT = "990003d2c13d4a06fec98e27d980595a9831ccc7"


def fetch(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new model directory")
    output.mkdir(parents=True)
    url = (
        "https://modelscope.cn/api/v1/models/wangg12/GDRNPP/repo?Revision=master&FilePath="
        + MODEL_PATH
    )
    destination = output / "model_final_wo_optim.pth"
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(url, timeout=60) as response, destination.open("xb") as stream:
        while block := response.read(1024 * 1024):
            size += len(block)
            if size > MODEL_BYTES:
                raise ValueError("Published model exceeds its pinned byte size")
            stream.write(block)
            digest.update(block)
    if size != MODEL_BYTES or digest.hexdigest() != MODEL_SHA256:
        raise ValueError("Model checksum/size mismatch; incomplete output is not a usable model")
    record = {
        "schema": "simple_gdrnpp_checkpoint/v1",
        "publisher": "wangg12/GDRNPP (author-linked ModelScope repository)",
        "publisher_repository_license": "Apache License 2.0",
        "license_metadata_url": "https://modelscope.cn/api/v1/models/wangg12/GDRNPP",
        "source_commit": SOURCE_COMMIT,
        "config": "configs/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/6.py",
        "dataset": "T-LESS",
        "object_id": 6,
        "training": "published PBR-only single-object checkpoint",
        "model_path": destination.name,
        "download_url": url,
        "bytes": size,
        "sha256": digest.hexdigest(),
        "revision_note": "Mutable branch; exact selected asset pinned by size and SHA-256",
    }
    (output / "checkpoint.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("new_output", type=Path)
    print(json.dumps(fetch(parser.parse_args().new_output), indent=2))


if __name__ == "__main__":
    main()
