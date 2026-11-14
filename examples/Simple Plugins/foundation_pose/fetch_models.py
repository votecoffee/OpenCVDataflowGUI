"""Intentionally fetch two pinned NVIDIA ONNX models; imports/inference never auto-download."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from opencv_simple_foundation_pose.runtime import ASSETS, REVISION


def fetch(directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for name, (size, digest) in ASSETS.items():
        path = root / name
        url = f"https://huggingface.co/nvidia/foundationpose/resolve/{REVISION}/{name}"
        if not path.exists():
            temporary = path.with_suffix(".download")
            # Preserve prior/interrupted files for review; use a fresh directory after failures.
            with urllib.request.urlopen(url, timeout=60) as source, temporary.open("xb") as target:
                total = 0
                while block := source.read(1024 * 1024):
                    total += len(block)
                    if total > size:
                        raise ValueError("Oversized model response")
                    target.write(block)
            if total != size or hashlib.sha256(temporary.read_bytes()).hexdigest() != digest:
                raise ValueError("Downloaded model differs from publisher size/SHA-256")
            temporary.rename(path)
        if path.stat().st_size != size or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Existing model differs; preserve and inspect {path}")
        records.append(
            {
                "name": name,
                "bytes": size,
                "sha256": digest,
                "source": url,
                "actual_bytes_verified": True,
            }
        )
        print(name, "verified", flush=True)
    (root / "provenance.json").write_text(
        json.dumps(
            {
                "revision": REVISION,
                "files": records,
                "license": "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    fetch(parser.parse_args().directory)
