"""Download only the pinned CC BY 4.0 VisA members used by this example."""

import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath


def download(target):
    manifest = json.loads(Path(__file__).with_name("source_manifest.json").read_bytes())
    target = Path(target)
    target.mkdir(parents=True, exist_ok=False)

    def fetch(item):
        member, info = item
        name = PurePosixPath(member)
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("Invalid dataset member path.")
        with urllib.request.urlopen(info["url"], timeout=30) as response:
            raw = response.read()
        if hashlib.sha256(raw).hexdigest() != info["sha256"]:
            raise ValueError(f"Source checksum changed: {member}")
        path = target / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(fetch, manifest["files"].items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New source cache directory")
    download(parser.parse_args().output)
