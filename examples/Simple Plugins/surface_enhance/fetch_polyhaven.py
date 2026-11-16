"""Fetch a pinned CC0 material-map control. Powered by Poly Haven."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

ASSET = "anti_slip_concrete"
UA = {"User-Agent": "OpenCVSimpleTools-SurfaceEnhance/0.1"}
FILES = {
    "diffuse.exr": ("Diffuse", "exr", 1974257, "43aec16abfd54c39f1d970b94477e10c"),
    "normal_gl.png": ("nor_gl", "png", 5720251, "31a303a343c8858aca2a455ca17631c1"),
}


def get(url, bound):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as response:
        raw = response.read(bound + 1)
    if len(raw) > bound:
        raise ValueError("Public asset exceeded its bounded request")
    return raw


def fetch(output):
    output.mkdir(parents=True, exist_ok=False)
    print("Powered by Poly Haven | Anti Slip Concrete | Charlotte Baglioni | CC0", flush=True)
    info = json.loads(get(f"https://api.polyhaven.com/info/{ASSET}", 2_000_000))
    files = json.loads(get(f"https://api.polyhaven.com/files/{ASSET}", 2_000_000))
    if info["files_hash"] != "6cb6548b7d539a2af92acae83c8943a7140ed444":
        raise ValueError("Material release changed; review and pin new source data")
    retained = {}
    for name, (kind, encoding, length, md5) in FILES.items():
        entry = files[kind]["1k"][encoding]
        if entry["size"] != length or entry["md5"] != md5:
            raise ValueError("Pinned material file identity changed")
        raw = get(entry["url"], length)
        if len(raw) != length or hashlib.md5(raw).hexdigest() != md5:
            raise ValueError("Publisher file checksum/length differs")
        (output / name).write_bytes(raw)
        retained[name] = dict(
            url=entry["url"],
            bytes=length,
            publisher_md5=md5,
            sha256=hashlib.sha256(raw).hexdigest(),
        )
    provenance = dict(
        asset_id=ASSET,
        title=info["name"],
        authors=info["authors"],
        asset_url="https://polyhaven.com/a/" + ASSET,
        license="CC0-1.0",
        license_url="https://polyhaven.com/license",
        source_files_hash=info["files_hash"],
        checked="2026-11-16",
        files=retained,
        api_credit="Powered by Poly Haven",
        api_terms="https://github.com/Poly-Haven/Public-API/blob/master/ToS.md",
        use=(
            "Material-map inputs for rendered diffuse controls, "
            "not acquired multi-light camera data"
        ),
    )
    (output / "provenance.json").write_bytes((json.dumps(provenance, indent=2) + "\n").encode())
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(fetch(parser.parse_args().output)))
