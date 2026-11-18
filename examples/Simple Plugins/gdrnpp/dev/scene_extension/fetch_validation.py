"""Fetch a frozen metadata-selected T-LESS subset through bounded verified ZIP ranges.

No model is loaded. Selection uses publisher IDs/visibility, never predicted results.
"""

import argparse
import hashlib
import io
import json
import struct
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

REVISION = "5fd309a04476a842d93abfb584fba9ee7caecdf1"
ARCHIVE_BYTES = 825276992
ARCHIVE_SHA256 = "1a18f6bbfb5ac4ced8529f7a35225adfed88c0f62ef38067933e2b541ef1d00b"
URL = (
    "https://huggingface.co/datasets/bop-benchmark/tless/resolve/"
    + REVISION
    + "/tless_test_primesense_bop19.zip"
)


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )


def fetch_range(start, size):
    if not 0 <= start < ARCHIVE_BYTES or not 0 < size <= 32_000_000:
        raise ValueError("Bound each range to the pinned archive and 32 MB")
    end = min(start + size, ARCHIVE_BYTES) - 1
    request = urllib.request.Request(
        URL + f"?selected_range={start}_{end}", headers={"Range": f"bytes={start}-{end}"}
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        expected = f"bytes {start}-{end}/{ARCHIVE_BYTES}"
        if response.status != 206 or response.headers.get("Content-Range") != expected:
            raise ValueError("Server did not honor the exact bounded archive range")
        data = response.read(end - start + 2)
    if len(data) != end - start + 1:
        raise ValueError("Truncated/oversized archive range")
    return data


class RemoteIndex(io.RawIOBase):
    def __init__(self):
        self.position = 0

    def seek(self, offset, whence=0):
        self.position = offset + (
            0 if whence == 0 else self.position if whence == 1 else ARCHIVE_BYTES
        )
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        size = (
            min(size, ARCHIVE_BYTES - self.position) if size >= 0 else ARCHIVE_BYTES - self.position
        )
        if size <= 0:
            return b""
        data = fetch_range(self.position, size)
        self.position += len(data)
        return data


def member(info):
    if info.flag_bits & 1 or not 0 <= info.file_size <= 32_000_000:
        raise ValueError("Reject encrypted or oversized dataset members")
    # One bounded request includes the local header, name/extra and compressed data.
    raw = fetch_range(info.header_offset, info.compress_size + 30 + 2 * 65535)
    header = struct.unpack("<4s5H3I2H", raw[:30])
    if header[0] != b"PK\x03\x04" or header[3] != info.compress_type:
        raise ValueError("ZIP local header differs from central directory")
    name_size, extra_size = header[-2:]
    name = raw[30 : 30 + name_size].decode("utf-8" if info.flag_bits & 0x800 else "cp437")
    if name != info.filename:
        raise ValueError("ZIP local member name differs")
    payload = raw[30 + name_size + extra_size : 30 + name_size + extra_size + info.compress_size]
    if info.compress_type == ZIP_STORED:
        data = payload
    elif info.compress_type == ZIP_DEFLATED:
        decoder = zlib.decompressobj(-15)
        data = decoder.decompress(payload, info.file_size + 1)
        if not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:
            raise ValueError("Truncated/oversized compressed member")
    else:
        raise ValueError("Unsupported dataset compression")
    if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
        raise ValueError("Dataset member size/CRC differs")
    return info.filename, data


def choose_cases(metadata, names):
    """Select 10 evenly spaced eligible frames plus least visible/most foreshortened.

    Negatives: first eligible frame of each target-absent scene; lowest object ID
    supplies the rectangle. Min 100 visible pixels / 10 pixels per bbox side.
    """
    positives, negatives = [], []
    for scene in range(1, 21):
        if scene in (2, 3):
            continue  # Original development scenes stay out of this extension.
        base = f"test_primesense/{scene:06d}"
        gt = metadata[base + "/scene_gt.json"]
        info = metadata[base + "/scene_gt_info.json"]
        available = sorted(
            int(p.rsplit("/", 1)[1][:-4])
            for p in names
            if p.startswith(base + "/rgb/") and p.endswith(".png")
        )
        eligible = []
        target_present = any(e["obj_id"] == 6 for values in gt.values() for e in values)
        for number in available:
            options = []
            for i, entry in enumerate(gt[str(number)]):
                label = info[str(number)][i]
                box = label["bbox_visib"]
                if label["px_count_visib"] >= 100 and min(box[2:]) >= 10:
                    if (entry["obj_id"] == 6) == target_present:
                        options.append((entry["obj_id"], i))
            if options:
                object_id, index = min(options)
                eligible.append(
                    dict(
                        scene=scene,
                        frame=number,
                        instance_index=index,
                        selected_object_id=object_id,
                        target_present=target_present,
                        bbox_xywh=info[str(number)][index]["bbox_visib"],
                        visibility=info[str(number)][index]["visib_fract"],
                    )
                )
        if not eligible:
            continue
        if target_present:
            count = min(10, len(eligible))
            indices = {round(i * (len(eligible) - 1) / max(1, count - 1)) for i in range(count)}
            indices.add(min(range(len(eligible)), key=lambda i: eligible[i]["visibility"]))
            indices.add(
                min(
                    range(len(eligible)),
                    key=lambda i: (
                        min(eligible[i]["bbox_xywh"][2:]) / max(eligible[i]["bbox_xywh"][2:])
                    ),
                )
            )
            positives.extend(eligible[i] for i in sorted(indices))
        else:
            negatives.append(eligible[0])
    return positives + negatives


def fetch(output, cache=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a fresh validation source directory")
    output.mkdir(parents=True)
    with ZipFile(RemoteIndex()) as archive:
        index = {v.filename: v for v in archive.infolist()}
    metadata_names = [
        f"test_primesense/{scene:06d}/{name}.json"
        for scene in range(1, 21)
        if scene not in (2, 3)
        for name in ("scene_gt", "scene_gt_info", "scene_camera")
    ]
    hashes = {}
    cache = Path(cache) if cache else None
    cached = json.loads((cache / "provenance.json").read_text()) if cache else {}
    if cache and cached.get("revision") != REVISION:
        raise ValueError("Cached source must use this pinned dataset revision")

    def read(info):
        name = info.filename
        if cache and name in cached["sha256"]:
            data = (cache / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != cached["sha256"][name]:
                raise ValueError("Cached member checksum differs: " + name)
            if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
                raise ValueError("Cached member differs from remote ZIP directory")
            return name, data
        return member(info)

    def retain(items):
        for name, data in items:
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()

    with ThreadPoolExecutor(max_workers=6) as pool:
        retain(pool.map(read, [index[name] for name in metadata_names]))
    metadata = {name: json.loads((output / name).read_bytes()) for name in metadata_names}
    cases = choose_cases(metadata, index)
    selection = dict(
        rule=choose_cases.__doc__,
        cases=cases,
        settings_selected_before_inference=True,
        excluded_development_scenes=[2, 3],
        checkpoint_object_id=6,
        scope="New scene-level development validation; not untouched production qualification",
    )
    write_json(output / "selection.json", selection)
    hashes["selection.json"] = hashlib.sha256((output / "selection.json").read_bytes()).hexdigest()
    members = sorted(
        {
            f"test_primesense/{row['scene']:06d}/{kind}/{row['frame']:06d}.png"
            for row in cases
            for kind in ("rgb", "depth")
        }
    )
    with ThreadPoolExecutor(max_workers=6) as pool:
        retain(pool.map(read, [index[name] for name in members]))
    write_json(
        output / "provenance.json",
        dict(
            dataset="T-LESS",
            license="CC-BY-4.0",
            source="https://cmp.felk.cvut.cz/t-less/",
            mirror="https://huggingface.co/datasets/bop-benchmark/tless",
            revision=REVISION,
            test_archive_bytes=ARCHIVE_BYTES,
            upstream_archive_sha256=ARCHIVE_SHA256,
            verification=(
                "Selected member sizes/ZIP CRC and local SHA-256; "
                "full ZIP not downloaded or rehashed"
            ),
            sha256=hashes,
        ),
    )
    print(
        json.dumps(
            dict(
                cases=len(cases),
                positives=sum(c["target_present"] for c in cases),
                negative_controls=sum(not c["target_present"] for c in cases),
                scenes=sorted({c["scene"] for c in cases}),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("new_source", type=Path)
    parser.add_argument("--verified-cache", type=Path)
    args = parser.parse_args()
    fetch(args.new_source, args.verified_cache)
