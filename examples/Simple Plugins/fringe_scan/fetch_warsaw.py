"""Fetch only pinned CC BY4 scan members through bounded HTTP ZIP ranges."""

import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

AGENT = {"User-Agent": "SimpleFringeScanExamples/0.1"}


class RangeFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.position, self.transferred = url, size, 0, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (
            offset
            if whence == 0
            else (self.position + offset if whence == 1 else self.size + offset)
        )
        if not 0 <= self.position <= self.size:
            raise ValueError("Out of archive bounds")
        return self.position

    def read(self, count=-1):
        count = min(count if count >= 0 else self.size, self.size - self.position)
        if count == 0:
            return b""
        if count > 8 * 1024 * 1024 or self.transferred + count > 48 * 1024 * 1024:
            raise ValueError("Bounded archive-member transfer budget exceeded")
        start, end = self.position, self.position + count - 1
        request = Request(self.url, headers={**AGENT, "Range": f"bytes={start}-{end}"})
        with urlopen(request, timeout=30) as response:
            if response.status != 206 or response.headers.get("Content-Range") != (
                f"bytes {start}-{end}/{self.size}"
            ):
                raise ValueError("Server does not support exact bounded ranges")
            raw = response.read(count + 1)
        if len(raw) != count:
            raise ValueError("Archive range length differs")
        self.position += count
        self.transferred += count
        return raw


def fetch(output):
    pinned = json.loads((Path(__file__).parent / "source_members.json").read_bytes())
    with urlopen(
        Request("https://zenodo.org/api/records/17826191", headers=AGENT), timeout=30
    ) as stream:
        metadata = stream.read(2 * 1024 * 1024 + 1)
    if len(metadata) > 2 * 1024 * 1024:
        raise ValueError("Dataset metadata exceeds bound")
    metadata = json.loads(metadata)
    if metadata["metadata"]["license"]["id"] != "cc-by-4.0":
        raise ValueError("Dataset license changed; review before downloading")
    file = next(row for row in metadata["files"] if row["key"] == pinned["archive_name"])
    if file["size"] != pinned["archive_size"] or file["checksum"] != pinned["publisher_checksum"]:
        raise ValueError("Publisher archive identity differs from the pinned release")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    reader = RangeFile(file["links"]["self"], file["size"])
    with ZipFile(reader) as archive:
        for row in pinned["members"]:
            info = archive.getinfo(row["member"])
            if info.file_size != row["size"] or info.file_size > 6 * 1024 * 1024:
                raise ValueError("Pinned member size differs")
            body = archive.read(info)  # CRC verified by ZipFile; SHA pinned independently below.
            if hashlib.sha256(body).hexdigest() != row["sha256"]:
                raise ValueError("Pinned member hash differs")
            target = (output / row["target"]).resolve()
            if not target.is_relative_to(output):
                raise ValueError("Member target is outside the example output")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(body)
    for source, relative in (
        ("raw_provenance", "warsaw_raw/provenance.json"),
        ("pattern_provenance", "warsaw_patterns/provenance.json"),
    ):
        (output / relative).write_bytes((json.dumps(pinned[source], indent=2) + "\n").encode())
    return dict(
        members=len(pinned["members"]),
        transferred_bytes=reader.transferred,
        whole_archive_downloaded=False,
        whole_archive_md5_verified=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(fetch(parser.parse_args().output)))
