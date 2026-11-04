"""Download the explicitly requested Lena test asset from a pinned OpenCV commit.

The third-party image is ignored by Git and is not covered by this project's MIT
license. Download is explicit, hash-checked, and never replaces a different file.
"""

import hashlib
import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/opencv/opencv/"
    "1007c06d44a118566bbc0200577942423b38911d/samples/data/lena.jpg"
)
SHA256 = "7de7ed51a1594fff247f4cae2301eceacf5313d6011e37b4a4c8733f7bb72c07"


def prepare(destination):
    destination = Path(destination)
    if destination.exists():
        data = destination.read_bytes()
    else:
        with urllib.request.urlopen(URL, timeout=30) as response:
            data = response.read(1_000_001)
        if len(data) > 1_000_000 or hashlib.sha256(data).hexdigest() != SHA256:
            raise ValueError("Downloaded Lena asset failed its expected size/hash check")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(data)
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError(f"Existing image differs from the pinned asset; preserved: {destination}")
    return destination


if __name__ == "__main__":
    path = prepare(Path(__file__).resolve().parents[1] / "examples" / "assets" / "lena.jpg")
    print(f"Lena ready: {path}\nSHA-256: {SHA256}")
