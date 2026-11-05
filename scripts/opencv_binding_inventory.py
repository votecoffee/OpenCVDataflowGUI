"""Catalog OpenCV callables against literal project references for the pinned wheel."""

import argparse
import ast
from collections import Counter
from pathlib import Path
from types import ModuleType

import cv2

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "docs/opencv-bindings.tsv"

# Group the long tail by its owning namespace. Top-level bindings are otherwise P3.
P1_PREFIXES = ("cv2.barcode.", "cv2.QRCode", "cv2.Video", "cv2.quality.", "cv2.text.")
P1_CALLS = {
    "compareHist",
    "createBackgroundSubtractorKNN",
    "createBackgroundSubtractorMOG2",
    "drawContours",
    "findContours",
    "fitEllipse",
    "fitLine",
    "floodFill",
    "grabCut",
    "matchShapes",
    "matchTemplate",
    "minAreaRect",
    "minEnclosingCircle",
    "PSNR",
    "pointPolygonTest",
    "watershed",
}
P2_PREFIXES = (
    "cv2.aruco.",
    "cv2.bgsegm.",
    "cv2.ccm.",
    "cv2.detail.",
    "cv2.dnn.",
    "cv2.fisheye.",
    "cv2.img_hash.",
    "cv2.ml.",
    "cv2.omnidir.",
    "cv2.optflow.",
    "cv2.rgbd.",
    "cv2.stereo.",
    "cv2.tracking.",
    "cv2.wechat_qrcode.",
    "cv2.xfeatures2d.",
    "cv2.ximgproc.",
)
P2_CALLS = {
    "AKAZE",
    "BRISK",
    "DISOpticalFlow",
    "KAZE",
    "MSER",
    "SIFT",
    "Stitcher",
    "calibrateHandEye",
    "calibrateRobotWorldHandEye",
    "convexHull",
    "convexityDefects",
    "createGeneralizedHoughBallard",
    "createGeneralizedHoughGuil",
    "dct",
    "dft",
    "fastNlMeansDenoising",
    "fastNlMeansDenoisingColored",
    "findTransformECC",
    "inpaint",
}
INFRASTRUCTURE = {
    "cuda",
    "gapi",
    "ipp",
    "ocl",
    "utils",
    "videoio_registry",
    "getBuildInformation",
    "getNumThreads",
    "getTickCount",
    "setNumThreads",
}
PYTHON_HELPERS = {"config", "config_3", "data", "load_config_py3", "misc", "typing", "version"}


def priority(name):
    suffix = name.removeprefix("cv2.")
    root = suffix.split(".")[0]
    operation = root.removesuffix("_create")
    if root in INFRASTRUCTURE or root.startswith(("cuda", "GAPI", "gapi")):
        return "P4"
    if operation in P1_CALLS or name.startswith(P1_PREFIXES):
        return "P1"
    if operation in P2_CALLS or name.startswith(P2_PREFIXES):
        return "P2"
    return "P3"


def source_references():
    references = set()
    method_names = set()
    for file in (ROOT / "src/opencv_dataflow").rglob("*.py"):
        for node in ast.walk(ast.parse(file.read_text(encoding="utf-8-sig"))):
            if not isinstance(node, ast.Attribute):
                continue
            method_names.add(node.attr)
            parts = []
            while isinstance(node, ast.Attribute):
                parts.append(node.attr)
                node = node.value
            if isinstance(node, ast.Name) and node.id == "cv2":
                references.add("cv2." + ".".join(reversed(parts)))
    return references, method_names


def public_callables(module):
    for member in dir(module):
        if member.startswith("_") or member in PYTHON_HELPERS:
            continue
        item = getattr(module, member)
        name = f"{module.__name__}.{member}"
        if isinstance(item, ModuleType) and item.__name__ == name:
            yield from public_callables(item)
        elif callable(item):
            yield name, "class" if isinstance(item, type) else "function", item


def render():
    references, method_names = source_references()
    by_identity = {}
    rows = []
    for name, kind, item in public_callables(cv2):
        by_identity.setdefault(id(item), []).append(name)
        rows.append((name, kind))
        if kind == "class":
            rows.extend(
                (f"{name}.{member}", "method")
                for member in vars(item)
                if not member.startswith("_") and callable(getattr(item, member))
            )
    aliases = {}
    for names in by_identity.values():
        if len(names) < 2:
            continue
        canonical = min(names, key=lambda n: (n not in references, -n.count("."), n))
        aliases.update((name, canonical) for name in names if name != canonical)
    lines = ["priority\tkind\tbinding\tproject_evidence\talias_of"]
    counts = Counter()
    for name, kind in sorted(rows, key=lambda row: (priority(row[0]), row[0].casefold())):
        parent = name.rsplit(".", 1)[0]
        alias = aliases.get(name) or (
            f"{aliases[parent]}.{name.rsplit('.', 1)[1]}"
            if kind == "method" and parent in aliases
            else ""
        )
        if name in references:
            evidence = "literal reference"
        elif alias:
            evidence = "alias"
        elif kind == "method" and name.rsplit(".", 1)[1] in method_names:
            evidence = "method name only"
        else:
            evidence = "no static reference"
        counts[evidence] += 1
        lines.append(f"{priority(name)}\t{kind}\t{name}\t{evidence}\t{alias or '-'}")
    return "\n".join(lines) + "\n", counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content, counts = render()
    if args.check:
        if not CATALOG.exists() or CATALOG.read_text(encoding="utf-8") != content:
            raise SystemExit("OpenCV binding inventory is out of date for this environment")
    else:
        CATALOG.write_text(content, encoding="utf-8")
    print(f"OpenCV {cv2.__version__}: {sum(counts.values())} rows, {dict(counts)}")


if __name__ == "__main__":
    main()
