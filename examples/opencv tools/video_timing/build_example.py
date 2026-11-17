"""Original explicit-time Matroska/JPEG controls, without a separate muxer dependency."""

import hashlib
import json
import struct
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
CLOCKS = {
    "constant": [0, 40, 80, 120, 160, 200, 240, 280],
    "variable": [0, 40, 130, 160, 340, 380, 550, 600],
    "duplicate": [0, 40, 40, 130, 160, 160, 340, 380],
    "backward": [0, 40, 100, 80, 140, 180, 220, 260],
    "zero_time": [0] * 8,
    "wrong_rate": [0, 40, 80, 120, 160, 200, 240, 280],
}
CASES = {
    "constant": {"clip": "constant", "rejected": 0},
    "variable": {"clip": "variable", "rejected": 0},
    "duplicate": {"clip": "duplicate", "rejected": 2},
    "backward": {"clip": "backward", "rejected": 1},
    "zero_time": {"clip": "zero_time", "rejected": 7},
    "wrong_rate": {"clip": "wrong_rate", "rejected": 7, "cadence_policy": "match_nominal_fps"},
    "wrong_reference": {"clip": "variable", "rejected": 6, "reference": "constant"},
    "paced_variable": {"clip": "variable", "rejected": 0, "paced": True},
}


def vint(value):
    for size in range(1, 9):
        if value < (1 << (7 * size)) - 1:
            return ((1 << (7 * size)) | value).to_bytes(size, "big")
    raise ValueError("Fixture element too large")


def element(tag, data):
    return bytes.fromhex(tag) + vint(len(data)) + data


def unsigned(tag, value):
    return element(tag, value.to_bytes(max(1, (value.bit_length() + 7) // 8), "big"))


def jpeg_frames():
    for index in range(8):
        image = np.full((96, 160, 3), 20, np.uint8)
        image[30:54, 18 + 8 * index : 46 + 8 * index] = [150, 220, 250]
        ok, data = cv2.imencode(
            ".jpg",
            image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95,
                cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
                cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
            ],
        )
        if not ok:
            raise ValueError("JPEG unavailable")
        yield data.tobytes()


def matroska(times, frames, default_duration):
    # Fixed, unlaced keyframe-only fixture subset; not a production Matroska writer.
    header = element(
        "1a45dfa3",
        unsigned("4286", 1)
        + unsigned("42f7", 1)
        + unsigned("42f2", 4)
        + unsigned("42f3", 8)
        + element("4282", b"matroska")
        + unsigned("4287", 4)
        + unsigned("4285", 2),
    )
    info = element(
        "1549a966",
        unsigned("2ad7b1", 1000000)
        + element("4489", struct.pack(">d", max(times) + 40))
        + element("4d80", b"Original OpenCVDataflowGUI fixture")
        + element("5741", b"Original OpenCVDataflowGUI fixture"),
    )
    track = unsigned("d7", 1) + unsigned("73c5", 1) + unsigned("83", 1) + element("86", b"V_MJPEG")
    if default_duration is not None:
        track += unsigned("23e383", default_duration)
    track += element("e0", unsigned("b0", 160) + unsigned("ba", 96))
    tracks = element("1654ae6b", element("ae", track))
    blocks = unsigned("e7", 0)
    for stamp, image in zip(times, frames, strict=True):
        blocks += element("a3", b"\x81" + struct.pack(">h", stamp) + b"\x80" + image)
    return header + element("18538067", info + tracks + element("1f43b675", blocks))


def build():
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    frames = list(jpeg_frames())
    hashes = {}
    for name, times in CLOCKS.items():
        duration = None if name == "zero_time" else 100000000 if name == "wrong_rate" else 40000000
        data = matroska(times, frames, duration)
        path = assets / f"{name}.mkv"
        if path.exists() and path.read_bytes() != data:
            raise ValueError("Existing fixture differs; regenerate in a fresh folder")
        if not path.exists():
            path.write_bytes(data)
        hashes[path.name] = hashlib.sha256(data).hexdigest()
    registry = builtin_registry()
    for name, options in CASES.items():
        graph = Graph()
        try:
            source = graph.add(
                registry.create(
                    "input.paced_video_file" if options.get("paced") else "input.video_file",
                    block_id="video",
                    parameters={
                        "path": f"assets/{options['clip']}.mkv",
                        "path_base": "project",
                        "backend": "ffmpeg",
                        **(
                            {"speed": 2.0, "timing_policy": "decoder_only"}
                            if options.get("paced")
                            else {}
                        ),
                    },
                )
            )
            timing = graph.add(
                registry.create(
                    "sequence.video_timing",
                    block_id="timing",
                    parameters={
                        "cadence_policy": options.get("cadence_policy", "allow_variable"),
                        "reference_timestamps_ms": CLOCKS[
                            options.get("reference", options["clip"])
                        ],
                    },
                )
            )
            timing.exposed_outputs.extend(["chart", "result_graphics"])
            if options.get("paced"):
                source.exposed_outputs.append("cycle_start")
                graph.connect("video", "cycle_start", "timing", "reset")
            gray = graph.add(registry.create("cv.gray", block_id="gray"))
            threshold = graph.add(registry.create("cv.threshold", block_id="threshold"))
            blobs = graph.add(
                registry.create("inspect.blobs", block_id="blobs", parameters={"min_area": 40})
            )
            blobs.exposed_outputs.append("result_graphics")
            overlay = graph.add(registry.create("cv.overlay_result_graphics", block_id="overlay"))
            film = graph.add(registry.create("output.filmstrip_image", block_id="filmstrip"))
            for block, position in zip(
                (source, timing, gray, threshold, blobs, overlay, film),
                ([0, 0], [380, 0], [760, 200], [1140, 200], [1520, 200], [1900, 0], [2280, 0]),
                strict=True,
            ):
                block.position = position
            source.node_size = timing.node_size = "large"
            for wire in (
                ("video", "image", "timing", "image"),
                ("video", "info", "timing", "info"),
                ("timing", "image", "gray", "image"),
                ("gray", "image", "threshold", "image"),
                ("threshold", "image", "blobs", "image"),
                ("timing", "image", "overlay", "image"),
                ("timing", "result_graphics", "overlay", "graphics_1"),
                ("blobs", "result_graphics", "overlay", "graphics_2"),
                ("overlay", "image", "filmstrip", "image"),
            ):
                graph.connect(*wire)
            note = graph.add(registry.create("layout.comment", block_id="attribution"))
            note.name, note.position, note.node_size = (
                "Original authored clocks - MIT",
                [0, -500],
                "large",
            )
            note.description = (
                "Original MIT synthetic images, explicit-time Matroska fixtures, generator and "
                "projects by OpenCVDataflowGUI contributors. No customer or third-party images. "
                "Timestamps are independently specified in truth.json and encoded directly into "
                "SimpleBlock headers. Native FFmpeg reads these actual unequal/repeated/backward "
                "times. Wrong rate declares 10 FPS but stores 40 ms intervals. Zero time omits "
                "DefaultDuration and all positions are zero; legacy replay estimates remain "
                "separately visible from raw properties. Default valid-VFR policy accepts unequal "
                "positive intervals. Intentional bad clocks/reference fail without hiding the "
                "current image or repairing its timing. Paced VFR wires Cycle start to Reset. "
                "The dedicated timing chart uses its own pixel grid; selected timing/blob "
                "objects overlay the original image. Matroska format references are in README."
            )
            graph.save(HERE / f"{name}.json")
        finally:
            graph.close()
    (HERE / "truth.json").write_bytes(
        (
            json.dumps(
                {
                    "license": "MIT - original repository fixtures",
                    "image_size": [160, 96],
                    "clocks_ms": CLOCKS,
                    "cases": CASES,
                    "asset_sha256": hashes,
                    "geometry": [
                        {"index": i, "center": [31.5 + 8 * i, 41.5], "area": 672} for i in range(8)
                    ],
                },
                indent=2,
            )
            + "\n"
        ).encode()
    )


if __name__ == "__main__":
    build()
