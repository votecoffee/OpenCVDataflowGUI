"""Original MIT network-stream fixtures and managed-service inspection projects."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
CASES = {"live_stream": "/stream", "reconnect_reset": "/drop"}


def images():
    for index in range(8):
        image = np.full((96, 160, 3), 20, np.uint8)
        image[30:54, 18 + 8 * index : 46 + 8 * index] = [150, 220, 250]
        ok, encoded = cv2.imencode(
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
            raise ValueError("JPEG encoder unavailable")
        yield encoded.tobytes()


def build():
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    hashes = {}
    for index, data in enumerate(images()):
        path = assets / f"frame_{index:02d}.jpg"
        if path.exists() and path.read_bytes() != data:
            raise ValueError("Existing fixture differs; use a fresh output folder")
        if not path.exists():
            path.write_bytes(data)
        hashes[path.name] = hashlib.sha256(data).hexdigest()
    registry = builtin_registry()
    for name, route in CASES.items():
        graph = Graph()
        try:
            source = graph.add(
                registry.create(
                    "input.network_video",
                    block_id="video",
                    parameters={
                        "url": f"http://127.0.0.1:18764{route}",
                        "source_name": "original-loopback",
                        "open_timeout_ms": 2000,
                        "read_timeout_ms": 1000,
                        "reconnect_delay_ms": 50,
                    },
                )
            )
            source.exposed_outputs.append("stream_start")
            gray = graph.add(registry.create("cv.gray", block_id="gray"))
            threshold = graph.add(registry.create("cv.threshold", block_id="threshold"))
            blobs = graph.add(
                registry.create(
                    "inspect.blobs",
                    block_id="blobs",
                    parameters={
                        "min_area": 40,
                    },
                )
            )
            track = graph.add(registry.create("sequence.track_image", block_id="track"))
            track.exposed_outputs.append("result_graphics")
            overlay = graph.add(registry.create("cv.overlay_result_graphics", block_id="overlay"))
            film = graph.add(registry.create("output.filmstrip_image", block_id="filmstrip"))
            for block, position in zip(
                (source, gray, threshold, blobs, track, overlay, film),
                ([0, 0], [380, 200], [760, 200], [1140, 200], [1520, 0], [1900, 0], [2280, 0]),
                strict=True,
            ):
                block.position = position
            source.node_size = track.node_size = "large"
            for wire in (
                ("video", "image", "gray", "image"),
                ("gray", "image", "threshold", "image"),
                ("threshold", "image", "blobs", "image"),
                ("blobs", "collection", "track", "detections"),
                ("video", "image", "track", "image"),
                ("video", "stream_start", "track", "reset"),
                ("video", "image", "overlay", "image"),
                ("track", "result_graphics", "overlay", "graphics_1"),
                ("overlay", "image", "filmstrip", "image"),
            ):
                graph.connect(*wire)
            note = graph.add(registry.create("layout.comment", block_id="attribution"))
            note.name, note.position, note.node_size = (
                "Original loopback stream - MIT",
                [0, -500],
                "large",
            )
            note.description = (
                "Original MIT synthetic images, server and projects by "
                "OpenCVDataflowGUI contributors. "
                "No customer/third-party images. Run starts the program-owned loopback-only MJPEG "
                "server; close/new program stops it. Continuous receives frames; Stop ends the "
                "decoder and Resume opens a new stream segment. The reconnect example closes "
                "each HTTP response after four frames. Stream start resets tracking and a new "
                "coordinate identity avoids silently crossing reconnects. Native timestamps are "
                "presentation metadata, not exposure time. JPEG is lossy; "
                "truth.json gives intended "
                "part centers and pinned JPEG hashes. Port 18764 must be free."
            )
            graph.set_program(
                {
                    "name": "Original loopback network video",
                    "enabled": True,
                    "services": [
                        {
                            "id": "mjpeg",
                            "runtime": "application",
                            "argv": [
                                "{python}",
                                "{project_dir}/serve_stream.py",
                                "--port",
                                "18764",
                            ],
                            "ready_text": "READY network-video 18764",
                            "startup_timeout_seconds": 10,
                            "shutdown_timeout_seconds": 2,
                        }
                    ],
                    "assets": ["serve_stream.py", *[f"assets/{n}" for n in hashes]],
                }
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
                    "frames": [
                        {"index": i, "center": [31.5 + 8 * i, 41.5], "area": 672} for i in range(8)
                    ],
                    "asset_sha256": hashes,
                    "cases": CASES,
                    "disconnect_after": 4,
                },
                indent=2,
            )
            + "\n"
        ).encode()
    )


if __name__ == "__main__":
    build()
