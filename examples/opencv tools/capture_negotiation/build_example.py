"""Original lossless capture fixtures; no camera hardware is opened."""

import hashlib
import json
from pathlib import Path

import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
CASES = {
    "unchanged": {},
    "report_only": {
        "width": 640,
        "height": 480,
        "fps": 30,
        "exposure_mode": "backend_value",
        "exposure_value": -6,
        "negotiation_policy": "report",
        "readback": "every_frame",
    },
    "reject_settings": {"width": 640, "fps": 30, "negotiation_policy": "require_setter"},
    "reject_format": {"fourcc": "MJPG", "negotiation_policy": "require_readback"},
    "converted_host": {"color_format": "gray", "pixel_depth": "uint16"},
}


def frames():
    y, x = np.mgrid[:96, :128]
    background = (25 + (x // 8 + y // 8) % 2 * 15).astype(np.uint8)
    for index in range(3):
        image = np.repeat(background[..., None], 3, axis=2)
        image[24:66, 12 + index * 18 : 38 + index * 18] = [50, 150, 240]
        yield image


def build():
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    path = assets / "reference.avi"
    registry = builtin_registry()
    images = list(frames())
    if not path.exists():
        writer = registry.create(
            "output.video",
            parameters={"path": str(path), "backend": "ffmpeg", "codec": "FFV1", "fps": 24},
        )
        try:
            for image in images:
                assert writer.execute({"image": image}) is not None, writer.error
        finally:
            writer.close()
    source = registry.create(
        "input.video_file", parameters={"path": str(path), "backend": "ffmpeg"}
    )
    try:
        for expected in images:
            result = source.execute({})
            if result is None or not np.array_equal(result.outputs["image"], expected):
                raise ValueError("Existing clip differs; regenerate in a fresh folder")
        assert source.exhausted
    finally:
        source.close()
    for name, options in CASES.items():
        graph = Graph()
        try:
            capture = graph.add(
                registry.create(
                    "input.capture_file",
                    block_id="capture",
                    parameters={
                        "path": "assets/reference.avi",
                        "path_base": "project",
                        "backend": "ffmpeg",
                        **options,
                    },
                )
            )
            capture.exposed_outputs.append("diagnostic")
            capture.node_size = "large"
            gray = graph.add(registry.create("cv.gray", block_id="gray"))
            gray.position = [400, 0]
            graph.connect("capture", "image", "gray", "image")
            note = graph.add(registry.create("layout.comment", block_id="attribution"))
            note.name = "Recorded capture qualification - original MIT fixtures"
            note.position, note.node_size = [0, -450], "large"
            note.description = (
                "Original MIT clip, code and annotations by OpenCVDataflowGUI contributors. "
                "No customer or third-party images. Three exact 128x96 BGR uint8 frames at "
                "24 FPS, encoded losslessly as FFV1. The recorded FFmpeg backend rejects "
                "camera width, FPS, exposure and FourCC setters. Report only keeps actual "
                "frames with warning details; strict policies withhold images and emit a "
                "failed ticket/report. Host conversion is separate from decoder format. "
                "These projects verify software behavior, not physical camera acceptance."
            )
            graph.save(HERE / (name + ".json"))
        finally:
            graph.close()
    (HERE / "truth.json").write_bytes(
        (
            json.dumps(
                {
                    "license": "MIT - original repository fixtures",
                    "size": [128, 96],
                    "fps": 24,
                    "decoded_format": "uint8 BGR",
                    "fourcc": "FFV1",
                    "frame_sha256": [hashlib.sha256(a.tobytes()).hexdigest() for a in images],
                    "asset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "cases": {
                        name: {"passed": not name.startswith("reject"), "settings": settings}
                        for name, settings in CASES.items()
                    },
                },
                indent=2,
            )
            + "\n"
        ).encode()
    )


if __name__ == "__main__":
    build()
