"""Original 16-bit inspection fixture and PNG/JP2 lossless comparisons."""

from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def project(folder, filename, color):
    graph, registry = Graph(), builtin_registry()
    try:
        source = graph.add(
            registry.create(
                "input.file",
                block_id="source",
                parameters={
                    "path": "assets/original-16bit.png",
                    "path_base": "project",
                    "color_format": color,
                    "pixel_depth": "unchanged",
                },
            )
        )
        source.name = f"Original 16-bit {color.upper()} fixture"
        for row, codec in enumerate(("png", "jpeg2000")):
            encoder = graph.add(
                registry.create(
                    "cv.encode_image",
                    block_id=f"encode_{codec}",
                    parameters={
                        "codec": codec,
                        "png_compression": 9,
                        "png_filter": "all",
                        "png_zlib_buffer_size": 131072,
                    },
                )
            )
            decoder = graph.add(registry.create("cv.decode_image", block_id=f"decode_{codec}"))
            verify = graph.add(
                registry.create(
                    "inspect.image_quality",
                    block_id=f"verify_{codec}",
                    parameters={
                        "metric": "max_absolute_error",
                        "check_threshold": True,
                        "threshold": 0.0,
                    },
                )
            )
            encoder.position, decoder.position, verify.position = (
                [300.0, row * 280.0],
                [600.0, row * 280.0],
                [900.0, row * 280.0],
            )
            graph.connect(source.id, "image", encoder.id, "image")
            graph.connect(encoder.id, "encoded", decoder.id, "encoded")
            graph.connect(source.id, "image", verify.id, "image_a")
            graph.connect(decoder.id, "image", verify.id, "image_b")
        note = graph.add(registry.create("layout.comment", block_id="fixture_note"))
        note.name = "Original high-bit-depth fixture — MIT"
        note.description = (
            "Original synthetic asset, generator and projects under the MIT license. "
            "Run once: both PNG and lossless JPEG 2000 must report zero maximum absolute "
            "error with uint16 output. The PNG filter, strategy and buffer affect encoding, "
            "not pixels. Selecting lossy JP2 mode will generally fail the zero-error check."
        )
        note.node_size, note.position = "large", [0.0, -300.0]
        graph.save(folder / filename)
    finally:
        graph.close()


def main():
    folder = Path(__file__).resolve().parent
    assets = folder / "assets"
    assets.mkdir(exist_ok=True)
    y, x = np.mgrid[:129, :193]
    image = np.dstack(
        (
            4000 + x * 130 + y * 40,
            7000 + x * 80 + y * 70,
            5000 + x * 90 + y * 60,
            np.full(x.shape, 65535),
        )
    ).astype(np.uint16)
    cv2.rectangle(image, (24, 24), (169, 108), (38000, 46000, 50000, 65535), -1)
    cv2.circle(image, (62, 61), 20, (500, 1300, 2200, 65535), -1)
    cv2.circle(image, (131, 61), 20, (500, 1300, 2200, 65535), -1)
    image[0, :4] = [
        [0, 0, 0, 0],
        [65535, 65535, 65535, 65535],
        [1, 257, 32768, 8192],
        [65534, 1024, 4096, 49152],
    ]
    ok, buffer = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise RuntimeError("Unable to create the original 16-bit fixture")
    (assets / "original-16bit.png").write_bytes(buffer.tobytes())
    project(folder, "project.json", "rgba")
    project(folder, "grayscale.json", "gray")


if __name__ == "__main__":
    main()
