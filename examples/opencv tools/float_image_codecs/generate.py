"""Original HDR-range fixtures for exact TIFF and quantized Radiance round trips."""

from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def project(folder, name, filename, color, include_hdr):
    graph, registry = Graph(), builtin_registry()
    try:
        source = graph.add(
            registry.create(
                "input.file",
                block_id="source",
                parameters={
                    "path": f"assets/{filename}",
                    "path_base": "project",
                    "color_format": color,
                    "pixel_depth": "unchanged",
                },
            )
        )
        for row, codec in enumerate(("tiff", "hdr") if include_hdr else ("tiff",)):
            encoder = graph.add(
                registry.create(
                    "cv.encode_image", block_id=f"encode_{codec}", parameters={"codec": codec}
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
                        "threshold": 0.0 if codec == "tiff" else 0.0625,
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
        preview = graph.add(
            registry.create("cv.normalize", block_id="preview", parameters={"preserve_alpha": True})
        )
        preview.name = "Display only: normalize HDR range"
        preview.position = [300.0, -270.0]
        graph.connect(source.id, "image", preview.id, "image")
        note = graph.add(registry.create("layout.comment", block_id="fixture_note"))
        note.name = "Original floating-point fixtures — MIT"
        note.description = (
            "Original synthetic assets, generator and projects under the repository's MIT license. "
            "Run once: TIFF must have zero error. The float32 project also checks the bounded "
            "loss of Radiance HDR. Normalization is on a display branch only. OpenEXR is not "
            "required; the current supported OpenCV build has no OpenEXR codec."
        )
        note.node_size, note.position = "large", [0.0, -300.0]
        graph.save(folder / name)
    finally:
        graph.close()


def main():
    folder = Path(__file__).resolve().parent
    assets = folder / "assets"
    assets.mkdir(exist_ok=True)
    y, x = np.mgrid[:65, :97]
    image32 = np.dstack((0.3 + x / 60, 0.4 + y / 40, 0.2 + (x + y) / 80)).astype(np.float32)
    cv2.rectangle(image32, (12, 12), (84, 54), (6, 9, 12), -1)
    cv2.circle(image32, (31, 31), 10, (0.2, 0.3, 0.4), -1)
    cv2.circle(image32, (65, 31), 10, (0.2, 0.3, 0.4), -1)
    image64 = np.dstack(
        (-0.5 + x / 30, 1.123456789012345 + y / 25, (x + y) / 40, np.ones(x.shape))
    ).astype(np.float64)
    for filename, image in (("float32-color.tiff", image32), ("float64-rgba.tiff", image64)):
        ok, buffer = cv2.imencode(
            ".tiff", image, [cv2.IMWRITE_TIFF_COMPRESSION, cv2.IMWRITE_TIFF_COMPRESSION_NONE]
        )
        if not ok:
            raise RuntimeError("Unable to create a floating TIFF fixture")
        (assets / filename).write_bytes(buffer.tobytes())
    project(folder, "project.json", "float32-color.tiff", "rgb", True)
    project(folder, "float64.json", "float64-rgba.tiff", "rgba", False)


if __name__ == "__main__":
    main()
