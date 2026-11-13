"""Original odd-sized high-bit-depth fixture and editable multiscale inspection."""

from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def main():
    folder = Path(__file__).resolve().parent
    assets = folder / "assets"
    assets.mkdir(exist_ok=True)
    y, x = np.mgrid[:129, :193]
    image = np.dstack(
        (6000 + x * 85 + y * 30, 7000 + x * 60 + y * 80, 5000 + x * 70 + y * 50)
    ).astype(np.uint16)
    cv2.rectangle(image, (22, 20), (170, 110), (37000, 42000, 46000), -1)
    cv2.circle(image, (57, 60), 19, (8000, 10000, 12000), -1)
    cv2.circle(image, (132, 60), 19, (8000, 10000, 12000), -1)
    for offset in range(0, 21, 3):
        cv2.line(image, (83 + offset, 85), (83 + offset, 100), (54000, 57000, 60000), 1)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Unable to encode the pyramid fixture")
    (assets / "original-16bit.png").write_bytes(encoded.tobytes())
    graph, registry = Graph(), builtin_registry()
    try:
        source = graph.add(
            registry.create(
                "input.file",
                block_id="source",
                parameters={
                    "path": "assets/original-16bit.png",
                    "path_base": "project",
                    "color_format": "bgr",
                    "pixel_depth": "uint16",
                },
            )
        )
        source.name = "Original 193 x 129, 16-bit part"
        build = graph.add(registry.create("cv.laplacian_pyramid", block_id="build"))
        build.position = [290.0, 0.0]
        graph.connect(source.id, "image", build.id, "image")
        for identifier, kind, settings, position in (
            ("detail", "cv.pyramid_level", {"level": "1"}, [600.0, -210.0]),
            ("restore", "cv.reconstruct_laplacian_pyramid", {}, [600.0, 40.0]),
            (
                "suppress_fine",
                "cv.reconstruct_laplacian_pyramid",
                {"detail_gains": [0.0, 1.0, 1.0]},
                [600.0, 310.0],
            ),
        ):
            block = graph.add(registry.create(kind, block_id=identifier, parameters=settings))
            block.position = position
            graph.connect(build.id, "images", block.id, "images")
        graph.blocks["suppress_fine"].name = "Remove finest detail band"
        verify = graph.add(
            registry.create(
                "inspect.image_quality",
                block_id="verify",
                parameters={
                    "metric": "max_absolute_error",
                    "check_threshold": True,
                    "threshold": 0.0,
                },
            )
        )
        verify.name = "Verify exact 16-bit round trip"
        verify.position = [930.0, 40.0]
        graph.connect(source.id, "image", verify.id, "image_a")
        graph.connect("restore", "image", verify.id, "image_b")
        note = graph.add(registry.create("layout.comment", block_id="fixture_note"))
        note.name = "Original multiscale fixture — MIT"
        note.description = (
            "Original synthetic image, generator and project under this repository's MIT license. "
            "Run once: the exact round trip has zero error. Select the detail block to see "
            "a signed band with zero shown as gray. The lower branch removes the finest "
            "detail while preserving output size, dtype and coordinates."
        )
        note.node_size, note.position = "large", [0.0, -310.0]
        graph.save(folder / "project.json")
    finally:
        graph.close()


if __name__ == "__main__":
    main()
