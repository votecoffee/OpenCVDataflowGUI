"""Original nested cutout fixture with independent slice-based reference masks."""

import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def main():
    folder = Path(__file__).resolve().parent
    assets = folder / "assets"
    assets.mkdir(exist_ok=True)
    scene = np.zeros((176, 256), np.uint8)
    scene[16:144, 16:160] = 255
    scene[40:120, 40:136] = 0
    scene[64:96, 64:112] = 255
    scene[72:88, 80:96] = 0
    scene[28:84, 184:240] = 255
    part = scene.copy()
    part[:, 176:] = 0
    solid = np.zeros_like(scene)
    solid[16:144, 16:160] = 255
    solid[28:84, 184:240] = 255
    counts = {
        "all": 144 * 128 - 96 * 80 + 48 * 32 - 16 * 16 + 56 * 56,
        "part": 144 * 128 - 96 * 80 + 48 * 32 - 16 * 16,
        "solid": 144 * 128 + 56 * 56,
    }
    for name, image in (("scene", scene), ("all", scene), ("part", part), ("solid", solid)):
        if name != "scene":
            assert np.count_nonzero(image) == counts[name]
        assert cv2.imwrite(str(assets / f"{name}.png"), image)
    contours, links = cv2.findContours(scene, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    part_id = max(
        (index for index, row in enumerate(links[0]) if row[3] == -1),
        key=lambda index: cv2.contourArea(contours[index]),
    )
    (assets / "truth.json").write_text(
        json.dumps(
            {
                "reference": "Original axis-aligned pixel slices; areas from rectangle arithmetic",
                "image_size": [256, 176],
                "foreground_pixels": counts,
                "part_contour_id": part_id,
                "maximum_allowed_pixel_error": 0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    graph, registry = Graph(), builtin_registry()
    try:
        source = graph.add(
            registry.create(
                "input.file",
                block_id="source",
                parameters={
                    "path": "assets/scene.png",
                    "path_base": "project",
                    "color_format": "gray",
                },
            )
        )
        source.name, source.position = "Original nested cutouts", [-420.0, 0.0]
        detector = graph.add(registry.create("inspect.contour_hierarchy", block_id="hierarchy"))
        detector.position = [-90.0, 0.0]
        graph.connect(source.id, "image", detector.id, "image")
        for row, name in enumerate(("all", "part", "solid")):
            raster = graph.add(
                registry.create(
                    "collection.rasterize_contours",
                    block_id=name,
                    parameters={
                        "selection": "ids" if name == "part" else "all",
                        "items": [part_id] if name == "part" else [],
                        "topology": "solid" if name == "solid" else "hierarchy",
                        "check_pixels": True,
                        "minimum_pixels": counts[name],
                        "maximum_pixels": counts[name],
                    },
                )
            )
            raster.name = {
                "all": "Preserve all topology",
                "part": "Select one part with holes",
                "solid": "Explicitly fill all holes",
            }[name]
            raster.description = f"Expected foreground area: {counts[name]:,} pixels."
            raster.position = [250.0, row * 360.0]
            graph.connect(detector.id, "collection", raster.id, "collection")
            graph.connect(source.id, "image", raster.id, "image")
            reference = graph.add(
                registry.create(
                    "input.file",
                    block_id=f"truth_{name}",
                    parameters={
                        "path": f"assets/{name}.png",
                        "path_base": "project",
                        "color_format": "gray",
                    },
                )
            )
            reference.name, reference.position = (
                f"Independent {name} reference",
                [590.0, 980 + row * 340.0],
            )
            verify = graph.add(
                registry.create(
                    "inspect.image_quality",
                    block_id=f"verify_{name}",
                    parameters={
                        "metric": "max_absolute_error",
                        "check_threshold": True,
                        "threshold": 0.0,
                    },
                )
            )
            verify.name, verify.position = f"Verify {name}: zero pixel error", [630.0, row * 360.0]
            graph.connect(raster.id, "mask", verify.id, "image_a")
            graph.connect(reference.id, "image", verify.id, "image_b")
        note = graph.add(registry.create("layout.comment", block_id="about"))
        note.name = "Original contour rasterization fixture — MIT"
        note.description = (
            "Original nested cutout images, generator, references and project under this "
            "repository's MIT license. Three branches preserve all topology, select one "
            "outer contour with its holes/islands, or deliberately fill holes. Pixel areas "
            "are independent rectangle arithmetic; all comparison errors must be zero. "
            "Contour IDs are local to this fixture/run, not persistent part identities. "
            "OpenCV reference: https://docs.opencv.org/5.0/main_modules/imgproc_draw.html"
        )
        note.node_size, note.position = "large", [-420.0, -350.0]
        graph.save(folder / "project.json")
    finally:
        graph.close()


if __name__ == "__main__":
    main()
