"""Original binary bracket masks with independently specified tab/notch geometry."""

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
    reference = np.zeros((128, 192), np.uint16)
    reference[20:104, 24:152] = 65535
    reference[40:50, 50:60] = 0
    reference[75:85, 115:125] = 0
    accepted = reference.copy()
    accepted[44:48, 152:156] = 65535
    defect = accepted.copy()
    defect[99:104, 70:77] = 0
    for name, image in (("reference", reference), ("accepted", accepted), ("defect", defect)):
        assert cv2.imwrite(str(assets / f"{name}.png"), image)
    truth = {
        "reference": "Direct half-open rectangle assignments; no distance-transform reference",
        "image_size": [192, 128],
        "reference_pixels": 128 * 84 - 2 * 10 * 10,
        "accepted": {"maximum_boundary_distance": 4, "missing_pixels": 0, "extra_pixels": 16},
        "defect": {"maximum_boundary_distance": 5, "missing_pixels": 35, "extra_pixels": 16},
        "tolerance": 4,
        "tab_xywh": [152, 44, 4, 4],
        "notch_xywh": [70, 99, 7, 5],
    }
    (assets / "truth.json").write_text(
        json.dumps(truth, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    registry = builtin_registry()
    for filename, sample in (("project.json", "accepted"), ("defect.json", "defect")):
        graph = Graph()
        try:
            for row, (identifier, asset) in enumerate(
                (("reference", "reference"), ("sample", sample))
            ):
                source = graph.add(
                    registry.create(
                        "input.file",
                        block_id=identifier,
                        parameters={
                            "path": f"assets/{asset}.png",
                            "path_base": "project",
                            "color_format": "gray",
                            "pixel_depth": "unchanged",
                        },
                    )
                )
                source.position = [-360.0, row * 340.0]
            tool = graph.add(
                registry.create(
                    "inspect.mask_boundary",
                    block_id="compare",
                    parameters={
                        "alignment": "pixel_grid",
                        "tolerance": 4.0,
                        "check_area": True,
                        "area_basis": "raw",
                        "maximum_missing_pixels": 0,
                        "maximum_extra_pixels": 16,
                    },
                )
            )
            tool.position, tool.node_size = [90.0, 80.0], "large"
            tool.name = "Accept tab, reject deep notch"
            tool.description = (
                "Four-pixel inclusive boundary tolerance; no missing area, up to 16 extra pixels. "
                "Both directed boundaries include holes and disconnected features."
            )
            graph.connect("reference", "image", "compare", "reference")
            graph.connect("sample", "image", "compare", "sample")
            note = graph.add(registry.create("layout.comment", block_id="about"))
            note.name, note.position, note.node_size = (
                "Original mask fixtures — MIT",
                [-360.0, -330.0],
                "large",
            )
            note.description = (
                "Original 16-bit bracket masks, projects and generator, under the repository's "
                "MIT license. The accepted tab adds 16 pixels and extends four pixels; the defect "
                "removes a 7 by 5 notch (35 pixels) and moves its inner boundary five pixels. "
                "Truth follows direct rectangle dimensions, independently of the inspection block. "
                "OpenCV reference: https://docs.opencv.org/5.0/main_modules/imgproc_misc.html"
            )
            graph.save(folder / filename)
        finally:
            graph.close()


if __name__ == "__main__":
    main()
