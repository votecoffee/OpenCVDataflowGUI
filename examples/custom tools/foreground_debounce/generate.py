"""Original labeled-mask sequence and independent expected foreground pixels."""

import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def main():
    folder = Path(__file__).resolve().parent
    inputs, expected = folder / "assets/inputs", folder / "assets/expected"
    inputs.mkdir(parents=True, exist_ok=True)
    expected.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(9):
        image = np.zeros((64, 112), np.uint8)
        image[44:52, 72:96] = 127  # Stable shadow label, excluded by raw threshold 127.
        image[60, 105] = 255  # Isolated pixel is below the minimum area.
        if index not in (4, 8):
            image[28:34, 10 + 6 * index : 14 + 6 * index] = 255  # Non-overlapping successive masks.
        if index in (1, 3):
            image[8:13, 88:94] = 255  # Repeated, but never consecutive, glint.
        truth = np.zeros_like(image)
        if index in (2, 3, 7):
            truth[28:34, 10 + 6 * index : 14 + 6 * index] = 255
        name = f"frame-{index:02d}.png"
        assert cv2.imwrite(str(inputs / name), image)
        assert cv2.imwrite(str(expected / name), truth)
        paths.append((inputs / name).relative_to(folder).as_posix())
    truth = {
        "origin": "Original synthetic labeled masks, analytic motion and expected pixels; MIT",
        "frame_count": 9,
        "image_size": [112, 64],
        "expected_counts": [None, None, 1, 1, 0, 0, 0, 1, 0],
        "null_means": "Warmup: no primary output event, not an empty emitted mask",
        "part_area_pixels": 24,
        "part_centers": [[11.5 + 6 * i, 30.5] if i not in (4, 8) else None for i in range(9)],
        "required_consecutive_hits": 3,
        "maximum_displacement_pixels": 8,
        "shadow_label": 127,
        "maximum_expected_mask_error": 0,
    }
    (folder / "assets/truth.json").write_text(
        json.dumps(truth, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    registry = builtin_registry()
    graph = Graph()
    try:
        source = graph.add(
            registry.create(
                "input.file",
                block_id="source",
                parameters={
                    "source_mode": "files",
                    "paths": paths,
                    "path_base": "project",
                    "color_format": "gray",
                    "pixel_depth": "unchanged",
                },
            )
        )
        source.name, source.position = "Part, glints and shadow labels", [0, 20]
        tool = graph.add(
            registry.create(
                "sequence.foreground_debounce",
                block_id="debounce",
                parameters={"foreground_threshold": 127, "maximum_displacement": 8},
            )
        )
        tool.name, tool.position, tool.node_size = (
            "Require three consecutive observations",
            [440, 20],
            "large",
        )
        tool.exposed_outputs = ["mask", "components", "count", "ready", "status"]
        graph.connect(source.id, "image", tool.id, "image")
        note = graph.add(registry.create("layout.comment", block_id="about"))
        note.name, note.position, note.node_size = (
            "Original foreground fixtures - MIT",
            [0, -300],
            "large",
        )
        note.description = (
            "Original labeled-mask images, expected masks, generator and graph under the "
            "repository's MIT license. Run all for nine acquisitions. Two warmup activations "
            "emit no primary data. Counts from frame 2 onward are 1,1,0,0,0,1,0. A disjoint "
            "moving part confirms after three consecutive unambiguous observations. Missing "
            "frame 4 resets it; reappearance requires confirmation again. Glints in frames "
            "1 and 3 never confirm. Label 127 is excluded by threshold 127; isolated pixels "
            "are below minimum area. Green boxes mean confirmed; amber boxes mean pending. "
            "All geometry is current-grid pixels. No old mask or old ticket is re-emitted. "
            "This is component debounce, not general identity tracking or occlusion recovery. "
            "API: https://docs.opencv.org/5.0/main_modules/imgproc_shape.html"
        )
        graph.save(folder / "project.json")
    finally:
        graph.close()


if __name__ == "__main__":
    main()
