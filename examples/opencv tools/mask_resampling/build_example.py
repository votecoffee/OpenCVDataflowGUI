"""Original MIT mask fixture and deterministic coverage-resize projects."""

from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
CREDIT = (
    "Original synthetic mask by OpenCVDataflowGUI contributors (MIT). No external images. "
    "Pixel-cell area coverage is a software validity model, not physical camera calibration "
    "or proof of full cubic/Lanczos interpolation support."
)


def build():
    source = np.full((192, 256), 255, np.uint8)
    source[:13] = source[:, :13] = 0
    source[60:101, 95:132] = 0
    source[120:160, 40:88:2] = 0
    source[35, 20:230] = 0
    ok, data = cv2.imencode(".png", source)
    assert ok
    (HERE / "mask.png").write_bytes(data.tobytes())
    registry = builtin_registry()
    for case in ("coverage_thresholds", "fractional_chain", "incompatible_target"):
        graph = Graph()

        def add(kind, name, layout_x, layout_y=0, **parameters):
            block = graph.add(registry.create(kind, block_id=name, parameters=parameters))
            block.position = [layout_x, layout_y]
            return block

        add("input.file", "source", 0, path="mask.png", path_base="project", color_format="gray")
        note = add("layout.comment", "scope", 0, -350)
        note.name = "Coverage, thresholds and source"
        note.node_size = "large"
        note.description = CREDIT
        if case == "coverage_thresholds":
            add("cv.resize", "target", 330, interpolation="area", width=128, height=96)
            graph.connect("source", "image", "target", "image")
            for index, (name, limit) in enumerate((("full", 1.0), ("half", 0.5), ("any", 0.0))):
                add("cv.resample_mask", name, 660, index * 350, minimum_coverage=limit)
                add("region.measure", f"measure_{name}", 990, index * 350)
                graph.connect("source", "image", name, "mask")
                graph.connect("target", "image", name, "target")
                graph.connect("target", "image", f"measure_{name}", "image")
                graph.connect(name, "mask", f"measure_{name}", "mask")
            note.description += (
                " Resize 256x192 to 128x96: thresholds one (full area), one-half (inclusive) "
                "and zero (any positive area) show different binary masks for holes, "
                "alternating columns and a one-pixel invalid line. Empty footprints are "
                "never accepted. The target supplies size and has the verified resize frame. "
                "Inspect Coverage and the diagnostic heatmap; measurement blocks count each "
                "accepted selection on the target grid. This graph performs no equipment I/O."
            )
        elif case == "fractional_chain":
            add("cv.resample_mask", "first", 330, width=128, height=96)
            add(
                "cv.resample_mask",
                "fractional",
                660,
                width=64,
                height=48,
                input_mode="coverage",
                minimum_coverage=0.5,
            )
            add(
                "cv.resample_mask",
                "thresholded_first",
                660,
                350,
                width=64,
                height=48,
                minimum_coverage=0.5,
            )
            add("cv.resample_mask", "direct", 660, 700, width=64, height=48, minimum_coverage=0.5)
            graph.connect("source", "image", "first", "mask")
            graph.connect("first", "coverage", "fractional", "mask")
            graph.connect("first", "mask", "thresholded_first", "mask")
            graph.connect("source", "image", "direct", "mask")
            note.description += (
                " Carry first-stage fractional coverage to the next resize instead of "
                "thresholding it early. These aligned factor-two footprints make the "
                "fractional chain match the direct factor-four area calculation exactly. "
                "The thresholded-first branch loses valid area. General repeated fractional "
                "resizing of nonaligned grids is not equivalent to resampling the original; "
                "keep the original mask when accuracy across such chains matters."
            )
        else:
            add("cv.crop", "cropped_target", 330, x=0, y=0, width=128, height=96)
            add("cv.resample_mask", "rejected", 660)
            graph.connect("source", "image", "cropped_target", "image")
            graph.connect("source", "image", "rejected", "mask")
            graph.connect("cropped_target", "image", "rejected", "target")
            note.description += (
                " EXPECTED INVALID RESULT: the target is a 128x96 crop, not a resize of "
                "the whole 256x192 grid. Equal target dimensions do not establish alignment. "
                "The resampler rejects the known frame mismatch and emits nothing. "
                "Use crop geometry to transform the selection separately."
            )
        graph.save(HERE / f"{case}.json")
        graph.close()


if __name__ == "__main__":
    build()
