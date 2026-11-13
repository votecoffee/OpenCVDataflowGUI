"""Original 16-bit patches with independently known means, noise and contrast."""

import json
import math
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def main():
    folder = Path(__file__).resolve().parent
    assets = folder / "assets"
    assets.mkdir(exist_ok=True)
    truth = {
        "origin": "Original 16-bit RGB patches, balanced +/-400 checker texture; MIT",
        "signal_rectangle": [20, 24, 40, 32],
        "background_rectangle": [88, 24, 40, 32],
        "pixels_per_selection": 1280,
        "population_stddev": 400,
        "minimum_cnr": 5,
    }
    yy, xx = np.indices((96, 160))
    noise = np.where((xx + yy) % 2, 400, -400)
    for name, means in [("strong", (24000, 22000, 20000)), ("weak", (21600, 20800, 20000))]:
        image = np.full((96, 160, 3), 12000, np.uint16)
        image[24:56, 20:60] = np.asarray(means) + noise[24:56, 20:60, None]
        image[24:56, 88:128] = 20000 + noise[24:56, 88:128, None]
        assert cv2.imwrite(str(assets / (name + ".png")), image[..., ::-1])
        cnr = [(mean - 20000) / math.sqrt(2 * 400**2) for mean in means]
        truth[name] = {
            "signal_means": list(means),
            "background_means": [20000] * 3,
            "channel_cnr": cnr,
            "cnr": cnr[0],
            "passed": cnr[0] >= 5,
        }
    (assets / "truth.json").write_text(
        json.dumps(truth, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    registry = builtin_registry()
    for filename, invalid in [("project.json", False), ("invalid_selection.json", True)]:
        graph = Graph()
        graph.set_branch_workers(2)
        try:
            source = graph.add(
                registry.create(
                    "input.file",
                    block_id="source",
                    parameters={
                        "source_mode": "single" if invalid else "files",
                        "path": "assets/strong.png",
                        "paths": ["assets/strong.png", "assets/weak.png"],
                        "path_base": "project",
                        "color_format": "rgb",
                        "pixel_depth": "unchanged",
                    },
                )
            )
            source.name = "Original strong / weak patches"
            source.position = [0, 180]
            source.node_size = "large"
            for index, name in enumerate(("signal", "background")):
                rectangle = (
                    truth["signal_rectangle"]
                    if name == "signal" or invalid
                    else truth["background_rectangle"]
                )
                region = graph.add(
                    registry.create(
                        "region.shape",
                        block_id=name,
                        parameters={"shape": "rectangle", "rectangle": rectangle},
                    )
                )
                region.name = name.title() + " selection"
                region.position = [430, index * 320]
                region.exposed_outputs = ["region", "mask"]
                graph.connect("source", "image", name, "image")
            tool = graph.add(
                registry.create(
                    "inspect.region_contrast",
                    block_id="contrast",
                    parameters={
                        "channel_mode": "native",
                        "channel": "R",
                        "check_threshold": True,
                        "minimum_value": 5,
                        "polarity": "bright",
                    },
                )
            )
            tool.position = [900, 170]
            tool.node_size = "large"
            tool.name = "Require red-channel CNR >= 5"
            tool.exposed_inputs = ["image", "signal_region", "background_region"]
            tool.exposed_outputs = ["image", "value", "valid", "measurements", "cnr"]
            graph.connect("source", "image", "contrast", "image")
            for name in ("signal", "background"):
                graph.connect(name, "region", "contrast", name + "_region")
            note = graph.add(registry.create("layout.comment", block_id="about"))
            note.name = "Original contrast fixtures - MIT"
            note.position = [0, -260]
            note.node_size = "large"
            note.description = (
                "Original 16-bit RGB patches, analytic truth, generator and graph under the "
                "repository's MIT license. Run all in project.json: strong.png passes and weak.png "
                "fails the R-channel CNR minimum of 5. Each 40x32 selection has 1,280 pixels "
                "and balanced +/-400 samples, so population noise is exactly 400 counts. "
                "CNR is abs(signal_mean-background_mean)/hypot(signal_stddev,background_stddev): "
                "strong R is 7.0710678119 and weak R is 2.8284271247. G/B results also appear "
                "in Measurements; the decision uses R only. Invalid_selection.json overlaps both "
                "selectors and must fail without a Value event. Cyan outlines signal and orange "
                "outlines background on the unmodified input. This is a descriptive spatial "
                "contrast ratio, not detection probability or a final ticket judgment. "
                "API: https://docs.opencv.org/5.0/main_modules/core_array.html"
            )
            graph.save(folder / filename)
        finally:
            graph.close()


if __name__ == "__main__":
    main()
