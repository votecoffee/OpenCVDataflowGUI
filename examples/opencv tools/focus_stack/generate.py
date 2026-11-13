"""Original stepped-resolution coupon with independent sharp image and capture shifts."""

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
    y, x = np.mgrid[:128, :192]
    level = (((x // 2 + y // 2) % 2) * 24000 + 8000).astype(np.uint16)
    for center in ((36, 40), (155, 87)):
        cv2.circle(level, center, 10, 3000, -1)
    cv2.rectangle(level, (80, 88), (110, 97), 3000, -1)
    wide = level.astype(np.uint32)
    truth = np.stack([wide, wide * 3 // 4, wide // 2], axis=2).astype(np.uint16)
    blurred = cv2.GaussianBlur(truth, (7, 7), 1.5)
    left, right = blurred.copy(), blurred.copy()
    left[:, :120], right[:, 72:] = truth[:, :120], truth[:, 72:]
    shifted = cv2.warpAffine(right, np.float64([[1, 0, 2], [0, 1, -1]]), (192, 128))
    for name, image in (
        ("sharp_truth", truth),
        ("left_focus", left),
        ("right_focus", right),
        ("shifted_right", shifted),
    ):
        assert cv2.imwrite(str(assets / f"{name}.png"), image[..., ::-1])
    (assets / "truth.json").write_text(
        json.dumps(
            {
                "reference": "Direct texture, holes and slot; truth is not a blurred capture",
                "image_size": [192, 128],
                "channel_order": "RGB",
                "input_depth": "uint16",
                "left_sharp_columns": [0, 120],
                "right_sharp_columns": [72, 192],
                "evaluation_xywh": [12, 12, 168, 104],
                "moving_source_shift_xy": [2, -1],
                "expected_source_to_target_shift_xy": [-2, 1],
                "maximum_error": 0,
                "ambiguity": "Equal sharp levels keep anchor pixels with focus index -1",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    registry = builtin_registry()
    for registered, filename in ((False, "project.json"), (True, "registered.json")):
        graph = Graph()
        try:
            for i, (identifier, asset) in enumerate(
                (
                    ("left", "left_focus"),
                    ("right", "shifted_right" if registered else "right_focus"),
                    ("truth", "sharp_truth"),
                )
            ):
                source = graph.add(
                    registry.create(
                        "input.file",
                        block_id=identifier,
                        parameters={
                            "path": f"assets/{asset}.png",
                            "path_base": "project",
                            "color_format": "rgb",
                            "pixel_depth": "unchanged",
                        },
                    )
                )
                source.position = [-500.0, i * 300.0]
            right_id = "right"
            if registered:
                alignment_region = graph.add(
                    registry.create(
                        "region.shape",
                        block_id="alignment_region",
                        parameters={"rectangle": [84, 16, 28, 96]},
                    )
                )
                alignment_region.name = "Shared sharp registration region"
                alignment_region.position = [-100.0, 600.0]
                graph.connect("left", "image", "alignment_region", "image")
                align = graph.add(
                    registry.create(
                        "inspect.align_ecc",
                        block_id="align",
                        parameters={
                            "motion": "translation",
                            "max_iterations": 100,
                            "gaussian_size": 7,
                            "interpolation": "nearest",
                            "min_coefficient": 0.5,
                        },
                    )
                )
                align.position, align.name = [-100.0, 300.0], "Register shifted focus capture"
                graph.connect("right", "image", "align", "image")
                graph.connect("left", "image", "align", "reference")
                graph.connect("alignment_region", "mask", "align", "reference_mask")
                right_id = "align"
            group = graph.add(
                registry.create("flow.batch", block_id="batch", parameters={"size": 2})
            )
            group.position = [300.0, 100.0]
            graph.connect("left", "image", "batch", "item_1")
            graph.connect(right_id, "image", "batch", "item_2")
            region = graph.add(
                registry.create(
                    "region.shape", block_id="region", parameters={"rectangle": [12, 12, 168, 104]}
                )
            )
            region.position = [-100.0, -200.0]
            graph.connect("left", "image", "region", "image")
            focus = graph.add(
                registry.create(
                    "cv.focus_stack",
                    block_id="focus",
                    parameters={
                        "alignment": "pixel_grid",
                        "minimum_valid_fraction": 0.3,
                    },
                )
            )
            focus.position, focus.node_size = [700.0, 100.0], "large"
            graph.connect("batch", "group", "focus", "images")
            graph.connect("region", "region", "focus", "region")
            for name, source in (("before", "left"), ("verify", "focus")):
                tool = graph.add(
                    registry.create(
                        "inspect.image_quality",
                        block_id=name,
                        parameters={
                            "metric": "max_absolute_error",
                            "check_threshold": name == "verify",
                            "threshold": 0,
                        },
                    )
                )
                tool.name = (
                    "Verify sharp pixel truth" if name == "verify" else "Anchor error before fusion"
                )
                tool.position = [1100.0, 100.0 if name == "verify" else 450.0]
                graph.connect(source, "image", name, "image_a")
                graph.connect("truth", "image", name, "image_b")
                graph.connect("region", "mask", name, "mask")
            note = graph.add(registry.create("layout.comment", block_id="about"))
            note.name, note.position, note.node_size = (
                "Original focus fixtures — MIT",
                [-500.0, -350.0],
                "large",
            )
            note.description = (
                "Original 16-bit RGB coupon, projects and generator under the repository's MIT "
                "license. Complementary captures share an equally sharp overlap. Independent truth "
                "is the unblurred coupon. Reliable pixels fuse; ambiguous pixels keep the anchor "
                "and index -1. The selected interior must match truth exactly. The registration "
                "project shifts the second capture by [2,-1] and applies ordinary ECC using the "
                "shared sharp registration region. Its evaluation region excludes resampling "
                "gaps and scoring halos. Focus index is not calibrated depth. "
                "OpenCV reference: https://docs.opencv.org/5.0/main_modules/imgproc_filter.html"
            )
            graph.save(folder / filename)
        finally:
            graph.close()


if __name__ == "__main__":
    main()
