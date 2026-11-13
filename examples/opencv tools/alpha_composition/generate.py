"""Original 16-bit alpha layers and a scalar, exact-rational source-over reference."""

import json
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph


def main():
    folder = Path(__file__).resolve().parent
    assets = folder / "assets"
    assets.mkdir(exist_ok=True)
    height, width, maximum = 128, 192, 65535
    y, x = np.mgrid[:height, :width]
    rng = np.random.default_rng(132)
    bg = np.empty((height, width, 4), np.uint16)
    bg[..., :3] = (
        np.array([42000, 45000, 39000]) + rng.integers(-900, 901, (height, width, 3)) * 3
    ).astype(np.uint16)
    bg[..., 3] = maximum
    bg[:36, :48, 3] = 0
    bg[:, 144:, 3] = 21845
    for cy in (36, 92):
        for cx in (34, 94, 154):
            # Draw on a contiguous three-channel patch; retain the separately defined alpha.
            colors = bg[..., :3].copy()
            cv2.circle(colors, (cx, cy), 12, (9000, 12000, 6000), -1)
            bg[..., :3] = colors
    radius = np.hypot(x - 96, y - 64)
    alpha = np.select(
        [radius < 22, radius < 36, radius < 48], [maximum, 43690, 21845], default=0
    ).astype(np.uint16)
    foreground = np.empty_like(bg)
    foreground[..., :3] = [9000, 15000, 60000]
    foreground[..., 3] = alpha
    premultiplied = foreground.copy()
    premultiplied[..., :3] = np.rint(
        foreground[..., :3].astype(float) * (alpha.astype(float) / maximum)[..., None]
    ).astype(np.uint16)
    reference = np.zeros_like(bg)
    for row in range(height):
        for column in range(width):
            af, ab = int(alpha[row, column]), int(bg[row, column, 3])
            denominator = af * maximum + ab * (maximum - af)
            reference[row, column, 3] = round(Fraction(denominator, maximum))
            if denominator:
                for channel in range(3):
                    numerator = int(foreground[row, column, channel]) * af * maximum + int(
                        bg[row, column, channel]
                    ) * ab * (maximum - af)
                    reference[row, column, channel] = round(Fraction(numerator, denominator))
    for name, image in (
        ("background", bg),
        ("foreground", foreground),
        ("premultiplied", premultiplied),
        ("alpha", alpha),
        ("reference", reference),
    ):
        assert cv2.imwrite(str(assets / f"{name}.png"), image)
    (assets / "truth.json").write_text(
        json.dumps(
            {
                "operator": "source_over",
                "sample_depth": "uint16",
                "maximum": maximum,
                "reference_method": "Scalar equations evaluated with exact rational arithmetic",
                "maximum_allowed_error": 1,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    graph, registry = Graph(), builtin_registry()
    try:

        def source(block_id, filename, position, color_format="unchanged"):
            block = graph.add(
                registry.create(
                    "input.file",
                    block_id=block_id,
                    parameters={
                        "path": f"assets/{filename}.png",
                        "path_base": "project",
                        "pixel_depth": "unchanged",
                        "color_format": color_format,
                    },
                )
            )
            block.position = position
            return block

        source("background", "background", [-330.0, 0.0])
        source("reference", "reference", [820.0, 900.0])
        source("foreground", "foreground", [0.0, 0.0])
        source("associated", "premultiplied", [0.0, 320.0])
        source("rgb", "foreground", [0.0, 640.0], "bgr")
        source("matte", "alpha", [0.0, 920.0], "gray")
        for index, (name, foreground_id) in enumerate(
            (("straight", "foreground"), ("premultiplied", "associated"), ("external", "rgb"))
        ):
            composition = graph.add(
                registry.create(
                    "cv.alpha_over",
                    block_id=name,
                    parameters={
                        "foreground_representation": "premultiplied"
                        if name == "premultiplied"
                        else "straight",
                        "alignment": "pixel_grid",
                        "ticket_policy": "background",
                    },
                )
            )
            composition.name = f"Source over — {name} alpha"
            composition.position = [400.0, index * 320.0]
            graph.connect(foreground_id, "image", name, "foreground")
            graph.connect("background", "image", name, "background")
            if name == "external":
                graph.connect("matte", "image", name, "alpha")
            verify = graph.add(
                registry.create(
                    "inspect.image_quality",
                    block_id=f"verify_{name}",
                    parameters={
                        "metric": "max_absolute_error",
                        "check_threshold": True,
                        "threshold": 1.0,
                    },
                )
            )
            verify.name, verify.position = (
                f"Verify {name} against rational reference",
                [820.0, index * 320.0],
            )
            graph.connect(name, "image", verify.id, "image_a")
            graph.connect("reference", "image", verify.id, "image_b")
        note = graph.add(registry.create("layout.comment", block_id="about"))
        note.name = "Original source-over fixtures — MIT"
        note.description = (
            "Original 16-bit background, transparent overlay, associated image, alpha map, "
            "generator and project under the repository MIT license. Three branches compare "
            "straight, premultiplied and separate-map representations against an independent "
            "exact-rational reference. Premultiplied PNG is an intentional processing fixture; "
            "its block explicitly declares that association. The standard output is straight "
            "BGRA. Checkerboard previews show real transparency. W3C source-over equations: "
            "https://www.w3.org/TR/compositing-1/#porterduffcompositingoperators"
        )
        note.node_size, note.position = "large", [-330.0, -350.0]
        graph.save(folder / "project.json")
    finally:
        graph.close()


if __name__ == "__main__":
    main()
