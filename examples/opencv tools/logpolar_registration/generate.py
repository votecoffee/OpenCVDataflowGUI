"""Original textured part and independent analytic similarity/ambiguity fixtures."""

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
    gray = np.zeros((192, 192), np.float64)
    cv2.rectangle(gray, (48, 48), (115, 134), 0.7, -1)
    cv2.circle(gray, (128, 72), 19, 1, -1)
    cv2.circle(gray, (72, 84), 11, 0, -1)
    cv2.rectangle(gray, (86, 106), (105, 125), 0.1, -1)
    gray += np.random.default_rng(137).random(gray.shape) * 0.12 * (gray > 0)
    gray = cv2.GaussianBlur(gray, (3, 3), 0.5)
    reference = np.rint(
        np.stack([gray, gray * 0.75, gray * 0.5], axis=2) / gray.max() * 60000
    ).astype(np.uint16)
    angle, scale, offset = 145.0, 0.93, (-3.0, 6.0)
    c = 95.5
    a, b = scale * math.cos(math.radians(angle)), scale * math.sin(math.radians(angle))
    forward = np.array(
        [[a, b, c - a * c - b * c + offset[0]], [-b, a, c + b * c - a * c + offset[1]], [0, 0, 1]]
    )
    moving = cv2.warpAffine(reference, forward[:2], (192, 192))
    symmetric = np.maximum(reference, reference[::-1, ::-1])
    for name, image in (("reference", reference), ("moving", moving), ("symmetric", symmetric)):
        assert cv2.imwrite(str(assets / f"{name}.png"), image[..., ::-1])
    truth = {
        "reference": "Original textured part; transform derived directly from sine/cosine",
        "image_size": [192, 192],
        "channel_order": "RGB",
        "input_depth": "uint16",
        "reference_to_moving": forward.tolist(),
        "moving_to_reference": np.linalg.inv(forward).tolist(),
        "generated_angle": angle,
        "generated_scale": scale,
        "generated_offset": list(offset),
        "controls_reference": [[50, 60], [120, 60], [120, 120], [50, 120]],
        "maximum_control_error_pixels": 0.3,
        "maximum_image_rmse_counts": 2000,
        "symmetric": "Exactly invariant under array reversal (180-degree rotation)",
    }
    (assets / "truth.json").write_text(
        json.dumps(truth, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    registry = builtin_registry()
    for filename, symmetry, prior in (
        ("project.json", False, False),
        ("symmetric.json", True, False),
        ("orientation_prior.json", True, True),
    ):
        graph = Graph()
        try:
            for i, (identifier, asset) in enumerate(
                (
                    ("reference", "symmetric" if symmetry else "reference"),
                    ("moving", "symmetric" if symmetry else "moving"),
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
                source.position = [-350.0, i * 320.0]
            tool = graph.add(
                registry.create(
                    "inspect.logpolar_registration",
                    block_id="register",
                    parameters={
                        **({"minimum_angle": -45.0, "maximum_angle": 45.0} if prior else {}),
                    },
                )
            )
            tool.position, tool.node_size = [50.0, 80.0], "large"
            tool.name = "Require a distinct verified similarity"
            graph.connect("reference", "image", "register", "reference")
            graph.connect("moving", "image", "register", "image")
            if not symmetry:
                tool.exposed_outputs.append("valid_mask")
                for identifier, source in (("before", "moving"), ("verify", "register")):
                    metric = graph.add(
                        registry.create(
                            "inspect.image_quality",
                            block_id=identifier,
                            parameters={
                                "metric": "rmse",
                                "check_threshold": identifier == "verify",
                                "threshold": 2000.0,
                            },
                        )
                    )
                    metric.name = (
                        "Registered image RMSE"
                        if identifier == "verify"
                        else "Before registration RMSE"
                    )
                    metric.position = [500.0, 80.0 if identifier == "verify" else 400.0]
                    graph.connect(source, "image", identifier, "image_a")
                    graph.connect("reference", "image", identifier, "image_b")
                    if identifier == "verify":
                        graph.connect("register", "valid_mask", identifier, "mask")
            note = graph.add(registry.create("layout.comment", block_id="about"))
            note.name, note.node_size, note.position = (
                "Original similarity fixtures — MIT",
                "large",
                [-350.0, -340.0],
            )
            note.description = (
                "Original 16-bit RGB part, projects and generator under the repository's MIT "
                "license. Sine/cosine geometry defines a 145-degree, 0.93-scale forward warp "
                "with offset [-3,6]. Registration recovers its inverse and verifies the "
                "180-degree spectral alternative spatially. The symmetric project fails and "
                "withholds its transform; a [-45,45] orientation prior resolves that fixture. "
                "Pixel scores are not probabilities or occlusion measurements. OpenCV references: "
                "https://docs.opencv.org/5.0/main_modules/imgproc_transform.html and "
                "https://docs.opencv.org/5.0/main_modules/imgproc_motion.html"
            )
            graph.save(folder / filename)
        finally:
            graph.close()


if __name__ == "__main__":
    main()
