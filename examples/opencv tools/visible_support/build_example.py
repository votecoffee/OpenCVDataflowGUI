"""Original MIT inspection controls with independently specified object support."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
CASES = {
    "clear": {"positions": [[20, 18], [110, 60], [200, 110]], "selected": [215, 120]},
    "touching": {"positions": [[20, 18], [130, 100], [161, 100]], "selected": [35, 28]},
    "covered": {
        "positions": [[20, 18], [110, 60], [200, 110]],
        "selected": [125, 70],
        "covered": True,
    },
    "required_three": {
        "positions": [[20, 18], [110, 60], [200, 110]],
        "selected": None,
        "covered": True,
        "minimum_matches": 3,
    },
}


def build():
    assets = HERE / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    template = np.random.default_rng(16961).integers(30, 241, (21, 31), np.uint8)
    template[7:12, 12:19] = 10  # A background hole, excluded from the expected-object region.
    selection = np.ones(template.shape, bool)
    selection[7:12, 12:19] = False
    hashes = {}

    def image_file(name, image):
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        path = assets / name
        path.write_bytes(encoded.tobytes())
        hashes[name] = hashlib.sha256(encoded).hexdigest()

    image_file("template.png", template)
    truth = {
        "license": "MIT - original repository fixtures",
        "image_size": [280, 170],
        "template_size": [31, 21],
        "expected_samples": int(selection.sum()),
        "cases": {},
        "asset_sha256": hashes,
    }
    registry = builtin_registry()
    for case, options in CASES.items():
        image = np.full((170, 280), 10, np.uint8)
        for x, y in options["positions"]:
            image[y : y + 21, x : x + 31] = template
        if options.get("covered"):
            x, y = options["positions"][-1]
            image[y : y + 21, x : x + 10] = 0
        image_file(f"{case}.png", image)
        expected = [int(selection.sum())] * 3
        if options.get("covered"):
            expected[-1] -= int(selection[:, :10].sum())
        truth["cases"][case] = {
            "positions": options["positions"],
            "matching_samples": expected,
            "accepted": 3 if not options.get("covered") else 2,
            "selected_pose": options["selected"],
            "expected_pass": case != "required_three",
        }
        graph = Graph()
        try:

            def add(kind, key, parameters=None, position=(0, 0)):
                block = graph.add(registry.create(kind, block_id=key, parameters=parameters or {}))
                block.position = list(position)
                return block

            add(
                "input.file",
                "source",
                {"path": f"assets/{case}.png", "path_base": "project", "color_format": "gray"},
            )
            add(
                "input.file",
                "template",
                {"path": "assets/template.png", "path_base": "project", "color_format": "gray"},
                (0, 350),
            )
            add(
                "region.shape",
                "expected_object",
                {"rectangle": [12, 7, 7, 5], "invert": True},
                (380, 350),
            )
            detect = add(
                "inspect.template_instances",
                "detector",
                {
                    "method": "ccoeff_normed",
                    "min_score": 0.35,
                    "maximum_matches": 3,
                    "minimum_matches": 3,
                    "maximum_iou": 0.15,
                    "draw_overlay": False,
                },
                (760, 0),
            )
            detect.description = (
                "Find all three candidate poses; visibility is a separate qualification."
            )
            support = add(
                "inspect.visible_support",
                "support",
                {"absolute_tolerance": 0.0, "minimum_matches": options.get("minimum_matches", 1)},
                (1140, 0),
            )
            support.exposed_outputs.append("result_graphics")
            support.node_size = "large"
            pick = add(
                "inspect.select_pick",
                "pick",
                {
                    "footprint": [-15, -10, 30, 20],
                    "clearance": 6,
                    "min_score": 0.35,
                    "draw_overlay": False,
                },
                (1520, 350),
            )
            pick.input_failure_policies = {"image": "reject_failed", "matches": "reject_failed"}
            add("cv.overlay_result_graphics", "overlay", position=(1520, 0))
            add("output.filmstrip_image", "filmstrip", position=(1900, 0))
            for wire in [
                ("template", "image", "expected_object", "image"),
                ("source", "image", "detector", "image"),
                ("template", "image", "detector", "template"),
                ("source", "image", "support", "image"),
                ("template", "image", "support", "template"),
                ("expected_object", "region", "support", "template_region"),
                ("detector", "matches", "support", "matches"),
                ("support", "image", "pick", "image"),
                ("support", "matches", "pick", "matches"),
                ("support", "image", "overlay", "image"),
                ("support", "result_graphics", "overlay", "graphics_1"),
                ("overlay", "image", "filmstrip", "image"),
            ]:
                graph.connect(*wire)
            note = add("layout.comment", "attribution", position=(0, -500))
            note.name, note.node_size = "Original object-support controls - MIT", "large"
            note.description = (
                "Original MIT synthetic images, generator, expected sample counts and projects "
                "by OpenCVDataflowGUI contributors. No customer or third-party images. "
                "Native template matching finds three poses; a template region excludes the "
                "35-pixel background hole. Clear and touching parts have all 616 samples; "
                "covering ten columns leaves 406/616 supported samples. Visibility filters "
                "before downstream/clearance pick selection. Touching parts still have full "
                "appearance support, but clearance rejects both neighbors. Required three "
                "intentionally fails the count gate and prevents a pick while preserving "
                "current diagnostics. This appearance proxy is not true occlusion, collision "
                "or robotic grasp certification. See README and truth.json."
            )
            graph.save(HERE / f"{case}.json")
        finally:
            graph.close()
    (HERE / "truth.json").write_bytes(
        (json.dumps(truth, indent=2, allow_nan=False) + "\n").encode()
    )


if __name__ == "__main__":
    build()
