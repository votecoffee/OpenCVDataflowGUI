"""Camera-health replays derived from a credited real VisA PCB photograph."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.camera_health import save_reference, train_reference
from opencv_dataflow.camera_health.model import full_region
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "assets/showcase/visa-pcb2-0000.png"
CASES = {
    "healthy_moving_part": {"expected": "healthy"},
    "position_warning": {"shift_px": 2, "expected": "warning"},
    "position_alarm": {"shift_px": 5, "expected": "alarm"},
    "lighting_warning": {"gain": 0.85, "expected": "warning"},
    "lighting_alarm": {"gain": 0.7, "expected": "alarm"},
    "focus_warning": {"blur_sigma": 0.68, "expected": "warning"},
    "focus_alarm": {"blur_sigma": 1.5, "expected": "alarm"},
    "fiducials_healthy": {"mode": "fiducials", "expected": "healthy"},
    "fiducials_alarm": {"mode": "fiducials", "shift_px": 5, "expected": "alarm"},
    "insufficient_evidence": {"exclude_all": True, "expected": "alarm"},
}
CREDIT = (
    "Photo: Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang and Onkar Dabeer, "
    "SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation, "
    "ECCV 2022. VisA PCB2 normal 0000, CC BY 4.0. "
    "https://registry.opendata.aws/visa/ ; https://creativecommons.org/licenses/by/4.0/ . "
    "Reused 480x320 resized/letterboxed photo; synthetic colored parts, translations, "
    "gain changes and blur added by OpenCVDataflowGUI contributors. See SOURCES.md/truth.json. "
    "No customer images or equipment I/O. These are controlled software changes to a real "
    "photo, not measured camera/focus ground truth or physical acceptance."
)


def _row(name, box):
    return {"name": name, "region": {**full_region([480, 320]), "rectangle": box}}


def _png(path, image):
    ok, data = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(data.tobytes())


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    image = cv2.imread(str(SOURCE), cv2.IMREAD_COLOR)
    assert image.shape == (320, 480, 3)
    regions = {
        "lighting": [
            _row("Left light", [80, 45, 100, 65]),
            _row("Right light", [300, 45, 100, 65]),
        ],
        "focus": [
            _row("Left texture", [100, 130, 90, 75]),
            _row("Right texture", [290, 130, 90, 75]),
        ],
        "fiducials": [
            _row("Datum A", [190, 103, 49, 49]),
            _row("Datum B", [27, 15, 49, 49]),
        ],
    }
    history = [image.copy() for _ in range(3)]
    for i, (capture, x) in enumerate(zip(history, [15, 210, 400])):
        capture[260:295, x : x + 50] = [30, 40, 200]
        _png(assets / f"training_{i + 1}.png", capture)
    reference = train_reference(history, regions)
    reference.info["attribution"] = CREDIT
    digest = save_reference(reference, HERE / "reference.npz")
    registry = builtin_registry()
    for case, condition in CASES.items():
        current = image.copy()
        shift = condition.get("shift_px", 0)
        if shift:
            current = cv2.warpAffine(current, np.float32([[1, 0, shift], [0, 1, 0]]), (480, 320))
        if "gain" in condition:
            current = np.rint(current.astype(float) * condition["gain"]).astype(np.uint8)
        if "blur_sigma" in condition:
            current = cv2.GaussianBlur(current, (9, 9), condition["blur_sigma"])
        # New part position, deliberately distinct from the three training positions.
        current[265:295, 100:150] = [30, 40, 200]
        _png(assets / f"{case}.png", current)
        graph = Graph()
        graph.bind_project_dir(HERE)

        def add(kind, name, position, **parameters):
            block = graph.add(registry.create(kind, block_id=name, parameters=parameters))
            block.position = list(position)
            return block

        add(
            "input.file",
            "source",
            (0, 0),
            path=f"assets/{case}.png",
            path_base="project",
            color_format="bgr",
            pixel_depth="unchanged",
        )
        tool = add(
            "inspect.camera_health",
            "monitor",
            (660, 0),
            reference_path="reference.npz",
            reference_sha256=digest,
            position_method=condition.get("mode", "background"),
        )
        tool.name = "Camera health: " + case.replace("_", " ")
        graph.connect("source", "image", "monitor", "image")
        if condition.get("exclude_all"):
            add("region.shape", "occlusion", (330, 350), rectangle=[0, 0, 480, 320])
            graph.connect("source", "image", "occlusion", "image")
            graph.connect("occlusion", "mask", "monitor", "foreground_mask")
        note = add("layout.comment", "scope", (0, -400))
        note.name = "Photo attribution and expected result"
        note.node_size = "large"
        note.description = (
            CREDIT + f" EXPECT {condition['expected'].upper()}: {case.replace('_', ' ')}. "
            "All projects share the same three-capture frozen baseline and default limits. "
            "Warnings pass with an amber overlay; alarms fail with reasons. The full-mask "
            "example intentionally lacks evidence and must alarm."
        )
        graph.save(HERE / f"{case}.json")
        graph.close()
    truth = {
        "source": "VisA PCB2 normal 0000; see SOURCES.md",
        "license_images_and_normalized_reference": "CC BY 4.0",
        "source_sha256": _sha(SOURCE),
        "scope": "Controlled software changes to a real photo; no physical ground truth",
        "image_size": [480, 320],
        "training_part_xywh": [[x, 260, 50, 35] for x in [15, 210, 400]],
        "current_part_xywh": [100, 265, 50, 30],
        "part_bgr": [30, 40, 200],
        "translation": "integer horizontal pixels, linear warp, black border",
        "lighting": "round(original * gain), uint8 BGR, before adding moving part",
        "blur": "GaussianBlur 9x9, sigma in pixels, before adding moving part",
        "regions": regions,
        "cases": CASES,
        "reference_sha256": digest,
        "asset_sha256": {p.name: _sha(p) for p in sorted(assets.glob("*.png"))},
    }
    (HERE / "truth.json").write_bytes((json.dumps(truth, indent=2) + "\n").encode())


if __name__ == "__main__":
    build()
