"""Explicit native camera refinement using original, shared known-camera captures."""

import json
from pathlib import Path

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
SOURCES = HERE.parent / "rotating_panorama"
K = [[420.0, 0, 240.0], [0, 420.0, 180.0], [0, 0, 1]]
INITIAL_K = [[390.0, 0, 240.0], [0, 390.0, 180.0], [0, 0, 1]]
CASES = {
    "reprojection": {},
    "ray": {"bundle_adjustment": "ray"},
    "fixed_intrinsics": {"initial_intrinsics": [K] * 3, "refine_focal": False},
    "initial_only": {"bundle_adjustment": "none", "initial_intrinsics": [K] * 3},
    "refine_prior": {"initial_intrinsics": [INITIAL_K] * 3},
    "validation_rejected": {
        "initial_intrinsics": [[[200.0, 0, 240.0], [0, 200.0, 180.0], [0, 0, 1]]] * 3,
        "refine_focal": False,
    },
    "disconnected": {},
    "wave_corrected": {
        "initial_intrinsics": [K] * 3,
        "refine_focal": False,
        "wave_correction": "horizontal",
    },
    "insufficient_validation": {"minimum_inliers": 500},
}
FAILED = {"validation_rejected", "disconnected", "insufficient_validation"}


def build():
    registry = builtin_registry()
    for name, settings in CASES.items():
        source = {"disconnected": "disconnected", "fixed_intrinsics": "uint16_plane"}.get(
            name, "unordered"
        )
        graph = Graph.load(SOURCES / f"{source}.json", registry)
        try:
            block = graph.blocks["panorama"]
            block.parameters.update(camera_solver="explicit", minimum_inlier_ratio=0.7, **settings)
            block.validate_parameters()
            note = graph.blocks["attribution"]
            note.name = "Original camera refinement controls - MIT"
            note.description = (
                "Original MIT fixtures by OpenCVDataflowGUI contributors, shared from the "
                "rotating_panorama example. No customer/third-party images. Known 480x360 "
                "pinhole views, focal 420 pixels, yaw -36/0/+36 degrees. Explicit native "
                "homography estimation and selected bundle adjustment consume only fit "
                "keypoints. Seeded rounded-center partition reserves keypoints before native "
                "matching; validation does not feed back into fitting. Intrinsics can be "
                "fixed or refined; ray adjustment requires centered square work intrinsics. "
                "Wrong fixed intrinsics and inadequate reserved support fail normally. "
                "Shared-image observations are correlated and do not replace external "
                "calibration data. See README and ../rotating_panorama/truth.json."
            )
            path = HERE / f"{name}.json"
            graph.save(path)
            raw = path.read_bytes()
            saved = json.loads(raw)
            for record in saved["blocks"]:
                if record["type"] == "input.file":
                    record["parameters"]["path"] = record["parameters"]["path"].replace("\\", "/")
            eol = b"\r\n" if b"\r\n" in raw else b"\n"
            path.write_bytes(
                (json.dumps(saved, indent=2, allow_nan=False) + "\n").encode().replace(b"\n", eol)
            )
        finally:
            graph.close()


if __name__ == "__main__":
    build()
