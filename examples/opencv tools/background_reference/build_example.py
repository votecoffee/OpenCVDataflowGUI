"""Original MIT conveyor scenes and sealed references; no camera or model downloads."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.background_reference import make_reference, reference_bytes
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import frame

HERE = Path(__file__).resolve().parent
H, W = 80, 112
CAMERA, CALIBRATION = "inspection-camera", "grid-revision-1"
CASES = {
    "mog2": {"kind": "sequence.mog2"},
    "knn": {"kind": "sequence.knn_background"},
    "guarded_mog2": {"kind": "sequence.mog2", "guarded": True},
    "guarded_knn": {"kind": "sequence.knn_background", "guarded": True},
    "replay": {"kind": "sequence.knn_background", "runs": 5},
    "float_mog2": {"kind": "sequence.mog2", "float": True},
    "camera_mismatch": {"kind": "sequence.mog2", "state": "error"},
    "prepare_reference": {"prepare": True},
}


def fixtures():
    y, x = np.mgrid[:H, :W]
    clean = (45 + (x // 7 + y // 9) % 4 * 9).astype(np.uint8)
    clean[::16] = 32
    a, b = clean.copy(), clean.copy()
    a[28:48, 40:64] = 220
    b[28:48, 72:96] = 220
    empty = np.zeros((H, W), np.uint8)
    ma, mb = empty.copy(), empty.copy()
    ma[28:48, 40:64] = 255
    mb[28:48, 72:96] = 255
    return {
        "clean": clean,
        "part_a0": a,
        "part_a1": a,
        "part_b0": b,
        "part_b1": b,
        "empty0": clean,
        "mask_a": ma,
        "mask_b": mb,
        "mask_empty": empty,
    }


def build():
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    values = fixtures()
    for name, image in values.items():
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        (assets / (name + ".png")).write_bytes(encoded.tobytes())
    for name in ("clean", "part_a0"):
        ok, encoded = cv2.imencode(
            ".tif",
            values[name].astype(np.float32) + np.float32(0.125),
            [cv2.IMWRITE_TIFF_COMPRESSION, 1],
        )
        assert ok
        (assets / (name + ".tif")).write_bytes(encoded.tobytes())
    for floating in (False, True):
        image = (
            values["clean"].astype(np.float32) + np.float32(0.125) if floating else values["clean"]
        )
        reference = make_reference(
            ImageArray(image, "gray", {"spatial": frame([W, H], "reference-capture")}),
            CAMERA,
            CALIBRATION,
        )
        (
            assets / ("clean_float.background.json" if floating else "clean.background.json")
        ).write_bytes(reference_bytes(reference))
    registry = builtin_registry()
    for case, options in CASES.items():
        graph = Graph()
        try:
            prepare = options.get("prepare", False)
            floating = options.get("float", False)
            paths = (
                [
                    f"assets/{name}.png"
                    for name in ("part_a0", "part_a1", "part_b0", "part_b1", "empty0")
                ]
                if case == "replay"
                else [
                    "assets/clean.png"
                    if prepare
                    else "assets/part_a0.tif"
                    if floating
                    else "assets/part_a0.png"
                ]
            )
            source = graph.add(
                registry.create(
                    "input.file",
                    block_id="capture",
                    parameters={
                        "path": paths[0],
                        "source_mode": "files" if case == "replay" else "single",
                        "paths": paths if case == "replay" else [],
                        "path_base": "project",
                        "color_format": "gray",
                        "pixel_depth": "unchanged",
                        "repeat": "stop",
                    },
                )
            )
            source.position = [0, 0]
            if prepare:
                capture = graph.add(
                    registry.create(
                        "sequence.capture_background_reference",
                        block_id="reference",
                        parameters={"camera_id": CAMERA, "calibration_id": CALIBRATION},
                    )
                )
                capture.position = [450, 0]
                graph.connect("capture", "image", "reference", "image")
                save = graph.add(
                    registry.create(
                        "output.background_reference",
                        block_id="save",
                        parameters={"path": "capture.generated.json"},
                    )
                )
                save.position = [900, 0]
                graph.connect("reference", "reference", "save", "reference")
            else:
                filename = "clean_float.background.json" if floating else "clean.background.json"
                digest = hashlib.sha256((assets / filename).read_bytes()).hexdigest()
                block = graph.add(
                    registry.create(
                        options["kind"],
                        block_id="background",
                        parameters={
                            "reference_path": "assets/" + filename,
                            "reference_sha256": digest,
                            "reference_camera_id": "wrong-camera"
                            if case == "camera_mismatch"
                            else CAMERA,
                            "reference_calibration_id": CALIBRATION,
                            "reference_update": "guarded" if options.get("guarded") else "frozen",
                            "learning_rate": 0.5,
                            "detect_shadows": False,
                        },
                    )
                )
                block.position, block.node_size = [450, 0], "large"
                graph.connect("capture", "image", "background", "image")
                if options.get("guarded"):
                    region = graph.add(
                        registry.create(
                            "region.shape",
                            block_id="known",
                            parameters={"rectangle": [40, 28, 24, 20]},
                        )
                    )
                    region.position = [0, 350]
                    graph.connect("capture", "image", "known", "image")
                    graph.connect("known", "mask", "background", "known_foreground")
                masks = (
                    [
                        "assets/" + name + ".png"
                        for name in ("mask_a", "mask_a", "mask_b", "mask_b", "mask_empty")
                    ]
                    if case == "replay"
                    else ["assets/mask_a.png"]
                )
                truth = graph.add(
                    registry.create(
                        "input.file",
                        block_id="truth",
                        parameters={
                            "path": masks[0],
                            "source_mode": "files" if case == "replay" else "single",
                            "paths": masks if case == "replay" else [],
                            "path_base": "project",
                            "color_format": "gray",
                            "pixel_depth": "unchanged",
                            "repeat": "stop",
                        },
                    )
                )
                truth.position = [450, 450]
                verify = graph.add(
                    registry.create(
                        "inspect.image_quality",
                        block_id="verify",
                        parameters={
                            "metric": "max_absolute_error",
                            "check_threshold": True,
                            "threshold": 0,
                        },
                    )
                )
                verify.position = [900, 0]
                graph.connect("background", "mask", "verify", "image_a")
                graph.connect("truth", "image", "verify", "image_b")
            note = graph.add(registry.create("layout.comment", block_id="attribution"))
            note.name, note.position, note.node_size = (
                "Original background-reference fixtures - MIT",
                [0, -400],
                "large",
            )
            note.description = (
                "Original MIT conveyor images, references, generator and projects by "
                "OpenCVDataflowGUI contributors; no external or customer data. A saved clean "
                "reference is replayed before the first current part is classified. "
                "Camera/calibration/grid/encoding bindings and SHA256 prevent accidental "
                "mismatches. Frozen MOG2 retains its native model. Frozen KNN rebuilds its "
                "query model because this wheel can update at learningRate=0. Guarded updates "
                "sanitize known foreground with trusted background pixels. Replay checks five "
                "independent current captures, including a persistent part and an empty last "
                "frame. Float MOG2 keeps original [0,255] fractional values. Camera mismatch "
                "intentionally errors before native work. Prepare reference writes only an "
                "explicitly requested generated JSON; never captures a scene with known "
                "foreground. This is reference replay, not opaque native-state serialization or "
                "physical camera qualification."
            )
            graph.save(HERE / (case + ".json"))
        finally:
            graph.close()
    manifest = {
        "license": "MIT - original repository fixtures",
        "image_size": [W, H],
        "camera_id": CAMERA,
        "calibration_id": CALIBRATION,
        "replay_cases": ["part_a0", "part_a1", "part_b0", "part_b1", "empty0"],
        "cases": {
            name: {
                "state": options.get("state", "pass"),
                "runs": options.get("runs", 1),
                "prepare": options.get("prepare", False),
            }
            for name, options in CASES.items()
        },
        "asset_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(assets.iterdir())
            if path.is_file()
        },
    }
    (HERE / "truth.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode())


if __name__ == "__main__":
    build()
