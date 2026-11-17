"""Original finite jitter controls with independent world pixels and sealed corrections."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.images import ImageArray
from opencv_dataflow.motion_trajectory import capture_trajectory
from opencv_dataflow.motion_trajectory_io import trajectory_bytes
from opencv_dataflow.payloads import FlowGroup, ValuePacket, frame
from opencv_dataflow.stabilization import _transform_payload

HERE = Path(__file__).resolve().parent
SIZE, MARGIN = [160, 112], 8
OFFSETS = [(-4, -2), (0, 0), (4, 2)]
CASES = {
    "project": {},
    "shifted_grids": {"grids": [1, -2, 2], "unfilled": "allow", "minimum_fill_fraction": 0.98},
    "changed_donor": {},
    "highdepth_mean": {"highdepth": True, "blend": "mean"},
    "no_consistent_neighbors": {"passed": False},
    "distance_limited": {"maximum_fill_distance": 1, "passed": False},
    "native_stabilization": {"native": True, "unfilled": "allow", "minimum_fill_fraction": 0.98},
}


def controls():
    width, height = SIZE
    y, x = np.mgrid[: height + MARGIN * 2, : width + MARGIN * 2]
    world = (24 + x // 2 + y // 3).astype(np.uint8)
    rng = np.random.default_rng(148)
    world[24:-24, 24:-24] = rng.integers(30, 190, world[24:-24, 24:-24].shape, np.uint8)
    frames = [
        world[MARGIN + dy : MARGIN + dy + height, MARGIN + dx : MARGIN + dx + width].copy()
        for dx, dy in OFFSETS
    ]
    images = {f"frame{i}": image for i, image in enumerate(frames)}
    images["world"] = world
    for key, image in list(images.items()):
        images[key + "16"] = np.dstack(
            (
                image.astype(np.uint16) * 257,
                image.astype(np.uint16) * 180,
                image.astype(np.uint16) * 120,
            )
        )
    changed = frames[1].copy()
    changed[50:58, 152:156] = 255
    images["changed"] = changed
    images["bright1"] = frames[1] + np.uint8(50)
    images["bright2"] = frames[2] + np.uint8(50)
    return images


def sources(case):
    if case == "changed_donor":
        return ["frame0", "changed", "frame2"]
    if case == "no_consistent_neighbors":
        return ["frame0", "bright1", "bright2"]
    return [f"frame{i}{'16' if case == 'highdepth_mean' else ''}" for i in range(3)]


def known_trajectory(images, names, grids):
    packets, transforms = [], []
    encoding = "bgr" if images[names[0]].ndim == 3 else "gray"
    for i, name in enumerate(names):
        context = {"spatial": frame(SIZE, "raw-capture")}
        image = ImageArray(images[name], encoding)
        packets.append(ValuePacket(image, context))
        dx, dy = OFFSETS[i]
        correction = np.array([[1.0, 0, dx + grids[i]], [0, 1.0, dy], [0, 0, 1.0]])
        spatial = frame(SIZE, "known-stabilized")
        spatial["to_root"][0][2] = -grids[i]
        transforms.append(
            ValuePacket(_transform_payload(correction, SIZE, context, spatial), context)
        )
    labels = tuple(f"capture{i}" for i in range(3))
    return capture_trajectory(
        FlowGroup(tuple(packets), labels),
        FlowGroup(tuple(transforms), labels),
        interpolation="nearest",
    )


def build():
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    images = controls()
    hashes = {}
    for name, image in images.items():
        ok, data = cv2.imencode(".png", image)
        assert ok
        (assets / f"{name}.png").write_bytes(data.tobytes())
        hashes[name + ".png"] = hashlib.sha256(data).hexdigest()
    registry = builtin_registry()
    for case, options in CASES.items():
        names = sources(case)
        graph = Graph()
        try:

            def add(kind, key, parameters=None, position=(0, 0)):
                block = graph.add(registry.create(kind, block_id=key, parameters=parameters or {}))
                block.position = list(position)
                return block

            for i, name in enumerate(names):
                add(
                    "input.file",
                    f"capture{i}",
                    {
                        "path": f"assets/{name}.png",
                        "path_base": "project",
                        "color_format": "bgr" if options.get("highdepth") else "gray",
                        "pixel_depth": "unchanged",
                    },
                    (0, i * 290),
                )
            add("flow.batch", "frames", {"size": 3}, (380, 0))
            for i in range(3):
                graph.connect(f"capture{i}", "image", "frames", f"item_{i + 1}")
            if options.get("native"):
                add(
                    "sequence.stabilize_batch",
                    "corrected",
                    {
                        "motion_model": "translation",
                        "radius_frames": 1,
                        "sigma_frames": 1.0,
                        "interpolation": "nearest",
                        "min_inliers": 8,
                    },
                    (760, 0),
                )
            else:
                value = known_trajectory(images, names, options.get("grids", [0, 0, 0]))
                raw = trajectory_bytes(value)
                record_path = assets / f"{case}.trajectory.json"
                record_path.write_bytes(raw)
                hashes[record_path.name] = hashlib.sha256(raw).hexdigest()
                add(
                    "sequence.load_trajectory",
                    "trajectory",
                    {
                        "path": f"assets/{record_path.name}",
                        "expected_sha256": hashlib.sha256(raw).hexdigest(),
                    },
                    (380, 400),
                )
                add("sequence.replay_trajectory", "corrected", position=(760, 0))
                graph.connect("trajectory", "trajectory", "corrected", "trajectory")
            graph.connect("frames", "group", "corrected", "frames")
            settings = {
                "interpolation": "nearest",
                "unfilled": options.get("unfilled", "fail"),
                "minimum_fill_fraction": options.get("minimum_fill_fraction", 0.0),
                "maximum_fill_distance": options.get("maximum_fill_distance", 32.0),
                "blend": options.get("blend", "nearest"),
            }
            fill = add("sequence.fill_stabilization_borders", "fill", settings, (1140, 0))
            fill.node_size = "large"
            fill.exposed_outputs.extend(["remaining_mask", "preview", "result_graphics"])
            graph.connect("corrected", "corrected", "fill", "frames")
            graph.connect("corrected", "coverage", "fill", "coverage")
            add("cv.overlay_result_graphics", "overlay", position=(1520, 0))
            graph.connect("fill", "display", "overlay", "image")
            graph.connect("fill", "result_graphics", "overlay", "graphics_1")
            add("output.filmstrip_image", "filmstrip", {"priority": 10}, (1900, 0))
            graph.connect("overlay", "image", "filmstrip", "image")
            note = add("layout.comment", "attribution", position=(0, -430))
            note.name, note.node_size = "Original jitter and border-fill controls - MIT", "large"
            note.description = (
                "Original MIT world pattern, captures, known trajectories, projects and controls "
                "by OpenCVDataflowGUI contributors; no customer or third-party images. Three "
                "160x112 captures have known world offsets (-4,-2), (0,0), (+4,+2). Sealed "
                "replay preserves source pixels and explicit measured support. The native "
                "stabilization case fits motion independently. Fill Image/Coverage remain "
                "current measured data; only Display uses neighboring captures. Filled pixels "
                "are review-only, not current defect evidence. Changed donors, distance limits, "
                "independent corrected grids and high-depth means have dedicated controls. "
                "Use README/truth.json for expected availability and numerical checks."
            )
            graph.save(HERE / f"{case}.json")
        finally:
            graph.close()
    truth = {
        "license": "MIT - original repository fixtures",
        "size": SIZE,
        "margin": MARGIN,
        "offsets": OFFSETS,
        "cases": CASES,
        "asset_sha256": hashes,
    }
    (HERE / "truth.json").write_bytes(
        (json.dumps(truth, indent=2, allow_nan=False) + "\n").encode()
    )


if __name__ == "__main__":
    build()
