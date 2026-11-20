"""Build a fisheye-to-omnidirectional region transfer with analytic plane truth."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

SOURCE = dict(width=160, height=110, fx=96, fy=93, cx=79.5, cy=54.5)
TARGET = dict(width=144, height=100, fx=122, fy=120, cx=71.5, cy=49.5, xi=0.55)
PATCH = [42, 26, 73, 51]
OCCLUDER = (slice(43, 55), slice(65, 76))
MISSING = (slice(36, 40), slice(59, 64))
YAW = 0.14
TRANSLATION = [-70.0, 13.0, 40.0]
SOURCE_Z = 1250


def plane_assets():
    source = np.full((SOURCE["height"], SOURCE["width"]), SOURCE_Z, np.uint16)
    source[35:41, 53:60] = 0
    y, x = np.indices((TARGET["height"], TARGET["width"]))
    mx, my = (x - TARGET["cx"]) / TARGET["fx"], (y - TARGET["cy"]) / TARGET["fy"]
    radius2 = mx**2 + my**2
    xi = TARGET["xi"]
    factor = (xi + np.sqrt(1 + (1 - xi**2) * radius2)) / (1 + radius2)
    ax = mx * factor / (factor - xi)
    normal = [np.sin(YAW), 0, np.cos(YAW)]
    offset = SOURCE_Z + np.dot(normal, TRANSLATION)
    target = np.rint(offset / (normal[0] * ax + normal[2])).astype(np.uint16)
    target[OCCLUDER] = 700
    target[MISSING] = 0
    return source, target


def save_graph(path):
    graph, registry = Graph(), builtin_registry()

    def add(kind, key, position, **settings):
        block = graph.add(registry.create(kind, block_id=key, parameters=settings))
        block.position = position
        return block

    try:
        comment = add("layout.comment", "notes", [0, -240])
        comment.name, comment.node_size = "Transfer a depth-backed patch", "large"
        comment.description = (
            "Original synthetic depth planes and camera models by OpenCVDataflowGUI contributors "
            "(MIT). A fisheye source patch transfers to an omnidirectional target. "
            "A nearer target object occludes part of the patch and a hole has no observed depth."
        )
        for i, (key, filename) in enumerate((("source", "source.png"), ("target", "target.png"))):
            node = add(
                "input.depth_file",
                key,
                [0, i * 240],
                path=f"assets/{filename}",
                path_base="project",
                units="mm",
                units_per_sample=1.0,
                display_min=600,
                display_max=1600,
            )
            node.name = f"{key.title()} depth"
            node.ticket_settings = {"mode": "fixed", "value": "wide-angle-plane"}
        source_cam = add(
            "3d.camera",
            "source_camera",
            [0, 470],
            lens="fisheye",
            distortion=[0.0] * 4,
            frame="source_optical",
            **SOURCE,
        )
        source_cam.name = "Fisheye source camera"
        target_cam = add(
            "3d.camera",
            "target_camera",
            [0, 650],
            lens="omnidir",
            distortion=[0.0] * 4,
            frame="target_optical",
            **TARGET,
        )
        target_cam.name = "Omnidirectional target camera"
        pose = add(
            "3d.pose",
            "pose",
            [315, 455],
            rvec=[0.0, YAW, 0.0],
            units="mm",
            source_frame="source_optical",
            target_frame="target_optical",
            translation=TRANSLATION,
        )
        region = add("region.shape", "patch", [315, 30], rectangle=PATCH)
        region.name = "Select physical patch"
        transfer = add(
            "3d.reproject_depth_region",
            "transfer",
            [685, 70],
            visibility_policy="require_agreement",
            reference_tolerance_m=0.012,
            minimum_retained_fraction=0.55,
        )
        transfer.preview_output = "Coverage"
        measure = add("region.measure", "measure", [1100, 255])
        measure.name = "Measure transferred support"
        graph.connect("source", "depth", region.id, "image")
        for src, out, port in (
            ("source", "depth", "source_depth"),
            ("target", "depth", "target_depth"),
            (source_cam.id, "camera", "source_camera"),
            (target_cam.id, "camera", "target_camera"),
            (pose.id, "transform", "transform"),
            (region.id, "region", "region"),
        ):
            graph.connect(src, out, transfer.id, port)
        graph.connect("target", "depth", measure.id, "image")
        graph.connect(transfer.id, "region", measure.id, "region")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise ValueError(checked["issues"])
        graph.save(path)
    finally:
        graph.close()


def build(output):
    assets = output / "assets"
    project = output / "depth_region_projection.json"
    if assets.exists() or project.exists():
        raise FileExistsError("Choose a new output directory to preserve the example")
    assets.mkdir(parents=True)
    for name, data in zip(("source.png", "target.png"), plane_assets(), strict=True):
        ok, raw = cv2.imencode(".png", data)
        if not ok:
            raise ValueError(f"Could not encode {name}")
        (assets / name).write_bytes(raw.tobytes())
    save_graph(project)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
