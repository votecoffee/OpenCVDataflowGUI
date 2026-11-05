"""Deterministic, hardware-free stereo, RGB-D, PnP and structured-light examples."""

import cv2
import numpy as np

from .core import Block, Graph, Port, Result
from .geometry3d_common import camera_record, pose_record
from .images import DepthArray
from .payloads import frame
from .processing import finite, integer
from .stereo3d import StereoRig


class StereoDemo(Block):
    type_id, title, category = "input.stereo_demo", "Synthetic stereo / RGB-D", "Input"
    block_type, subtype = "Input", "Depth and RGB-D"
    outputs = (
        Port("left"),
        Port("right"),
        Port("depth", "depth"),
        Port("camera", "data"),
        Port("rig", "data"),
    )
    defaults = {
        "width": 320,
        "height": 180,
        "disparity": 8,
        "focal_length": 200.0,
        "baseline_m": 0.12,
        "seed": 3,
    }

    def process(self, inputs):
        p = self.parameters
        w, h = (integer(p[k], k, 64, 2048) for k in ("width", "height"))
        disparity = integer(p["disparity"], "Disparity", 1, w // 4)
        focal = finite(p["focal_length"], "Focal length", 1)
        baseline = finite(p["baseline_m"], "Baseline", 0.0001)
        rng = np.random.default_rng(integer(p["seed"], "Seed", 0, 2**32 - 1))
        left = cv2.GaussianBlur(rng.integers(0, 256, (h, w), np.uint8), (3, 3), 0)
        right = np.zeros_like(left)
        right[:, :-disparity] = left[:, disparity:]
        k = [[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]]
        a, b = (
            camera_record(k, [], (w, h), frame=name) for name in ("left_camera", "right_camera")
        )
        transform = np.eye(4)
        transform[0, 3] = -baseline
        rig = (
            StereoRig()
            .process(
                {
                    "left_camera": a,
                    "right_camera": b,
                    "transform": pose_record(transform, a["frame"], b["frame"], "m"),
                }
            )
            .outputs["rig"]
        )
        z = focal * baseline / disparity
        depth = DepthArray(np.full((h, w), z, np.float32), "m", 0, 2 * z)
        depth[:10, :10] = np.nan
        left_frame = {"spatial": frame((w, h), f"{self.id}:left")}
        return Result(
            {"left": left, "right": right, "depth": depth, "camera": a, "rig": rig},
            payloads={
                "left": left_frame,
                "depth": left_frame,
                "right": {"spatial": frame((w, h), f"{self.id}:right")},
            },
            message=f"Known {disparity} px disparity, Z = {z:g} m",
        )


def make_graph(registry, specs, wires):
    graph = Graph()
    for key, kind, position, parameters in specs:
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.position = position
    for source, output, target, input_name in wires:
        graph.connect(source, output, target, input_name)
    return graph


def stereo_cloud_graph(registry):
    graph = make_graph(
        registry,
        [
            ("pair", "input.stereo_demo", [0, 0], {}),
            ("rectify", "3d.stereo_rectify", [330, 0], {}),
            ("disparity", "3d.disparity", [660, 0], {"num_disparities": 32}),
            ("cloud", "3d.reproject", [990, 0], {"sample_step": 3, "max_depth": 5.0}),
            ("measure", "3d.cloud_measure", [1320, 0], {}),
            ("view", "3d.cloud_render", [1320, 360], {}),
        ],
        [
            ("pair", "left", "rectify", "left"),
            ("pair", "right", "rectify", "right"),
            ("pair", "rig", "rectify", "rig"),
            ("rectify", "left", "disparity", "left"),
            ("rectify", "right", "disparity", "right"),
            ("disparity", "disparity", "cloud", "disparity"),
            ("rectify", "rectification", "cloud", "rectification"),
            ("rectify", "left", "cloud", "image"),
            ("cloud", "cloud", "measure", "cloud"),
            ("cloud", "cloud", "view", "cloud"),
        ],
    )
    graph.blocks["disparity"].preview_output = "preview"
    graph.blocks["cloud"].preview_output = "point cloud"
    return graph


def rgbd_cloud_graph(registry):
    return make_graph(
        registry,
        [
            ("pair", "input.stereo_demo", [0, 0], {}),
            ("project", "3d.depth_cloud", [330, 0], {"sample_step": 4}),
            (
                "filter",
                "3d.cloud_filter",
                [660, 0],
                {"minimum": [-1, -1, 2.9], "maximum": [1, 1, 3.1]},
            ),
            ("sample", "3d.cloud_voxel", [990, 0], {"voxel_size": 0.08, "mode": "centroid"}),
            ("normals", "3d.cloud_normals", [1320, 0], {"neighbors": 10}),
            ("measure", "3d.cloud_measure", [1650, 0], {}),
            ("view", "3d.cloud_render", [1650, 350], {}),
        ],
        [
            ("pair", "depth", "project", "depth"),
            ("pair", "left", "project", "image"),
            ("pair", "camera", "project", "camera"),
            ("project", "cloud", "filter", "cloud"),
            ("filter", "cloud", "sample", "cloud"),
            ("sample", "cloud", "normals", "cloud"),
            ("normals", "cloud", "measure", "cloud"),
            ("normals", "cloud", "view", "cloud"),
        ],
    )


def pose_3d_graph(registry):
    xyz = (
        np.random.default_rng(2026)
        .uniform([-0.4, -0.3, 2.0], [0.4, 0.3, 4.0], size=(25, 3))
        .round(6)
    )
    return make_graph(
        registry,
        [
            ("xyz", "3d.points", [0, 0], {"points": xyz.tolist(), "frame": "object"}),
            ("left_cam", "3d.camera", [0, 360], {"frame": "left_camera"}),
            ("right_cam", "3d.camera", [0, 720], {"frame": "right_camera"}),
            ("left_pose", "3d.pose", [330, 360], {"target_frame": "left_camera"}),
            (
                "right_pose",
                "3d.pose",
                [330, 720],
                {"target_frame": "right_camera", "translation": [-0.15, 0, 0]},
            ),
            ("left_pixels", "3d.project", [660, 0], {}),
            ("right_pixels", "3d.project", [660, 420], {}),
            ("pnp", "3d.pnp", [990, 0], {}),
            ("triangulate", "3d.triangulate", [990, 420], {"frame": "object"}),
            ("view", "3d.cloud_render", [1320, 420], {}),
        ],
        [
            ("xyz", "points", "left_pixels", "points"),
            ("xyz", "points", "right_pixels", "points"),
            ("left_cam", "camera", "left_pixels", "camera"),
            ("right_cam", "camera", "right_pixels", "camera"),
            ("left_pose", "transform", "left_pixels", "transform"),
            ("right_pose", "transform", "right_pixels", "transform"),
            ("xyz", "points", "pnp", "object_points"),
            ("right_pixels", "points", "pnp", "image_points"),
            ("right_cam", "camera", "pnp", "camera"),
            ("left_pixels", "points", "triangulate", "points_left"),
            ("right_pixels", "points", "triangulate", "points_right"),
            ("left_pixels", "projection", "triangulate", "projection_left"),
            ("right_pixels", "projection", "triangulate", "projection_right"),
            ("triangulate", "cloud", "view", "cloud"),
        ],
    )


def structured_light_graph(registry):
    return make_graph(
        registry,
        [
            ("patterns", "3d.graycode", [0, 0], {"width": 160, "height": 120}),
            (
                "decode",
                "3d.graycode_decode",
                [360, 0],
                {"width": 160, "height": 120, "sample_step": 20},
            ),
            ("first", "flow.select_entry", [360, 400], {"index": 0}),
        ],
        [("patterns", name, "decode", name) for name in ("patterns", "white", "black")]
        + [("patterns", "patterns", "first", "group")],
    )


def tsdf_graph(registry):
    return make_graph(
        registry,
        [
            ("pair", "input.stereo_demo", [0, 0], {}),
            ("pose", "3d.pose", [0, 380], {"source_frame": "left_camera", "target_frame": "world"}),
            ("volume", "3d.tsdf_fuse", [360, 0], {"volume_origin_m": [-0.64, -0.64, 2.5]}),
            ("view", "3d.cloud_render", [720, 0], {}),
        ],
        [
            ("pair", "depth", "volume", "depth"),
            ("pair", "camera", "volume", "camera"),
            ("pose", "transform", "volume", "poses"),
            ("volume", "cloud", "view", "cloud"),
        ],
    )
