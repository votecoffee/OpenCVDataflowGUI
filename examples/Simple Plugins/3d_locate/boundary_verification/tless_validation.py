"""Adapters for the frozen T-LESS validation subset; poses are evaluation only."""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from opencv_simple_3d_locate import Camera, DepthScene, ValidationScene
from opencv_simple_3d_locate.model import rigid


def discrete_symmetries(source, object_id):
    metadata = json.loads((Path(source) / "models_cad/models_info.json").read_bytes())[
        str(object_id)
    ]
    if metadata.get("symmetries_continuous"):
        raise ValueError("This pose-tolerance validation requires finite discrete symmetries.")
    return tuple(
        rigid(np.asarray(value).reshape(4, 4)) for value in metadata.get("symmetries_discrete", [])
    )


@dataclass(frozen=True)
class Capture:
    image: np.ndarray
    points: np.ndarray
    camera: Camera
    depth: DepthScene
    truth: tuple
    symmetries: tuple
    scene_id: int
    frame_id: int
    isolated: bool

    @property
    def name(self):
        kind = "isolated" if self.isolated else "full"
        return f"scene{self.scene_id:06}_frame{self.frame_id:06}_{kind}"

    def validation(self):
        return ValidationScene(
            self.name,
            self.points,
            "mm",
            "camera",
            self.truth,
            self.camera,
            self.symmetries,
            self.depth,
        )


def read_capture(source, scene_id, frame_id, object_id, *, isolated=False):
    source = Path(source)
    folder = source / f"test_primesense/{scene_id:06}"
    metadata = json.loads((folder / "scene_camera.json").read_bytes())[str(frame_id)]
    entries = json.loads((folder / "scene_gt.json").read_bytes())[str(frame_id)]
    image = cv2.imread(str(folder / f"rgb/{frame_id:06}.png"))
    depth = cv2.imread(str(folder / f"depth/{frame_id:06}.png"), cv2.IMREAD_UNCHANGED)
    if image is None or depth is None:
        raise ValueError("Missing selected RGB/depth capture.")
    camera = Camera(np.asarray(metadata["cam_K"]).reshape(3, 3), (image.shape[1], image.shape[0]))
    full = camera.backproject(depth, scale=metadata["depth_scale"])
    truth, ids = [], []
    for i, entry in enumerate(entries):
        if entry["obj_id"] == object_id:
            pose = np.eye(4)
            pose[:3, :3] = np.asarray(entry["cam_R_m2c"]).reshape(3, 3)
            pose[:3, 3] = entry["cam_t_m2c"]
            truth.append(rigid(pose))
            ids.append(i)
    if isolated:
        selected = np.zeros(depth.shape, bool)
        for i in ids:
            mask = cv2.imread(str(folder / f"mask_visib/{frame_id:06}_{i:06}.png"), 0)
            if mask is None or mask.shape != depth.shape:
                raise ValueError("Missing or mismatched supplied visible mask.")
            selected |= mask != 0
        points = camera.backproject(np.where(selected, depth, 0), scale=metadata["depth_scale"])
    else:
        points = full.copy()
    return Capture(
        image,
        points,
        camera,
        DepthScene(full, "mm", "camera"),
        tuple(truth),
        discrete_symmetries(source, object_id),
        scene_id,
        frame_id,
        isolated,
    )


def validation_captures(source, selection):
    captures = []
    for frame in selection["positive_frames"]:
        for isolated in (True, False):
            capture = read_capture(
                source,
                selection["positive_scene"],
                frame,
                selection["object_id"],
                isolated=isolated,
            )
            if not capture.truth:
                raise ValueError("Positive capture has no annotated target.")
            captures.append(capture)
    for frame in selection["negative_frames"]:
        capture = read_capture(source, selection["negative_scene"], frame, selection["object_id"])
        if capture.truth:
            raise ValueError("Target-absent validation capture contains the target.")
        captures.append(capture)
    return tuple(captures)
