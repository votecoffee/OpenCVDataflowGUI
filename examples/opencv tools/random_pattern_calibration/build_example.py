"""Build the saved graph and two calibrated synthetic photographs of the target."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

EXAMPLE = "random_pattern_calibration.json"
IMAGE_SIZE = (640, 480)
CAMERA = np.array([[670.0, 0, 319.5], [0, 680.0, 239.5], [0, 0, 1]])
TARGET = {"width": 240.0, "height": 160.0, "units": "mm", "seed": 13, "raster_width": 900}
VIEWS = (
    ((-0.19, 0.30, 0.17), (-145.0, -70.0, 282.0), (215, 210, 375, 348)),
    ((0.25, -0.27, -0.13), (-98.0, -89.0, 245.0), (302, 154, 420, 304)),
)


def rotation(angles):
    ax, ay, az = angles
    sx, sy, sz = np.sin(angles)
    cx, cy, cz = np.cos((ax, ay, az))
    return (
        np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    )


def project(xyz, r, t):
    points = (np.asarray(xyz) @ r.T + t) @ CAMERA.T
    return points[:, :2] / points[:, 2:3]


def render_view(target_image, angles, translation, cover):
    r = rotation(angles)
    units_per_px = np.array([TARGET["width"], TARGET["height"]]) / target_image.shape[::-1]
    plane_to_camera = np.column_stack((r[:, :2], translation))
    raster_to_plane = np.diag([*units_per_px, 1])
    raster_to_plane[:2, 2] = units_per_px / 2
    homography = CAMERA @ plane_to_camera @ raster_to_plane
    homography /= homography[2, 2]
    photo = cv2.warpPerspective(target_image, homography, IMAGE_SIZE, borderValue=110)
    x0, y0, x1, y1 = cover
    cv2.rectangle(photo, (x0, y0), (x1, y1), 72, -1)
    cv2.line(photo, (x0, y0), (x1, y0), 95, 2, cv2.LINE_AA)

    x, y = np.meshgrid(np.linspace(16, 224, 24), np.linspace(16, 144, 19))
    candidates = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    uv = project(candidates, r, translation)
    hidden = (
        (uv[:, 0] > x0 + 12) & (uv[:, 0] < x1 - 12) & (uv[:, 1] > y0 + 12) & (uv[:, 1] < y1 - 12)
    )
    return photo, {
        "rotation": r.tolist(),
        "translation_mm": translation,
        "occlusion_xyxy": cover,
        "reference_to_image": homography.tolist(),
        "holdout_xyz_mm": candidates[hidden][::3].tolist(),
        "holdout_pixels": uv[hidden][::3].tolist(),
    }


def png(path, pixels):
    ok, encoded = cv2.imencode(".png", pixels, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise OSError(f"Cannot encode {path}")
    path.write_bytes(encoded.tobytes())


def build(output):
    assets = output / "assets"
    assets.mkdir(parents=True)
    registry, graph = builtin_registry(), Graph()
    filenames = [f"assets/covered-{i}.png" for i in range(1, len(VIEWS) + 1)]

    def add(kind, key, name, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = name, [x, y]
        return block

    try:
        note = add(
            "layout.comment", "note", "Localize a partially visible calibration target", 0, -260
        )
        note.node_size = "large"
        note.description = (
            "Original synthetic images by OpenCVDataflowGUI contributors (MIT). "
            "Run both photographs. The patterned target is cropped and covered in each view. "
            "The detector matches visible landmarks and the pose uses the camera intrinsics."
        )
        generator = add(
            "calibration.generate_random_pattern", "pattern", "Known printed target", 0, 0, **TARGET
        )
        result = generator.execute({})
        if result is None or not result.passed:
            raise RuntimeError(generator.error)
        target_image = result.outputs["image"]
        png(assets / "target.png", target_image)
        truth = {"camera_matrix": CAMERA.tolist(), "views": []}
        for file, settings in zip(filenames, VIEWS, strict=True):
            photo, record = render_view(target_image, *settings)
            png(output / file, photo)
            truth["views"].append({"filename": Path(file).name, **record})
        (assets / "truth.json").write_text(
            json.dumps(truth, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        add(
            "input.file",
            "views",
            "Two cropped, covered photographs",
            0,
            240,
            source_mode="files",
            paths=filenames,
            path_base="project",
            color_format="gray",
        )
        add(
            "3d.camera",
            "camera",
            "Known pinhole intrinsics",
            390,
            366,
            fx=CAMERA[0, 0],
            fy=CAMERA[1, 1],
            cx=CAMERA[0, 2],
            cy=CAMERA[1, 2],
            width=IMAGE_SIZE[0],
            height=IMAGE_SIZE[1],
            frame="calibration_camera",
            units="mm",
        )
        add("calibration.detect_random_pattern", "match", "Observe visible landmarks", 380, 0)
        add(
            "3d.planar_pose_hypotheses",
            "pose",
            "Estimate target-to-camera pose",
            800,
            0,
            source_frame="reference_target",
            max_rms_px=1.5,
            normal_length=40.0,
        )
        for block, port, consumer, terminal in (
            ("pattern", "image", "match", "reference"),
            ("pattern", "target", "match", "target"),
            ("views", "image", "match", "image"),
            ("camera", "camera", "pose", "camera"),
            ("match", "image", "pose", "image"),
            ("match", "image_points", "pose", "image_points"),
            ("match", "object_points", "pose", "object_points"),
        ):
            graph.connect(block, port, consumer, terminal)
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise ValueError(checked["issues"])
        graph.save(output / EXAMPLE)
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output / EXAMPLE).exists():
        parser.error("Choose an output directory without an existing project")
    build(args.output)
