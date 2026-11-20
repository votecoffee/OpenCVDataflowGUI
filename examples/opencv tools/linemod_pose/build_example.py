"""Build a calibrated two-view LINEMOD pose example with analytic depth."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

WIDTH, HEIGHT = 421, 333
SIZE = (WIDTH, HEIGHT)
K = np.array([[535.0, 0, 210.0], [0, 529.0, 166.0], [0, 0, 1]])
OBJECT, CAMERA = "marker_plate", "registered_optical"
PLATE = np.array(
    [[-0.106, -0.077], [0.064, -0.077], [0.106, -0.042], [0.106, 0.077], [-0.106, 0.077]]
)
TEXTURE_SIZE = (321, 233)
TEXTURE_CORNERS = np.float32([[0, 0], [320, 0], [320, 232], [0, 232]])
OBJECT_CORNERS = np.float64(
    [[-0.106, -0.077, 0], [0.106, -0.077, 0], [0.106, 0.077, 0], [-0.106, 0.077, 0]]
)


def rotation(angle):
    tilt = cv2.Rodrigues(np.array([0.11, -0.18, 0.0]))[0]
    spin = cv2.Rodrigues(np.array([0.0, 0.0, np.deg2rad(angle)]))[0]
    return spin @ tilt


def render(angle, translation, gain=1.0, bias=0):
    r = rotation(angle)
    t = np.array(translation)
    rvec = cv2.Rodrigues(r)[0]
    projected = cv2.projectPoints(OBJECT_CORNERS, rvec, t, K, None)[0].reshape(-1, 2)
    homography = cv2.getPerspectiveTransform(TEXTURE_CORNERS, projected.astype(np.float32))
    surface = np.full((TEXTURE_SIZE[1], TEXTURE_SIZE[0], 3), (125, 166, 201), np.uint8)
    rng = np.random.default_rng(19)
    for _ in range(27):
        point = tuple(rng.integers([33, 28], [282, 200]).tolist())
        color = rng.integers(15, 242, size=3).tolist()
        cv2.circle(surface, point, int(rng.integers(5, 18)), color, -1, cv2.LINE_AA)
    cv2.rectangle(surface, (39, 42), (92, 105), (190, 48, 37), 3, cv2.LINE_AA)
    cv2.line(surface, (145, 183), (240, 135), (66, 31, 130), 5, cv2.LINE_AA)
    cv2.putText(surface, "P7", (117, 79), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (45, 55, 44), 3)
    outline = cv2.projectPoints(np.column_stack((PLATE, np.zeros(len(PLATE)))), rvec, t, K, None)[
        0
    ].reshape(-1, 2)
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    cv2.fillConvexPoly(mask, np.rint(outline).astype(np.int32), 255)
    image = cv2.warpPerspective(surface, homography, SIZE, borderMode=cv2.BORDER_REPLICATE)
    image[mask == 0] = (35, 39, 44)
    image = np.clip(np.rint(gain * image + bias), 0, 255).astype(np.uint8)

    # Intersect each calibrated camera ray with the known plane.
    v, u = np.mgrid[:HEIGHT, :WIDTH]
    rays = np.stack(
        ((u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones(mask.shape)), axis=-1
    )
    normal = r[:, 2]
    z = (normal @ t) / (rays @ normal)
    depth = np.where(mask != 0, np.rint(1000 * z), 2450).astype(np.uint16)
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = r, t
    return image, depth, mask, transform


def save_graph(path, truth):
    graph, registry = Graph(), builtin_registry()

    def add(kind, name, position, **config):
        block = graph.add(registry.create(kind, block_id=name, parameters=config))
        block.position = position
        return block

    def connect_capture(label, block, row, *, teach=False):
        image = add(
            "input.file",
            f"{label}_image",
            [0, row],
            path=f"assets/{label}_color.png",
            path_base="project",
            color_format="bgr",
        )
        image.ticket_settings = {"mode": "fixed", "value": label}
        depth = add(
            "input.depth_file",
            f"{label}_depth",
            [0, row + 165],
            path=f"assets/{label}_depth.png",
            path_base="project",
            units="m",
            units_per_sample=0.001,
        )
        depth.ticket_settings = image.ticket_settings.copy()
        graph.connect(image.id, "image", block, "image")
        graph.connect(f"{label}_depth", "depth", block, "depth")
        graph.connect("camera", "camera", block, "camera")
        if teach:
            mask = add(
                "input.file",
                f"{label}_mask",
                [0, row + 330],
                path=f"assets/{label}_mask.png",
                path_base="project",
                color_format="gray",
            )
            mask.ticket_settings = image.ticket_settings.copy()
            learned = np.array(truth[label])
            add(
                "3d.pose",
                f"{label}_pose",
                [0, row + 495],
                source_frame=OBJECT,
                target_frame=CAMERA,
                translation=learned[:3, 3].tolist(),
                rvec=cv2.Rodrigues(learned[:3, :3])[0].ravel().tolist(),
            )
            graph.connect(mask.id, "image", block, "mask")
            graph.connect(f"{label}_pose", "transform", block, "pose")

    try:
        note = add("layout.comment", "notes", [0, -230])
        note.name, note.node_size = "Find a part from calibrated RGB-D views", "large"
        note.description = (
            "Synthetic plate, captures and calibration by OpenCVDataflowGUI contributors (MIT). "
            "Two isolated views teach the part. The search frame changes position and brightness."
        )
        add(
            "3d.camera",
            "camera",
            [0, 1640],
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
            width=WIDTH,
            height=HEIGHT,
            frame=CAMERA,
        )
        for i, label in enumerate(("straight", "quarter_turn")):
            name = f"train_{label}"
            add(
                "3d.train_linemod_pose",
                name,
                [430 + i * 300, 260 + i * 700],
                view_name=label,
                alignment="pixel_grid",
            )
            connect_capture(label, name, i * 660, teach=True)
            if i:
                graph.connect("train_straight", "model", name, "previous_model")
        match = add("3d.match_linemod_pose", "match", [1040, 1170], alignment="pixel_grid")
        match.node_size = "large"
        match.preview_output = "RGB-D candidates and pose"
        connect_capture("search", "match", 1320)
        graph.connect("train_quarter_turn", "model", "match", "model")
        checked = validate_graph(graph)
        if not checked["valid"]:
            raise ValueError(checked["issues"])
        graph.save(path)
    finally:
        graph.close()


def build(output):
    assets = output / "assets"
    if assets.exists() or (output / "linemod_pose.json").exists():
        raise FileExistsError("Choose a new directory to preserve the fixture")
    assets.mkdir(parents=True)
    truth = {}
    for label, angle, t, gain, bias in (
        ("straight", 0, (0, 0, 0.66), 1.0, 0),
        ("quarter_turn", 90, (0, 0, 0.66), 1.0, 0),
        ("search", 90, (0.034, -0.025, 0.667), 0.85, 14),
    ):
        image, depth, mask, transform = render(angle, t, gain, bias)
        truth[label] = transform.tolist()
        captures = {"color": image, "depth": depth}
        if label != "search":
            captures["mask"] = mask
        for suffix, raster in captures.items():
            ok, png = cv2.imencode(".png", raster)
            if not ok:
                raise ValueError(f"Could not encode {label}_{suffix}")
            (assets / f"{label}_{suffix}.png").write_bytes(png.tobytes())
    (assets / "ground_truth.json").write_bytes((json.dumps(truth, indent=2) + "\n").encode())
    save_graph(output / "linemod_pose.json", truth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().output)
