"""Build independent-camera pose fixtures using analytic camera projections."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

LEFT = {
    "lens": "pinhole",
    "width": 800,
    "height": 540,
    "fx": 685.0,
    "fy": 674.0,
    "cx": 395.0,
    "cy": 272.0,
    "distortion": [-0.085, 0.019, 0.001, -0.0006, 0.0],
}
RIGHT = {
    "pinhole": {"fx": 812.0, "fy": 766.0, "distortion": [-0.11, 0.027, -0.0005, 0.0009, 0.0]},
    "fisheye": {"fx": 351.0, "fy": 342.0, "distortion": [-0.05, 0.017, -0.0015, 0.0003]},
    "omnidir": {"fx": 465.0, "fy": 452.0, "distortion": [-0.04, 0.01, 0.0006, -0.0005], "xi": 0.78},
}
CENTER = np.array([0.245, 0.038, -0.027])
ANGLES = (0.039, -0.063, 0.027)
CORRUPT = np.array([2, 6, 13, 18, 25, 37, 40, 48, 61, 75, 83, 91])


def right_camera(lens):
    return {"lens": lens, "width": 930, "height": 670, "cx": 466.0, "cy": 328.0, **RIGHT[lens]}


def rotation():
    ax, ay, az = ANGLES
    sx, sy, sz = np.sin(ANGLES)
    cx, cy, cz = np.cos((ax, ay, az))
    return (
        np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    )


def pixels(xyz, camera):
    xy = xyz[:, :2]
    lens = camera["lens"]
    coeff = camera["distortion"]
    if lens == "fisheye":
        radius = np.linalg.norm(xy, axis=1)
        angles = np.arctan2(radius, xyz[:, 2])
        warped = angles * np.polynomial.polynomial.polyval(angles**2, [1, *coeff])
        xy = xy * np.divide(warped, radius, out=np.zeros_like(radius), where=radius > 0)[:, None]
    else:
        w = xyz[:, 2].copy()
        if lens == "omnidir":
            w += camera["xi"] * np.linalg.norm(xyz, axis=1)
        x, y = (xy / w[:, None]).T
        k1, k2, p1, p2 = coeff[:4]
        r2 = x * x + y * y
        gain = 1 + r2 * (k1 + k2 * r2)
        xy = np.column_stack(
            (
                x * gain + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                y * gain + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y,
            )
        )
    return xy * [camera["fx"], camera["fy"]] + [camera["cx"], camera["cy"]]


def scene(lens, center=CENTER):
    r = rotation()
    translation = -r @ center
    right = right_camera(lens)
    rng = np.random.default_rng(40188)
    pool = rng.uniform([-1.1, -0.7, 1.15], [1.1, 0.7, 4.5], (170, 3))
    first, second = pixels(pool, LEFT), pixels(pool @ r.T + translation, right)
    valid = np.ones(len(pool), dtype=bool)
    for xy, model in ((first, LEFT), (second, right)):
        size = np.array([model["width"], model["height"]])
        valid &= ((xy >= 12) & (xy < size - 12)).all(axis=1)
    xyz = pool[valid][:104]
    first, second = first[valid][:104], second[valid][:104]
    assert len(xyz) == 104
    a = first[:96] + rng.normal(0, 0.016, (96, 2))
    b = second[:96] + rng.normal(0, 0.018, (96, 2))
    b[CORRUPT] = b[np.roll(CORRUPT, 3)]
    invalid = [9] if lens == "fisheye" else []
    if invalid:
        b[invalid[0]] = [4, 5]
    return {
        "rotation": r.tolist(),
        "translation_m": translation.tolist(),
        "baseline_m": float(np.linalg.norm(center)),
        "right_center_in_left": center.tolist(),
        "known_xyz_m": xyz[:96].tolist(),
        "left": a.tolist(),
        "right": b.tolist(),
        "corrupt_indices": CORRUPT.tolist(),
        "invalid_ray_indices": invalid,
        "left_truth": first[:96].tolist(),
        "right_truth": second[:96].tolist(),
        "inliers": np.setdiff1d(np.arange(96), [*CORRUPT, *invalid]).tolist(),
        "held_xyz_m": xyz[96:].tolist(),
        "held_left": first[96:].tolist(),
        "held_right": second[96:].tolist(),
    }


def png(path, xy, camera):
    canvas = np.full((camera["height"], camera["width"], 3), (55, 45, 35), np.uint8)
    for row, (x, y) in enumerate(np.rint(xy).astype(int)):
        cv2.circle(canvas, (int(x), int(y)), 3, (215, 222, 230), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(row),
            (int(x) + 6, int(y) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (173, 194, 219),
            1,
            cv2.LINE_AA,
        )
    ok, image = cv2.imencode(".png", canvas, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise OSError(path)
    path.write_bytes(image.tobytes())


def save_graph(output, lens, known):
    registry, graph = builtin_registry(), Graph()

    def add(kind, key, label, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = label, [x, y]
        return block

    try:
        note = add("layout.comment", "note", "Relative pose from independent views", 0, -245)
        note.node_size = "large"
        note.description = (
            "Original synthetic scene by OpenCVDataflowGUI contributors (MIT). "
            "Compare independently calibrated images with different focal lengths, dimensions "
            "and distortion. The point rows include intentionally mismatched pairs."
        )
        for side, y, model in (("left", 0, LEFT), ("right", 285, right_camera(lens))):
            add(
                "3d.camera",
                f"camera_{side}",
                f"{side.title()} {model['lens']} camera",
                0,
                y,
                frame=side + "_camera",
                units="m",
                **model,
            )
            add(
                "ml.vector_data",
                f"points_{side}",
                f"{side.title()} observed XY",
                275,
                y,
                values=known[side],
            )
            add(
                "input.file",
                f"{side}_image",
                f"{side.title()} numbered landmarks",
                530,
                y,
                path_base="project",
                path=f"assets/{lens}_{side}.png",
            )
        add(
            "3d.independent_relative_pose",
            "pose",
            "Estimate left-to-right pose",
            870,
            105,
            baseline_length=known["baseline_m"] if lens == "fisheye" else 0,
            baseline_units="m",
            maximum_reprojection_px=0.9,
            minimum_inliers=74,
        )
        for side in ("left", "right"):
            graph.connect(f"camera_{side}", "camera", "pose", f"camera_{side}")
            graph.connect(f"points_{side}", "vectors", "pose", f"points_{side}")
            graph.connect(f"{side}_image", "image", "pose", f"{side}_image")
        check = validate_graph(graph)
        if not check["valid"]:
            raise ValueError(check["issues"])
        graph.save(output / f"relative_pose_{lens}.json")
    finally:
        graph.close()


def build(output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty output directory")
    assets = output / "assets"
    assets.mkdir(parents=True)
    truth = {}
    for lens in ("pinhole", "fisheye"):
        known = truth[lens] = scene(lens)
        for side, model in (("left", LEFT), ("right", right_camera(lens))):
            png(assets / f"{lens}_{side}.png", known[f"{side}_truth"], model)
        save_graph(output, lens, known)
    (assets / "truth.json").write_text(
        json.dumps(truth, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
