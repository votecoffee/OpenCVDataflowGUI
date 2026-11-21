"""Build positive and weak capture-set audit projects from a known synthetic lens."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

SIZE = (640, 480)
CAMERA = np.array([[660.0, 0, 319.5], [0, 654.0, 239.5], [0, 0, 1]])
DISTORTION = np.array([-0.12, 0.035, 0.0007, -0.0005, 0])
COLUMNS, ROWS, SPACING = 8, 6, 14.0
LAYOUT = (
    (100, 96, 445, -0.24, 0.27),
    (542, 96, 450, 0.30, -0.20),
    (100, 382, 420, 0.22, 0.28),
    (548, 382, 440, -0.29, -0.24),
    (325, 102, 316, -0.34, 0.12),
    (310, 382, 333, 0.24, -0.17),
    (110, 234, 357, -0.06, 0.33),
    (532, 248, 369, 0.08, -0.35),
    (320, 238, 210, 0.28, 0.08),
    (341, 262, 244, -0.30, -0.25),
)
HOLDOUT = (418, 286, 285, 0.17, -0.28)
WEAK = tuple(
    (303 + 6 * (i % 3), 224 + 5 * i, 439 + 2 * i, 0.065 + 0.006 * i, -0.07 + 0.004 * i)
    for i in range(7)
)


def board_points():
    x, y = np.meshgrid(np.arange(COLUMNS), np.arange(ROWS))
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))) * SPACING
    return points - points.mean(axis=0)


def image(path, view, angle, rng):
    u, v, z, rx, ry = view
    t = np.array([(u - CAMERA[0, 2]) * z / CAMERA[0, 0], (v - CAMERA[1, 2]) * z / CAMERA[1, 1], z])
    r = np.array([rx, ry, angle])
    pixels = cv2.projectPoints(board_points(), r, t, CAMERA, DISTORTION)[0].reshape(-1, 2)
    noisy = pixels + rng.normal(0, 0.035, pixels.shape)
    radius = round(3 * CAMERA[0, 0] / z * 16)
    assert (noisy > 10).all() and (noisy < np.asarray(SIZE) - 10).all(), (view, noisy)
    pixels16 = np.rint(noisy * 16).astype(int)
    raster = np.full(SIZE[::-1], 240, np.uint8)
    for x, y in pixels16:
        cv2.circle(raster, (int(x), int(y)), radius, 17, -1, cv2.LINE_AA, shift=4)
    ok, data = cv2.imencode(".png", raster, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise OSError(f"Unable to encode {path}")
    path.write_bytes(data.tobytes())
    return {
        "path": path.name,
        "observed_truth_px": pixels.tolist(),
        "rotation": r.tolist(),
        "center": t.tolist(),
    }


def write_assets(output):
    assets = output / "assets"
    assets.mkdir(parents=True)
    rng = np.random.default_rng(712)
    truth = {"camera_matrix": CAMERA.tolist(), "distortion": DISTORTION.tolist()}
    for key, poses in (("primary", LAYOUT), ("repeat", LAYOUT), ("weak", WEAK)):
        truth[key] = []
        for i, pose in enumerate(poses, 1):
            if key == "repeat":
                pose = np.asarray(pose) + rng.uniform(-1, 1, 5) * [7, 7, 12, 0.075, 0.075]
            angle = (i % 5 - 2) * 0.065 if key != "weak" else 0.02 * i
            truth[key].append(image(assets / f"{key}-{i:02d}.png", pose, angle, rng))
    truth["holdout"] = image(assets / "held-out.png", HOLDOUT, -0.19, rng)
    (assets / "truth.json").write_text(
        json.dumps(truth, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n"
    )
    return truth


def save_project(output, truth, weak):
    registry, graph = builtin_registry(), Graph()

    def add(kind, key, label, x, y, **parameters):
        block = graph.add(registry.create(kind, block_id=key, parameters=parameters))
        block.name, block.position = label, [x, y]
        return block

    grid = dict(detector="circles_symmetric", columns=COLUMNS, rows=ROWS, spacing=SPACING)
    try:
        note = add("layout.comment", "note", "Audit lens calibration captures", 0, -260)
        note.node_size = "large"
        note.description = (
            "Original synthetic circle grids by OpenCVDataflowGUI contributors (MIT). "
            + (
                "The near-frontal views have low fit RMS and weak capture geometry."
                if weak
                else "Two independent capture sets fit the same lens and share a held-out view."
            )
        )
        key = "weak" if weak else "primary"
        fit_options = dict(
            **grid, solver="lu", fix_k3=True, path_base="project", max_rms_pixels=0.5
        )
        add(
            "calibration.lens",
            "fit",
            "Fit " + key,
            0,
            0,
            image_paths=[f"assets/{view['path']}" for view in truth[key]],
            **fit_options,
        )
        add("calibration.audit_capture", "audit", "Evaluate capture quality", 760, 0)
        graph.connect("fit", "fit_report", "audit", "fit_report")
        if not weak:
            add(
                "calibration.lens",
                "repeat",
                "Fit repeat captures",
                0,
                230,
                image_paths=[f"assets/{view['path']}" for view in truth["repeat"]],
                **fit_options,
            )
            add(
                "input.file",
                "independent",
                "Read held-out image",
                0,
                450,
                path="assets/held-out.png",
                path_base="project",
                color_format="gray",
            )
            graph.connect("repeat", "fit_report", "audit", "repeat_fit_report")
            for fit, audit_port, y in (("fit", "holdout", 15), ("repeat", "repeat_holdout", 280)):
                add("calibration.validate_lens", audit_port, "Validate " + fit, 380, y, **grid)
                graph.connect("independent", "image", audit_port, "image")
                graph.connect(fit, "calibration", audit_port, "calibration")
                graph.connect(audit_port, "report", "audit", audit_port)
        validated = validate_graph(graph)
        if not validated["valid"]:
            raise ValueError(validated["issues"])
        basename = "weak_capture" if weak else "calibration_capture_audit"
        graph.save(output / f"{basename}.json")
    finally:
        graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if options.output.exists() and any(options.output.iterdir()):
        parser.error("Use an empty directory")
    known = write_assets(options.output)
    for weak in (False, True):
        save_project(options.output, known, weak)
