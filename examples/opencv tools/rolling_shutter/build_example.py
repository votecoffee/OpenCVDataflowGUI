"""Original MIT reference coupon and known row-distorted captures, including FFV1."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
H, W = 240, 320
AMPLITUDES = (0, 1, -1, 0.7, -0.5, 1.2)
COEFFICIENTS = [[1.3, -0.25], [2.1, 0], [-0.7, 0.4], [0.9, 0]]
CASES = {
    "project": {"source": "wobbled", "reference": "reference", "truth": "reference"},
    "video": {"source": "clip", "reference": "reference", "truth": "reference", "runs": 6},
    "highdepth": {
        "source": "wobbled16",
        "reference": "reference16",
        "truth": "reference16",
        "color": "bgr",
    },
    "moving_part": {
        "source": "part_wobbled",
        "reference": "reference",
        "truth": "part_truth",
        "fit_region": [0, 0, 120, 240],
    },
    "no_texture": {
        "source": "flat",
        "reference": "flat",
        "truth": "flat",
        "state": "fail",
        "reason": "insufficient_tracks",
    },
    "row_gap": {
        "source": "wobbled",
        "reference": "reference",
        "truth": "reference",
        "fit_region": [0, 96, 320, 48],
        "state": "fail",
        "reason": "insufficient_row_coverage",
    },
}


def displacement(y, amplitude=1):
    u = 2 * y / (H - 1) - 1
    return amplitude * np.stack(
        (1.3 + 2.1 * u - 0.7 * u**2 + 0.9 * u**3, 0.4 * u**2 - 0.25), axis=-1
    )


def distort(image, amplitude=1):
    """Invert monotone scanline displacement with bisection and array sampling."""
    if amplitude == 0:
        return image.copy()
    yy, xx = np.mgrid[:H, :W].astype(float)
    low, high = np.full((H, W), -16.0), np.full((H, W), H + 16.0)
    for _ in range(40):
        middle = (low + high) / 2
        before = middle + displacement(middle, amplitude)[..., 1] < yy
        low = np.where(before, middle, low)
        high = np.where(before, high, middle)
    y = (low + high) / 2
    x = xx - displacement(y, amplitude)[..., 0]
    left, top = np.floor(x).astype(int), np.floor(y).astype(int)
    output = np.zeros(image.shape)
    for dy in (0, 1):
        for dx in (0, 1):
            xi, yi = left + dx, top + dy
            good = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            weight = (x - left if dx else 1 - (x - left)) * (y - top if dy else 1 - (y - top))
            output[good] += image[yi[good], xi[good]] * (
                weight[good, None] if image.ndim == 3 else weight[good]
            )
    return output


def fixtures():
    values = cv2.GaussianBlur(np.random.default_rng(169).uniform(0.15, 0.85, (H, W)), (5, 5), 0.8)
    for x in (12, 60, 156, 300):
        values[:, x : x + 2] = 0.9
    for y in (9, 60, 120, 180, 230):
        values[y : y + 2] = 0.1
    part = values.copy()
    part[76:145, 180:230] = 0.95
    color = np.dstack((values, values * 0.8, values * 0.6))
    raw = {
        "reference": values,
        "wobbled": distort(values),
        "reference16": color,
        "wobbled16": distort(color),
        "part_truth": part,
        "part_wobbled": distort(part),
        "flat": np.full((H, W), 0.5),
    }
    images = {
        name: np.rint(image * (65535 if name.endswith("16") else 255)).astype(
            np.uint16 if name.endswith("16") else np.uint8
        )
        for name, image in raw.items()
    }
    clip = [np.rint(distort(values, amplitude) * 255).astype(np.uint8) for amplitude in AMPLITUDES]
    return images, clip


def build():
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    images, frames = fixtures()
    for name, image in images.items():
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        (assets / (name + ".png")).write_bytes(encoded.tobytes())
    writer = cv2.VideoWriter(
        str(assets / "row_wobble.avi"),
        cv2.CAP_FFMPEG,
        cv2.VideoWriter_fourcc(*"FFV1"),
        10,
        (W, H),
        False,
    )
    assert writer.isOpened()
    try:
        for image in frames:
            writer.write(image)
    finally:
        writer.release()
    registry = builtin_registry()
    for case, settings in CASES.items():
        graph = Graph()
        try:
            color = settings.get("color", "gray")
            for identifier, asset in (
                ("capture", settings["source"]),
                ("reference", settings["reference"]),
                ("truth", settings["truth"]),
            ):
                video = asset == "clip"
                params = {
                    "path": "assets/row_wobble.avi" if video else f"assets/{asset}.png",
                    "path_base": "project",
                    "color_format": color,
                    "pixel_depth": "unchanged",
                }
                if video:
                    params.update(end_frame=len(AMPLITUDES), repeat="stop")
                block = graph.add(
                    registry.create(
                        "input.video_file" if video else "input.file",
                        block_id=identifier,
                        parameters=params,
                    )
                )
                block.position = [0, {"capture": 0, "reference": 300, "truth": 600}[identifier]]
            fit = graph.add(registry.create("inspect.scanline_motion", block_id="motion"))
            fit.position, fit.node_size = [450, 0], "large"
            graph.connect("capture", "image", "motion", "image")
            graph.connect("reference", "image", "motion", "reference")
            if "fit_region" in settings:
                fit.exposed_inputs.append("reference_region")
                region = graph.add(
                    registry.create(
                        "region.shape",
                        block_id="fit_region",
                        parameters={"rectangle": settings["fit_region"]},
                    )
                )
                region.position = [0, 900]
                graph.connect("reference", "image", "fit_region", "image")
                graph.connect("fit_region", "region", "motion", "reference_region")
            warp = graph.add(
                registry.create(
                    "cv.warp_flow",
                    block_id="correct",
                    parameters={"consistency": "off", "minimum_coverage": 0.85},
                )
            )
            warp.position, warp.node_size = [900, 0], "large"
            warp.exposed_inputs += ["target", "flow_support"]
            graph.connect("capture", "image", "correct", "image")
            graph.connect("reference", "image", "correct", "target")
            graph.connect("motion", "flow", "correct", "flow")
            graph.connect("motion", "support", "correct", "flow_support")
            for identifier, source in (("before", "capture"), ("verify", "correct")):
                tool = graph.add(
                    registry.create(
                        "inspect.image_quality",
                        block_id=identifier,
                        parameters={
                            "metric": "rmse",
                            "check_threshold": identifier == "verify",
                            "threshold": 4000 if case == "highdepth" else 16,
                            "mask_alignment": "pixel_grid",
                        },
                    )
                )
                tool.position = [1350, 0 if identifier == "verify" else 350]
                graph.connect(source, "image", identifier, "image_a")
                graph.connect("truth", "image", identifier, "image_b")
                graph.connect("correct", "coverage", identifier, "mask")
            note = graph.add(registry.create("layout.comment", block_id="attribution"))
            note.name, note.position, note.node_size = (
                "Original rolling-shutter fixtures - MIT",
                [0, -400],
                "large",
            )
            note.description = (
                "Original MIT coupon, video, generator and projects by OpenCVDataflowGUI "
                "contributors. "
                "No external or customer images. Known row-dependent horizontal and vertical cubic "
                "translations distort a sharp reference; independent bisection and array sampling "
                "create the captures. Native corner tracking estimates motion, separate tracks "
                "validate it, and the existing dense-flow warper corrects current pixels with "
                "explicit "
                "coverage and coordinate maps. Video has six known amplitudes at 10 fps. "
                "Moving-part "
                "fitting uses only the left static strip; the current part remains in the output. "
                "No-texture and row-gap intentionally fail without emitting a motion field. "
                "This is a row-translation baseline, not physical exposure calibration, general "
                "projective correction or defect qualification. OpenCV tracking reference: "
                "https://docs.opencv.org/5.0/main_modules/video_track.html"
            )
            graph.save(HERE / (case + ".json"))
        finally:
            graph.close()
    truth = {
        "license": "MIT - original repository fixtures",
        "image_size": [W, H],
        "coefficients_xy": COEFFICIENTS,
        "row_coordinate": "u=2*y/(height-1)-1; ascending powers",
        "video_amplitudes": list(AMPLITUDES),
        "video_period_ms": 100,
        "part_xywh": [180, 76, 50, 69],
        "cases": {
            name: {
                "state": settings.get("state", "pass"),
                "reason": settings.get("reason"),
                "runs": settings.get("runs", 1),
            }
            for name, settings in CASES.items()
        },
        "asset_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(assets.iterdir())
            if path.is_file()
        },
    }
    (HERE / "truth.json").write_bytes((json.dumps(truth, indent=2) + "\n").encode())


if __name__ == "__main__":
    build()
