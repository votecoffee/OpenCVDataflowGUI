"""Compare intervals with held-out photographs from the OpenCV chessboard sample."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import stdev

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.calibration import find_grid, grid_settings

# OpenCV 5.0.0 samples/data. Images stay in the caller's local sample directory.
PINNED = {
    1: "9621899098adffa1440c8264606bb7be535ccfd9ad126ae8e0bd9b0c5a5b8676",
    2: "052e700bbcc865c112958ec2ea105cb1bce4deca50ac8d6efae13adca9df7a2c",
    3: "c85b9426c3d8fcf161e1e727356105091009c3fd0b1b1a65bc2da91fea2724d2",
    6: "1cb164db61b3bd52f58a7e9340bb27c5e0278895bc80b6f292fba5cf9bccc4a1",
    7: "d2211fe9f971646b6ff88ed93a4f0d3f443741eb2894095cba15caade7c8c6e4",
    8: "f7a0737894812a28243af5b409c5689f4fadfe7610958b8ca5fc79aec2f2cd1b",
    9: "753939bc62d093c001d9118e989bef2b887f1b2f9ac03603df915a8b85aa81d3",
    11: "e736ace3d515c94c547009c3c10d818c81506ed7059cb31123749b9cc3f94ee8",
    12: "04bdd78b3b507fe0551aa7062a3ffd94846a147ee2bea7fb38082c720bdbe231",
    13: "2637fb6dd5e557f4fadb06b353a453eb0b9223f4671f6f572f815302bc4a70b3",
    14: "d4c032971dd7b859f9d6aed5090c9888bd8039b82e0604bee5bfe05444fce834",
}
FIT = (1, 2, 6, 7, 11, 14)
HOLDOUT = (3, 8, 9, 12, 13)
ENDS = np.array([18, 26])


def checked(block, inputs):
    result = block.execute(inputs)
    if not result or not result.passed:
        raise AssertionError((block.name, block.error or result.message))
    return result.outputs


def compare(directory):
    paths = {index: directory / f"left{index:02}.jpg" for index in PINNED}
    for index, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != PINNED[index]:
            raise ValueError(f"Source digest differs for {path.name}")
    registry = builtin_registry()
    fit = registry.create(
        "calibration.lens",
        parameters={
            "columns": 9,
            "rows": 6,
            "spacing": 1.0,
            "units": "unitless",
            "image_paths": [str(paths[index]) for index in FIT],
            "solver": "lu",
            "accuracy": True,
        },
    )
    lens = checked(fit, {})
    camera = checked(
        registry.create("3d.camera", parameters={"units": "unitless"}),
        {"calibration": lens["calibration"]},
    )["camera"]
    grid, board = grid_settings(fit.parameters)
    pose_block = registry.create("3d.pnp", parameters={"ransac": False, "max_rms": 1.0})
    measure = registry.create(
        "calibration.measure_uncertainty", parameters={"pixel_stddev_px": 0.25}
    )
    other = np.setdiff1d(np.arange(len(board)), ENDS)
    nominal = float(np.linalg.norm(board[ENDS[1]] - board[ENDS[0]]))
    values = []
    for index in HOLDOUT:
        path = paths[index]
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
        observed, _ = find_grid(image, grid, fit.parameters)
        pose = checked(
            pose_block,
            {
                "camera": camera,
                "object_points": board[other].tolist(),
                "image_points": observed[other].tolist(),
            },
        )
        report = checked(
            measure,
            {
                "camera": camera,
                "plane_to_camera": pose["pose"],
                "image_points": observed[ENDS].tolist(),
                "image": image,
            },
        )["report"]
        low, high = report["interval"]
        values.append(
            {
                "capture": path.name,
                "distance": report["nominal_distance"],
                "interval": report["interval"],
                "sigma": report["standard_uncertainty"],
                "covers_nominal_spacing": low <= nominal <= high,
                "pose_fit_rms_px": pose["rms"],
            }
        )
    return {
        "units": "nominal chessboard square pitch",
        "fit_images": [paths[i].name for i in FIT],
        "nominal_distance": nominal,
        "pixel_sigma_assumption_px": 0.25,
        "covers_nominal_spacing_count": sum(view["covers_nominal_spacing"] for view in values),
        "observed_distance_stddev": stdev(view["distance"] for view in values),
        "mean_predicted_stddev": sum(view["sigma"] for view in values) / len(values),
        "holdout": values,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("opencv_sample_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.opencv_sample_directory), indent=2))
