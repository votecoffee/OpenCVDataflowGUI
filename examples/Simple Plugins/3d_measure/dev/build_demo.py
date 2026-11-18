"""Known generated step, a raised surface feature and an absent measured surface."""

import argparse
import json
from pathlib import Path

import numpy as np
from opencv_simple_3d_measure import Measurement, Measurer, PlaneRegion, save_recipe, train


def surface(*, seed=1, bump=False, missing=False, scale=1):
    rng = np.random.default_rng(seed)
    floor_xy = rng.uniform([-15, -10], [-2, 10], (900, 2))
    floor = np.c_[floor_xy, rng.normal(0, .025, len(floor_xy))]
    top_xy = rng.uniform([2, -10], [15, 10], (800, 2))
    height = 5 + rng.normal(0, .025, len(top_xy))
    if bump:
        height[np.all(abs(top_xy - [7, 0]) < [2, 3], axis=1)] += 1.2
    top = np.c_[top_xy, height]
    return np.ascontiguousarray(floor if missing else np.r_[floor, top]) * scale


def configuration(scale=1):
    planes = (
        PlaneRegion("datum", tuple(np.array([-16, -11, -1]) * scale),
                    tuple(np.array([-1, 11, 1]) * scale), normal_hint=(0, 0, 1)),
        PlaneRegion("top", tuple(np.array([1, -11, 3]) * scale),
                    tuple(np.array([16, 11, 7]) * scale), normal_hint=(0, 0, 1)),
    )
    measurements = (
        Measurement("height", "step_height", "datum", "top", lower=4.9 * scale, upper=5.1 * scale),
        Measurement("angle", "plane_angle", "datum", "top", upper=1),
        Measurement("surface", "surface_range", "top", upper=.4 * scale),
        Measurement("probe", "point_plane_distance", "datum", point=(0, 0, 3 * scale)),
    )
    return planes, measurements


def build(output, *, host_projects=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    recipe = train(surface(), *configuration(), units="mm", frame="part")
    save_recipe(output / "measurement.s3m", recipe)
    (output / "configuration.json").write_text(
        json.dumps(recipe.configuration(), indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(output / "reference.npz", points=recipe.reference, units="mm", frame="part")
    for name, kwargs in (("good", {}), ("raised_feature", {"bump": True}),
                         ("missing_surface", {"missing": True})):
        scene = surface(seed=15, **kwargs)
        np.savez_compressed(output / f"{name}.npz", points=scene, units="mm", frame="part")
        result = Measurer(recipe).measure(scene, units="mm", frame="part")
        (output / f"{name}.json").write_text(
            json.dumps(result.report(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(name, result.passed, [(m["name"], m["value"], m["status"])
                                   for m in result.measurements])
        if host_projects:
            from host_examples import project

            project(output / name, recipe, scene,
                    description="Generated first-visible height surfaces; known 5 mm step. "
                                "Independent sample/noise; reference fit bands are frozen.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--host-projects", action="store_true")
    args = parser.parse_args()
    build(args.output, host_projects=args.host_projects)
