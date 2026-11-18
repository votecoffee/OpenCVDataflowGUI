"""Export station reviews from completed public projects without recipe changes."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from opencv_simple_path_width.calibration import (
    PhysicalGap,
    PhysicalPathWidthResult,
    PhysicalStation,
)
from opencv_simple_path_width.dataflow import _calibration_for_image
from opencv_simple_path_width.measurement import Gap, PathWidthResult, Station
from opencv_simple_path_width.recipe import load_recipe
from opencv_simple_path_width.station_review import save_station_review, station_indices

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.processing import gray_image


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _numeric(outputs):
    physical = outputs["physical_measurements"]
    return json.dumps(
        [outputs["measurements"], None if physical is NO_VALUE else physical],
        sort_keys=True,
        allow_nan=False,
    ).encode()


def _completed_result(outputs):
    """Restore the already measured records; do not search for edge pairs again."""
    pixel = {key: value for key, value in outputs["measurements"].items() if key != "model"}
    pixel["stations"] = tuple(Station(**row) for row in pixel["stations"])
    pixel["gaps"] = tuple(Gap(**row) for row in pixel["gaps"])
    plane = outputs["physical_measurements"]
    if plane is not NO_VALUE:
        plane = {key: value for key, value in plane.items() if key != "model"}
        plane["stations"] = tuple(PhysicalStation(**row) for row in plane["stations"])
        plane["gaps"] = tuple(PhysicalGap(**row) for row in plane["gaps"])
        plane = PhysicalPathWidthResult(**plane)
    return PathWidthResult(**pixel, physical_measurements=None if plane is NO_VALUE else plane)


def _representative(indices):
    # Choose the middle of the longest consecutive run of a status, purely
    # for presentation. No acceptance thresholds or measured values change.
    runs = np.split(np.asarray(indices), np.where(np.diff(indices) != 1)[0] + 1)
    run = max(runs, key=len)
    return int(run[len(run) // 2])


def build(target):
    root = Path(__file__).resolve().parents[1]
    target = Path(target)
    target.mkdir(parents=True, exist_ok=False)
    receipt = {}
    for relative, frame_count in (("", 2), ("calibrated_bead", 3), ("weld_bead", 3)):
        folder = root / relative
        graph = Graph.load(folder / "project.json", builtin_registry())
        try:
            block = next(
                b for b in graph.blocks.values() if b.type_id == "inspect.simple_path_width"
            )
            recipe_path = folder / block.parameters["recipe_path"]
            recipe = load_recipe(recipe_path)
            for frame_index in range(frame_count):
                graph.run()
                if block.state not in ("pass", "fail"):
                    raise ValueError(block.error)
                outputs = block.last_result.outputs
                before = _numeric(outputs)
                source = block.last_inputs["image"]
                image = gray_image(source, (np.uint8, np.uint16, np.float32))
                input_digest = _digest(image.tobytes())
                calibration = None
                if "calibration" in block.last_inputs:
                    calibration = _calibration_for_image(
                        block.last_inputs["calibration"],
                        block.input_payloads.get("calibration", {}),
                        source,
                    )
                result = _completed_result(outputs)
                groups = ("out_of_tolerance", "missing", "ambiguous", "unobserved")
                choices = [(key, station_indices(result, key)) for key in groups]
                choices = [(key, indices) for key, indices in choices if indices]
                if not choices:
                    choices = [("measured", station_indices(result, "all"))]
                for group, indices in choices:
                    selected = _representative(indices)
                    name = f"{relative or 'generated'}_{frame_index}_{group}"
                    save_station_review(
                        target / name,
                        image,
                        result,
                        selected,
                        recipe.settings,
                        calibration=calibration,
                    )
                    receipt[name] = {
                        "project": (folder / "project.json").relative_to(root).as_posix(),
                        "frame_index": frame_index,
                        "station_number": selected + 1,
                        "group": group,
                        "path_passed": result.passed,
                        "station_count": len(result.stations),
                        "recipe_sha256": _digest(recipe_path.read_bytes()),
                        "input_pixels_sha256": input_digest,
                        "measurement_sha256": _digest(before),
                    }
                if before != _numeric(outputs) or input_digest != _digest(image.tobytes()):
                    raise AssertionError(
                        "Station review changed completed measurements or source pixels."
                    )
        finally:
            graph.close()
    (target / "receipt.json").write_bytes(
        (json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode()
    )
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New directory; existing directories are refused")
    receipt = build(parser.parse_args().output)
    print(f"Exported {len(receipt)} station reviews.")
