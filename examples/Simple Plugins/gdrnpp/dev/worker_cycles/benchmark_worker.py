"""Actual cold/warm worker round trips, calibrated connected arrays and optional graphics.

One cold call and three warm calls per case/graphics mode. No neural pose labels
or fitted thresholds enter this replay; all runtime settings come from the recipe.
"""

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from worker_metrics import snapshot

from opencv_simple_gdrnpp.bridge import WorkerClient
from opencv_simple_gdrnpp.workflow import load_capture


def run(recipe, models, extension, output, *, worker_python=sys.executable, repeats=3):
    if output.exists():
        raise FileExistsError("Choose a fresh worker-cycle receipt")
    if not 1 <= repeats <= 7:
        raise ValueError("Use 1..7 measured warm requests")
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(recipe.read_bytes()).hexdigest()
    external = load_capture(extension / "reviews/scene000006_frame000002/capture.json")
    scenarios = [
        ("retained_reference", None),
        ("connected_independent_capture", external),
    ]
    rows, poses = [], {}
    for label, capture in scenarios:
        for graphics in (False, True):
            data = dict(
                base=dict(kind="recipe", path=str(recipe.resolve()), sha256=digest),
                models_dir=str(models.resolve()),
                graphics=graphics,
            )
            arrays = {}
            if capture is not None:
                data["current"] = dict(
                    kind="arrays",
                    metadata=dict(
                        capture_id=capture.capture_id,
                        model_frame=capture.model_frame,
                        camera_frame=capture.frame.frame,
                        detections=[
                            dict(identifier=d.identifier, box=d.box, source=d.source)
                            for d in capture.detections
                        ],
                    ),
                )
                arrays = dict(
                    rgb=capture.frame.rgb,
                    depth=capture.frame.depth,
                    camera=capture.frame.camera,
                    mask=capture.frame.mask,
                )
            samples = []
            with WorkerClient(python=worker_python) as client:
                for index in range(repeats + 1):
                    started = time.perf_counter()
                    metadata, result = client.call("preview", data, arrays)
                    elapsed = time.perf_counter() - started
                    report = metadata["report"]
                    candidate = report["candidates"][0]
                    pose = candidate["pose"]["source_to_target"]
                    if label in poses:
                        np.testing.assert_array_equal(pose, poses[label])
                    poses[label] = pose
                    assert report["accepted_candidates"] == 1
                    assert set(result) == ({"overlay", "points", "colors"} if graphics else set())
                    assert report["worker"]["model_load_count"] == 1
                    assert report["worker"]["model_cache_hit"] == (index > 0)
                    if index:
                        assert report["worker"]["recipe_cache_hit"]
                    sample = dict(
                        cold=index == 0,
                        client_round_trip_seconds=elapsed,
                        report_seconds=report["seconds"],
                        worker_profile=report.get("execution_profile"),
                        request_timing=metadata.get("request_timing"),
                        model_cache_hit=report["worker"]["model_cache_hit"],
                        recipe_cache_hit=report["worker"]["recipe_cache_hit"],
                        owned_processes=snapshot(client.process.pid),
                        returned_array_bytes=sum(value.nbytes for value in result.values()),
                        geometry_backend=report["geometry_execution_backend"],
                        worker_implementation_sha256=report["worker"].get("implementation_sha256"),
                        fit=candidate["fit"],
                        pose=pose,
                    )
                    samples.append(sample)
                    print(
                        json.dumps(
                            dict(case=label, graphics=graphics, cold=index == 0, seconds=elapsed)
                        ),
                        flush=True,
                    )
            rows.append(
                dict(
                    case=label,
                    graphics=graphics,
                    samples=samples,
                    cold_seconds=samples[0]["client_round_trip_seconds"],
                    warm_median_seconds=statistics.median(
                        sample["client_round_trip_seconds"] for sample in samples[1:]
                    ),
                )
            )
    value = dict(
        python=platform.python_version(),
        recipe_sha256=digest,
        warm_requests=repeats,
        scenarios=rows,
        poses_identical_across_repeats_and_graphics=True,
        scope=(
            "Actual isolated worker calls include process/model startup on cold requests, "
            "client packet writes, connected input transfer/validation, inference, ICP/fit, "
            "requested graphics, result transfer/read and private request cleanup. "
            "No camera acquisition, GUI composition, robot or hardware acceptance."
        ),
        attribution="T-LESS, Hodan et al., WACV 2017, CC BY 4.0",
    )
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("recipe", "models", "extension", "new_receipt"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--worker-python", default=sys.executable)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    run(
        args.recipe,
        args.models,
        args.extension,
        args.new_receipt,
        worker_python=args.worker_python,
        repeats=args.repeats,
    )
