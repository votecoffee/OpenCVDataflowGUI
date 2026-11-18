"""Compare real first-image requests with explicit resource-only preparation.

Each arm owns one persistent worker, then processes two independent-capture
requests on that SAME worker. Preparation never runs a forward pass. Orders
alternate; model loading is moved to readiness, not claimed as eliminated.
"""

import argparse
import hashlib
import json
import platform
import statistics
import sys
from pathlib import Path

import numpy as np
from worker_metrics import snapshot

from opencv_simple_gdrnpp.bridge import WorkerClient
from opencv_simple_gdrnpp.workflow import load_capture


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(recipe, models, extension, output, *, worker_python=sys.executable, repeats=3):
    if output.exists():
        raise FileExistsError("Choose a fresh preparation receipt")
    if not 1 <= repeats <= 5:
        raise ValueError("Use 1..5 alternating pairs per graphics mode")
    capture_path = extension / "reviews/scene000006_frame000002/capture.json"
    capture = load_capture(capture_path)
    base = dict(kind="recipe", path=str(recipe.resolve()), sha256=digest(recipe))
    resources = dict(base=base, models_dir=str(models.resolve()))
    current = dict(
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
    rows, first_candidate, graphics_hashes, implementations = [], None, {}, set()
    for graphics in (False, True):
        for pair in range(repeats):
            for preloaded in (False, True) if pair % 2 == 0 else (True, False):
                with WorkerClient(python=worker_python) as client:
                    prepared = None
                    if preloaded:
                        prepared, returned = client.call("prepare", resources)
                        assert returned == {} and set(prepared) == {"prepared", "request_timing"}
                        state = prepared["prepared"]
                        assert state["ready"] and not state["inspection_performed"]
                        assert state["worker"]["inference_request_count"] == 0
                        assert state["worker"]["model_load_count"] == 1
                        assert not state["worker"]["model_cache_hit"]
                    samples = []
                    for index in range(2):
                        metadata, returned = client.call(
                            "preview",
                            {**resources, "current": current, "graphics": graphics},
                            arrays,
                        )
                        report = metadata["report"]
                        info, candidate = report["worker"], report["candidates"][0]
                        assert report["capture_id"] == capture.capture_id
                        assert info["model_load_count"] == 1
                        assert info["inference_request_count"] == index + 1
                        assert info["model_cache_hit"] == (preloaded or index > 0)
                        assert set(returned) == (
                            {"overlay", "points", "colors"} if graphics else set()
                        )
                        if prepared:
                            assert info["session"] == prepared["prepared"]["worker"]["session"]
                        stable = dict(
                            pose=candidate["pose"],
                            fit=candidate["fit"],
                            accepted_count=report["accepted_candidates"],
                        )
                        if first_candidate is None:
                            first_candidate = stable
                        assert stable == first_candidate
                        for name, array in returned.items():
                            sha = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
                            assert name not in graphics_hashes or graphics_hashes[name] == sha
                            graphics_hashes[name] = sha
                        implementations.add(info["implementation_sha256"])
                        assert list(client.root.iterdir()) == []
                        samples.append(
                            dict(
                                first=index == 0,
                                request_timing=metadata["request_timing"],
                                execution_profile=report["execution_profile"],
                                worker=info,
                                owned_processes=snapshot(client.process.pid),
                            )
                        )
                    row = dict(
                        pair=pair,
                        graphics=graphics,
                        preloaded=preloaded,
                        preparation=prepared,
                        inspections=samples,
                    )
                    rows.append(row)
                    print(
                        json.dumps(
                            dict(
                                pair=pair,
                                graphics=graphics,
                                preloaded=preloaded,
                                preparation_seconds=(
                                    prepared["request_timing"]["client_total_seconds"]
                                    if prepared
                                    else 0
                                ),
                                first_image_seconds=samples[0]["request_timing"][
                                    "client_total_seconds"
                                ],
                            )
                        ),
                        flush=True,
                    )
    assert len(implementations) == 1, "Worker source changed during the measurements"
    summary = []
    for graphics in (False, True):
        selected = [r for r in rows if r["graphics"] == graphics]
        plain = [r for r in selected if not r["preloaded"]]
        ready = [r for r in selected if r["preloaded"]]

        def first(r):
            return r["inspections"][0]["request_timing"]["client_total_seconds"]

        def prep(r):
            return r["preparation"]["request_timing"]["client_total_seconds"]

        summary.append(
            dict(
                graphics=graphics,
                unprepared_first_image_median_seconds=statistics.median(map(first, plain)),
                preparation_median_seconds=statistics.median(map(prep, ready)),
                prepared_first_image_median_seconds=statistics.median(map(first, ready)),
                prepared_startup_plus_first_image_median_seconds=statistics.median(
                    prep(r) + first(r) for r in ready
                ),
            )
        )
    value = dict(
        schema="simple_gdrnpp_preparation_benchmark/v1",
        python=platform.python_version(),
        recipe_sha256=base["sha256"],
        repeats=repeats,
        rows=rows,
        summary=summary,
        worker_implementation_sha256=next(iter(implementations)),
        unchanged_candidate=first_candidate,
        graphics_array_sha256=graphics_hashes,
        capture_sha256={
            p.name: digest(p)
            for p in (
                capture_path,
                capture_path.parent / "source.png",
                capture_path.parent / "depth.png",
            )
        },
        scope=(
            "Fresh worker processes, alternating prepared/unprepared arms. Same sealed settings, "
            "one real calibrated capture, two inspection calls per session. Preparation consumes "
            "no image and runs no network forward. First-image times include packet IO, inference, "
            "ICP/fit, requested graphics and cleanup; exclude acquisition/file decoding, GUI, "
            "robot and hardware acceptance. Preparation shifts startup; it does not remove it."
        ),
        attribution="T-LESS, Hodan et al., WACV 2017, CC BY 4.0",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
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
