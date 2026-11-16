"""Alternating batch modes on one real multi-proposal capture, using the actual worker.

Resource preparation and warmups are separate. All measured samples retain
full request/fit outcomes; no application-specific gate or thread tuning.
"""

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import psutil
from worker_metrics import cpu_delta, snapshot

from opencv_simple_gdrnpp.bridge import WorkerClient
from opencv_simple_gdrnpp.workflow import load_capture


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(recipe, models, capture_path, output, *, worker_python=sys.executable):
    if output.exists():
        raise FileExistsError("Choose a fresh complete-batch timing receipt")
    capture = load_capture(capture_path)
    count = len(capture.detections)
    if count < 2:
        raise ValueError("Use a real capture with multiple supplied instance rectangles")
    arrays = dict(
        rgb=capture.frame.rgb,
        depth=capture.frame.depth,
        camera=capture.frame.camera,
        mask=capture.frame.mask,
    )
    data = dict(
        base=dict(
            kind="recipe",
            path=str(recipe.resolve()),
            sha256=hashlib.sha256(recipe.read_bytes()).hexdigest(),
        ),
        models_dir=str(models.resolve()),
        current=dict(
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
        ),
    )
    rows, preparations = [], []
    for graphics in (False, True):
        with WorkerClient(python=worker_python) as client:
            prepared, none = client.call("prepare", {k: data[k] for k in ("base", "models_dir")})
            assert none == {} and not prepared["prepared"]["inspection_performed"]
            preparations.append(
                dict(
                    graphics=graphics,
                    readiness=prepared["prepared"],
                    request_timing=prepared["request_timing"],
                )
            )
            reference = None
            for round_number in range(4):
                order = [1, 2, 4, 8] if round_number % 2 == 0 else [8, 4, 2, 1]
                for size in order:
                    before = snapshot(client.process.pid)
                    started = time.perf_counter()
                    metadata, result = client.call(
                        "preview",
                        dict(data, graphics=graphics, draft=dict(settings=dict(batch_size=size))),
                        arrays,
                    )
                    elapsed = time.perf_counter() - started
                    after = snapshot(client.process.pid)
                    cpu_seconds = cpu_delta(before, after)
                    report = metadata["report"]
                    assert report["capture_id"] == capture.capture_id
                    expected = [min(size, count - i) for i in range(0, count, size)]
                    assert report["batch_execution"]["group_sizes"] == expected
                    assert report["worker"]["model_load_count"] == 1
                    assert report["worker"]["model_cache_hit"]
                    assert set(result) == ({"overlay", "points", "colors"} if graphics else set())
                    candidates = [
                        dict(id=c["id"], pose=c["pose"], fit=c["fit"]) for c in report["candidates"]
                    ]
                    if reference is None:
                        assert size == 1
                        reference = candidates
                    row = dict(
                        graphics=graphics,
                        round=round_number,
                        warmup=round_number == 0,
                        batch_size=size,
                        group_sizes=expected,
                        client_round_trip_seconds=elapsed,
                        worker_cpu_seconds=cpu_seconds,
                        worker_mean_cores_busy=cpu_seconds / elapsed
                        if cpu_seconds is not None
                        else None,
                        owned_processes_before=before,
                        owned_processes_after=after,
                        request_timing=metadata["request_timing"],
                        worker_profile=report["execution_profile"],
                        seconds=report["seconds"],
                        worker=report["worker"],
                        tree_rss_after_request_bytes=after["tree_rss_bytes"],
                        returned_array_bytes=sum(value.nbytes for value in result.values()),
                        accepted_count=report["accepted_candidates"],
                        candidates=candidates,
                        decisions_match_batch1=[c["fit"]["status"] for c in candidates]
                        == [c["fit"]["status"] for c in reference],
                    )
                    rows.append(row)
                    print(
                        json.dumps(
                            dict(
                                graphics=graphics,
                                round=round_number,
                                batch=size,
                                seconds=elapsed,
                                accepted=row["accepted_count"],
                            )
                        ),
                        flush=True,
                    )
    summaries = []
    for graphics in (False, True):
        baseline = [
            r["client_round_trip_seconds"]
            for r in rows
            if r["graphics"] == graphics and r["batch_size"] == 1 and not r["warmup"]
        ]
        for size in (1, 2, 4, 8):
            group = [
                r
                for r in rows
                if r["graphics"] == graphics and r["batch_size"] == size and not r["warmup"]
            ]
            times = [r["client_round_trip_seconds"] for r in group]
            summaries.append(
                dict(
                    graphics=graphics,
                    batch_size=size,
                    measured_samples=len(times),
                    median_seconds=statistics.median(times),
                    minimum_seconds=min(times),
                    maximum_seconds=max(times),
                    network_median_seconds=statistics.median(
                        r["seconds"]["network"] for r in group
                    ),
                    mean_cores_busy_median=(
                        statistics.median(r["worker_mean_cores_busy"] for r in group)
                        if all(r["worker_mean_cores_busy"] is not None for r in group)
                        else None
                    ),
                    median_speed_ratio_to_batch1=statistics.median(baseline)
                    / statistics.median(times),
                    all_decisions_match_batch1=all(r["decisions_match_batch1"] for r in group),
                )
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        output,
        dict(
            summary=summaries,
            samples=rows,
            preparation=preparations,
            python=platform.python_version(),
            logical_cpu_count=psutil.cpu_count(),
            capture_id=capture.capture_id,
            instances=count,
            checkpoint_sha256=prepared["prepared"]["binding"]["checkpoint_sha256"],
            scope=(
                "One persistent four-thread CPU worker per graphics mode; explicit preparation, "
                "one warmup and three measured calls per batch size, alternating order. "
                "In-memory capture before timing; acquisition, caller decoding and GUI excluded. "
                "CPU/RSS cover the launcher and owned descendants, not launcher alone. "
                "Core occupancy is summed process CPU / client round-trip wall time. "
                "RSS sums include shared pages; per-process peak working sets are cumulative "
                "and do not isolate individual batch sizes. Nested timings overlap. "
                "One shared workstation/scene, not general capacity or GPU qualification."
            ),
            attribution="T-LESS, Hodan et al., WACV 2017, CC BY 4.0",
        ),
    )
    print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("recipe", "models", "capture", "new_output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.recipe, args.models, args.capture, args.new_output)
