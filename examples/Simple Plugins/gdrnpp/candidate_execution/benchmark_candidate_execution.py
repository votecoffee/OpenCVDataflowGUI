"""Real complete-worker comparison of serial, two-task and automatic depth stages.

The trusted benchmark selects a fixed Python API override, never recipe code.
Neural inference, thread/batch policy, fit gates and graphics stay unchanged.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
from worker_metrics import cpu_delta, snapshot

from opencv_simple_gdrnpp import bridge
from opencv_simple_gdrnpp.bridge import WorkerClient


class DepthClient(WorkerClient):
    def __init__(self, workers, **kwargs):
        super().__init__(**kwargs)
        self.workers = workers

    def _start(self):
        original_popen = subprocess.Popen
        code = f"""
from opencv_simple_gdrnpp import worker
original = worker.run_recipe
def run(*args, **kwargs):
    return original(*args, **kwargs, geometry_workers={self.workers!r})
worker.run_recipe = run
worker.main()
"""

        def start(args, **kwargs):
            if args[1:4] != ["-u", "-m", "opencv_simple_gdrnpp.worker"]:
                raise ValueError("Unexpected entry point in explicit depth-stage benchmark")
            return original_popen([args[0], "-u", "-c", code, *args[4:]], **kwargs)

        with patch.object(bridge.subprocess, "Popen", start):
            super()._start()


def run(examples, models, output):
    if output.exists():
        raise FileExistsError("Choose a fresh candidate-stage timing receipt")
    paths = dict(
        five_proposals=examples / "batch_execution/batch1.gdrnpp",
        repeated_target=examples / "overlap_review/enabled.gdrnpp",
    )
    modes = dict(serial=1, two=2, automatic=None)
    resources = {
        name: dict(
            base=dict(
                kind="recipe",
                path=str(path.resolve()),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            ),
            models_dir=str(models.resolve()),
        )
        for name, path in paths.items()
    }
    rows, preparations, numerical, hashes = [], [], {}, {}
    with ExitStack() as stack:
        clients = {
            mode: stack.enter_context(DepthClient(workers, python=sys.executable))
            for mode, workers in modes.items()
        }
        for name, data in resources.items():
            for mode, client in clients.items():
                ready, empty = client.call("prepare", data)
                assert empty == {} and not ready["prepared"]["inspection_performed"]
                assert ready["prepared"]["worker"]["model_load_count"] == 1
                preparations.append(dict(case=name, mode=mode, readiness=ready))
            for graphics in (False, True) if name == "five_proposals" else (False,):
                for round_number in range(4):
                    order = list(modes) if round_number % 2 == 0 else list(reversed(modes))
                    for mode in order:
                        client = clients[mode]
                        before = snapshot(client.process.pid)
                        started = time.perf_counter()
                        metadata, returned = client.call("preview", dict(data, graphics=graphics))
                        elapsed = time.perf_counter() - started
                        after = snapshot(client.process.pid)
                        report = metadata["report"]
                        stable = dict(
                            candidates=report["candidates"],
                            best_pose=report["best_pose"],
                            accepted_count=report["accepted_candidates"],
                            overlap_selection=report.get("overlap_selection"),
                        )
                        if name not in numerical:
                            numerical[name] = stable
                        assert stable == numerical[name]
                        assert report["accepted_candidates"] == 1
                        assert report["worker"]["model_load_count"] == 1
                        assert report["worker"]["model_cache_hit"]
                        expected_workers = 4 if mode == "automatic" else modes[mode]
                        plan = report["candidate_execution"]
                        assert plan["effective_workers"] == expected_workers
                        assert all(s["peak_in_flight"] <= expected_workers for s in plan["stages"])
                        assert set(returned) == (
                            {"overlay", "points", "colors"} if graphics else set()
                        )
                        for key, array in returned.items():
                            digest = hashlib.sha256(
                                np.ascontiguousarray(array).tobytes()
                            ).hexdigest()
                            identity = name + "/" + key
                            assert identity not in hashes or hashes[identity] == digest
                            hashes[identity] = digest
                        rows.append(
                            dict(
                                case=name,
                                mode=mode,
                                graphics=graphics,
                                round=round_number,
                                warmup=round_number == 0,
                                client_round_trip_seconds=elapsed,
                                candidate_execution=plan,
                                seconds=report["seconds"],
                                worker_profile=report["execution_profile"],
                                request_timing=metadata["request_timing"],
                                worker=report["worker"],
                                owned_processes=after,
                                worker_cpu_seconds=cpu_delta(before, after),
                                accepted_count=1,
                                all_candidates_pose_fit_overlap_exact=True,
                                returned_arrays_bit_identical=True,
                            )
                        )
                        print(
                            json.dumps(
                                dict(
                                    case=name,
                                    mode=mode,
                                    graphics=graphics,
                                    round=round_number,
                                    seconds=elapsed,
                                )
                            ),
                            flush=True,
                        )
    summaries = []
    for name in paths:
        for graphics in (False, True) if name == "five_proposals" else (False,):
            for mode in modes:
                group = [
                    r
                    for r in rows
                    if r["case"] == name
                    and r["graphics"] == graphics
                    and r["mode"] == mode
                    and not r["warmup"]
                ]
                values = [r["client_round_trip_seconds"] for r in group]
                summaries.append(
                    dict(
                        case=name,
                        mode=mode,
                        graphics=graphics,
                        measured_samples=len(group),
                        median_seconds=statistics.median(values),
                        minimum_seconds=min(values),
                        maximum_seconds=max(values),
                        depth_refinement_median_seconds=statistics.median(
                            r["seconds"]["depth_refinement"] for r in group
                        ),
                        depth_verification_median_seconds=statistics.median(
                            r["seconds"]["depth_verification"] for r in group
                        ),
                    )
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(
            dict(
                summary=summaries,
                samples=rows,
                preparations=preparations,
                stable_results=numerical,
                graphics_sha256=hashes,
                neural_threads=4,
                batch_size=1,
                scope=(
                    "Three persistent checkpoint workers, two related real-scene proposal sets. "
                    "Separate preparation, one warmup/three alternating measured rounds per mode. "
                    "Five boxes: graphics off/on; six boxes including a repeated target: off. "
                    "Same retained RGB-D/model/neural threads/batch/gates. "
                    "Only depth-task concurrency changes. Nested timings overlap. "
                    "Acquisition, caller decoding and GUI excluded; "
                    "one shared workstation, not capacity qualification."
                ),
                attribution="T-LESS, Hodan et al., WACV 2017, CC BY 4.0",
            ),
            stream,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")
    print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("examples", "models", "new_output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.examples, args.models, args.new_output)
