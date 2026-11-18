"""Interrupt a busy real worker, recover and compare every frozen result/graphic.

Uses the existing five-proposal T-LESS recipe, fixed gates and unchanged network.
No ground-truth poses enter inference. These checks do not qualify pick accuracy.
"""

import argparse
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
from worker_metrics import cpu_delta, snapshot

from opencv_simple_gdrnpp.bridge import WorkerClient


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def stable(report):
    return dict(
        candidates=report["candidates"],
        best_pose=report["best_pose"],
        accepted_count=report["accepted_candidates"],
        overlap_selection=report.get("overlap_selection"),
    )


def run(recipe, models, output):
    output.mkdir(parents=True, exist_ok=False)
    frozen = json.loads(
        (recipe.parent.parent / "candidate_execution/worker_comparison.json").read_text()
    )
    data = dict(
        base=dict(
            kind="recipe",
            path=str(recipe.resolve()),
            sha256=hashlib.sha256(recipe.read_bytes()).hexdigest(),
        ),
        models_dir=str(models.resolve()),
    )
    client = WorkerClient(python=sys.executable)
    rows, completed = [], []

    def preview(label):
        metadata, arrays = client.call("preview", dict(data, graphics=True))
        assert stable(metadata["report"]) == frozen["stable_results"]["five_proposals"]
        assert set(arrays) == {"overlay", "points", "colors"}
        hashes = {
            key: hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
            for key, array in arrays.items()
        }
        assert all(frozen["graphics_sha256"]["five_proposals/" + k] == v for k, v in hashes.items())
        assert metadata["report"]["candidate_execution"]["effective_workers"] == 4
        assert list(client.root.iterdir()) == []
        write(output / (label + ".json"), metadata)
        completed.append(
            dict(
                label=label,
                worker=metadata["report"]["worker"],
                graphics_sha256=hashes,
                all_candidates_exact=True,
            )
        )
        return metadata

    try:
        prepared, empty = client.call("prepare", data)
        assert empty == {} and not prepared["prepared"]["inspection_performed"]
        assert prepared["prepared"]["worker"]["inference_request_count"] == 0
        first = preview("baseline")
        previous = first
        for action in ("cancel", "timeout", "close"):
            before = snapshot(client.process.pid)
            assert before["complete"]
            processes = [psutil.Process(row["pid"]) for row in before["processes"]]
            root, readers = client.root, list(client._readers)
            cancel, outcome = threading.Event(), {}
            client.timeout = 0.75 if action == "timeout" else 180

            def request():
                try:
                    outcome["result"] = client.call(
                        "preview", dict(data, graphics=False), cancelled=cancel.is_set
                    )
                except Exception as error:
                    outcome["error"] = error

            thread = threading.Thread(target=request)
            thread.start()
            started = time.perf_counter()
            observed_cpu, requested_stop = 0.0, None
            while thread.is_alive():
                after = snapshot(before["processes"][0]["pid"])
                used = cpu_delta(before, after)
                if used is not None:
                    observed_cpu = max(used, observed_cpu)
                if action != "timeout" and requested_stop is None and observed_cpu >= 0.25:
                    requested_stop = time.perf_counter()
                    if action == "close":
                        client.close()
                    else:
                        cancel.set()
                if time.perf_counter() - started > 15:
                    cancel.set()
                    raise TimeoutError("Lifecycle replay exceeded its watchdog")
                thread.join(0.01)
            finished = time.perf_counter()
            assert "result" not in outcome and observed_cpu >= 0.25
            assert isinstance(
                outcome["error"], TimeoutError if action == "timeout" else RuntimeError
            )
            gone, alive = psutil.wait_procs(processes, timeout=5)
            assert not alive and len(gone) == len(processes)
            assert client.process is None and not root.exists()
            assert all(not reader.is_alive() for reader in readers)
            rows.append(
                dict(
                    action=action,
                    owned_processes_before=before,
                    observed_owned_cpu_seconds=observed_cpu,
                    request_elapsed_seconds=finished - started,
                    stop_to_return_seconds=None
                    if requested_stop is None
                    else finished - requested_stop,
                    error_type=type(outcome["error"]).__name__,
                    no_result_returned=True,
                    all_observed_processes_retired=True,
                    pipe_readers_joined=True,
                    private_storage_removed=True,
                )
            )
            if action == "close":
                client = WorkerClient(python=sys.executable)
            client.timeout = 180
            recovered = preview("after_" + action)
            assert (
                recovered["report"]["worker"]["session"] != previous["report"]["worker"]["session"]
            )
            assert recovered["report"]["worker"]["model_load_count"] == 1
            assert recovered["report"]["worker"]["inference_request_count"] == 1
            previous = recovered
            print(
                json.dumps(dict(action=action, retired=True, recovered_frozen_results=True)),
                flush=True,
            )
        try:
            client.call("render", dict(token=first["token"], index=0))
        except ValueError as error:
            assert "expired" in str(error)
        else:
            raise AssertionError("A token from a retired worker was accepted")
        assert client.process is None and not client.root.exists()
        receipt = dict(
            scope="Windows CPU real model; busy request interruption and fresh-session recovery",
            recipe_sha256=hashlib.sha256(recipe.read_bytes()).hexdigest(),
            prepared_without_inspection=True,
            interrupted=rows,
            completed=completed,
            completed_previews=4,
            stale_token_rejected=True,
            gates_or_recipe_changed=False,
            numerical_graphics_or_accuracy_change=False,
            attribution="T. Hodan et al., T-LESS, WACV 2017, CC BY 4.0; unchanged retained recipe",
            source="https://cmp.felk.cvut.cz/t-less/",
        )
        write(output / "receipt.json", receipt)
        return receipt
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("models", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.recipe, args.models, args.output)
