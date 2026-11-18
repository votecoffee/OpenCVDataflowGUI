"""Actual paired worker calls, swapping only an explicitly pinned transport implementation.

Benchmark-only child entry points are selected here by trusted developer code,
never by a saved project or recipe. Both modes retain the current numerical code.
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
from benchmark_packets import REFERENCE_SHA256, reference_module
from worker_metrics import cpu_delta, snapshot

from opencv_simple_gdrnpp import bridge, wire
from opencv_simple_gdrnpp.bridge import WorkerClient
from opencv_simple_gdrnpp.workflow import load_capture


class MethodClient(WorkerClient):
    def __init__(self, method, reference, **kwargs):
        super().__init__(**kwargs)
        self.method, self.reference = method, reference.resolve()
        self.original = reference_module(reference)

    def _start(self):
        original_popen = subprocess.Popen
        code = f"""
import hashlib, importlib.util
from pathlib import Path
from opencv_simple_gdrnpp import worker, wire
path = Path({str(self.reference)!r})
assert hashlib.sha256(path.read_bytes()).hexdigest() == {REFERENCE_SHA256!r}
method = {self.method!r}
if method == 'legacy':
    spec = importlib.util.spec_from_file_location('transport_reference', path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    worker.read_packet, writer = reference.read_packet, reference.write_packet
else:
    writer = wire.write_packet
def send(directory, metadata, arrays):
    metadata = dict(metadata, benchmark_transport=dict(method=method, benchmark_only=True))
    writer(directory, metadata, arrays)
worker.write_packet = send
worker.main()
"""

        def start(args, **kwargs):
            if args[1:4] != ["-u", "-m", "opencv_simple_gdrnpp.worker"]:
                raise ValueError("Unexpected worker entry point in explicit transport benchmark")
            return original_popen([args[0], "-u", "-c", code, *args[4:]], **kwargs)

        with patch.object(bridge.subprocess, "Popen", start):
            super()._start()

    def call(self, *args, **kwargs):
        implementation = self.original if self.method == "legacy" else wire
        with (
            patch.object(bridge, "write_packet", implementation.write_packet),
            patch.object(bridge, "read_packet", implementation.read_packet),
        ):
            result = super().call(*args, **kwargs)
        assert result[0]["benchmark_transport"] == dict(method=self.method, benchmark_only=True)
        return result


def run(recipe, capture_path, models, reference, output):
    if output.exists():
        raise FileExistsError("Choose a fresh complete-transport receipt")
    capture = load_capture(capture_path)
    arrays = dict(
        rgb=capture.frame.rgb,
        depth=capture.frame.depth,
        camera=capture.frame.camera,
        mask=capture.frame.mask,
    )
    resources = dict(
        base=dict(
            kind="recipe",
            path=str(recipe.resolve()),
            sha256=hashlib.sha256(recipe.read_bytes()).hexdigest(),
        ),
        models_dir=str(models.resolve()),
    )
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
    rows, preparations, frozen, graphics_hashes = [], [], None, {}
    for graphics in (False, True):
        with ExitStack() as stack:
            clients = {
                method: stack.enter_context(MethodClient(method, reference, python=sys.executable))
                for method in ("legacy", "stored")
            }
            for method, client in clients.items():
                ready, empty = client.call("prepare", resources)
                assert empty == {} and not ready["prepared"]["inspection_performed"]
                preparations.append(dict(method=method, graphics=graphics, readiness=ready))
            for round_number in range(4):
                order = ["legacy", "stored"] if round_number % 2 == 0 else ["stored", "legacy"]
                for method in order:
                    client = clients[method]
                    before = snapshot(client.process.pid)
                    started = time.perf_counter()
                    metadata, returned = client.call(
                        "preview", dict(resources, current=current, graphics=graphics), arrays
                    )
                    elapsed = time.perf_counter() - started
                    after = snapshot(client.process.pid)
                    report = metadata["report"]
                    numerical = dict(
                        candidates=report["candidates"],
                        best_pose=report["best_pose"],
                        accepted_count=report["accepted_candidates"],
                    )
                    if frozen is None:
                        frozen = numerical
                    assert numerical == frozen
                    assert report["accepted_candidates"] == 1
                    assert report["worker"]["model_load_count"] == 1
                    assert report["worker"]["model_cache_hit"]
                    assert set(returned) == ({"overlay", "points", "colors"} if graphics else set())
                    for key, array in returned.items():
                        digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
                        assert key not in graphics_hashes or graphics_hashes[key] == digest
                        graphics_hashes[key] = digest
                    rows.append(
                        dict(
                            method=method,
                            graphics=graphics,
                            round=round_number,
                            warmup=round_number == 0,
                            client_round_trip_seconds=elapsed,
                            request_timing=metadata["request_timing"],
                            execution_profile=report["execution_profile"],
                            seconds=report["seconds"],
                            worker=report["worker"],
                            owned_processes=after,
                            worker_cpu_seconds=cpu_delta(before, after),
                            returned_array_bytes=sum(a.nbytes for a in returned.values()),
                            candidates_pose_fit_exact=True,
                            result_arrays_bit_identical=True,
                            accepted_count=report["accepted_candidates"],
                        )
                    )
                    print(
                        json.dumps(
                            dict(
                                method=method,
                                graphics=graphics,
                                round=round_number,
                                seconds=elapsed,
                            )
                        ),
                        flush=True,
                    )
    summaries = []
    for graphics in (False, True):
        for method in ("legacy", "stored"):
            group = [
                r
                for r in rows
                if r["graphics"] == graphics and r["method"] == method and not r["warmup"]
            ]
            times = [r["client_round_trip_seconds"] for r in group]
            summaries.append(
                dict(
                    method=method,
                    graphics=graphics,
                    measured_samples=len(group),
                    median_seconds=statistics.median(times),
                    minimum_seconds=min(times),
                    maximum_seconds=max(times),
                    client_packet_write_median_ms=statistics.median(
                        r["request_timing"]["client_packet_write_seconds"] for r in group
                    )
                    * 1000,
                    client_packet_read_median_ms=statistics.median(
                        r["request_timing"]["client_packet_read_seconds"] for r in group
                    )
                    * 1000,
                    worker_packet_write_median_ms=statistics.median(
                        r["request_timing"]["worker"]["packet_write_seconds"] for r in group
                    )
                    * 1000,
                )
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(
            dict(
                summary=summaries,
                samples=rows,
                preparations=preparations,
                graphics_sha256=graphics_hashes,
                stable_result=frozen,
                reference_sha256=REFERENCE_SHA256,
                candidate_wire_sha256=hashlib.sha256(Path(wire.__file__).read_bytes()).hexdigest(),
                scope=(
                    "Explicit benchmark swaps both transport endpoints against pinned reference. "
                    "Same numerical code/model/recipe/four CPU threads. Two prepared workers "
                    "per graphics mode; one warmup and three alternating measured rounds. "
                    "Capture arrays in memory; acquisition, caller decoding and GUI excluded. "
                    "Nested times overlap; shared workstation, "
                    "not capacity or stall-elimination proof."
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
    for name in ("recipe", "capture", "models", "reference", "new_output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.recipe, args.capture, args.models, args.reference, args.new_output)
