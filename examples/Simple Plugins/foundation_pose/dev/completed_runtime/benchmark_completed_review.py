"""Exploratory fit-retention and receipt bookkeeping costs on frozen real evidence."""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from opencv_simple_foundation_pose.completed import RetainedExecution
from opencv_simple_foundation_pose.persistence import load_recipe
from opencv_simple_foundation_pose.render import fit_evidence
from opencv_simple_foundation_pose.specs import Frame, Settings


def benchmark(root, repeats=100):
    rows, receipts = [], []
    for directory in sorted(Path(root).glob("*/frozen")):
        meta = json.loads((directory / "review.json").read_bytes())
        report = meta["report"]
        with np.load(directory / "sources.npz", allow_pickle=False) as source:
            frame = Frame(
                *(source[k] for k in ("rgb", "depth", "camera", "mask")), report["target_frame"]
            )
        settings = Settings(**report["execution"]["settings"])
        for index, candidate in enumerate(report["candidates"]):
            with np.load(directory / f"candidate_{index + 1:02d}.npz", allow_pickle=False) as e:
                rendered = e["points"].reshape(-1, 1, 3)
            values, retained_bytes = {False: [], True: []}, 0
            for iteration in range(repeats + 5):
                for retain in (False, True) if iteration % 2 else (True, False):
                    sink = {} if retain else None
                    start = time.perf_counter()
                    result = fit_evidence(None, frame, None, settings, rendered, retained=sink)
                    elapsed = time.perf_counter() - start
                    assert result == candidate["fit"]
                    if sink:
                        retained_bytes = sum(v.nbytes for v in sink.values())
                    if iteration >= 5:
                        values[retain].append(elapsed * 1000)
            rows.append(
                dict(
                    case=directory.parent.name,
                    rank=index + 1,
                    original_fit_identical=True,
                    retained_bytes=retained_bytes,
                    baseline_median_ms=statistics.median(values[False]),
                    retained_median_ms=statistics.median(values[True]),
                )
            )
    for directory in sorted(Path(root).iterdir()):
        path = directory / "frozen/review.json"
        if path.is_file():
            report = json.loads(path.read_bytes())["report"]
            recipe = load_recipe(directory / "frozen/executed.sfp")
        elif (directory / "report.json").is_file():
            report = json.loads((directory / "report.json").read_bytes())
            recipe = load_recipe(directory / "trained.sfp")
        else:
            continue
        values = []
        for i in range(repeats + 5):
            start = time.perf_counter()
            RetainedExecution(
                "session", "token", recipe, None, report["capture_id"], None, report, {}, {}, {}
            )
            if i >= 5:
                values.append((time.perf_counter() - start) * 1000)
        receipts.append(
            dict(
                case=directory.name,
                candidates=len(report["candidates"]),
                median_ms=statistics.median(values),
            )
        )
    return dict(
        repeats=repeats,
        warmup=5,
        alternating_fit_order=True,
        fit=rows,
        receipt=receipts,
        scope="Fit kernel and receipt only; excludes neural/render/IPC/GUI/source capture. "
        "Exploratory warm timings, not end-to-end latency or accuracy qualification.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("examples", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_bytes((json.dumps(benchmark(args.examples), indent=2) + "\n").encode())
