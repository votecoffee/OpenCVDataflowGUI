"""Bounded fit-kernel repeat with frozen real evidence; no inference or rendering."""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
from opencv_simple_foundation_pose.render import fit_evidence
from opencv_simple_foundation_pose.specs import Frame, Settings


def benchmark(root, repeats=100):
    rows = []
    for folder in sorted(Path(root).glob("case_*/frozen")):
        meta = json.loads((folder / "review.json").read_bytes())
        report = meta["report"]
        with np.load(folder / "sources.npz", allow_pickle=False) as arrays:
            source = {k: arrays[k] for k in ("rgb", "depth", "camera", "mask")}
        fit = report["fit_settings"]
        settings = Settings(
            distance_tolerance=fit["distance_tolerance_m"],
            minimum_coverage=fit["minimum_coverage"],
            maximum_contradiction=fit["maximum_contradiction"],
        )
        for index, candidate in enumerate(report["candidates"]):
            mask = source["mask"].copy()
            if candidate.get("crop", {}).get("source_xyxy"):
                x0, y0, x1, y1 = candidate["crop"]["source_xyxy"]
                h, w = mask.shape
                mask &= (np.arange(w)[None, :] >= x0) & (np.arange(w)[None, :] < x1)
                mask &= (np.arange(h)[:, None] >= y0) & (np.arange(h)[:, None] < y1)
            frame = Frame(
                source["rgb"], source["depth"], source["camera"], mask, report["target_frame"]
            )
            with np.load(folder / f"candidate_{index + 1:02d}.npz", allow_pickle=False) as evidence:
                rendered = evidence["points"].reshape(-1, 1, 3)
            samples, retained_bytes = {False: [], True: []}, 0
            for iteration in range(repeats + 5):
                for retain in (False, True) if iteration % 2 else (True, False):
                    sink = {} if retain else None
                    start = time.perf_counter()
                    result = fit_evidence(None, frame, None, settings, rendered, retained=sink)
                    duration = time.perf_counter() - start
                    assert result == (candidate.get("depth_fit") or candidate["fit"])
                    if sink:
                        retained_bytes = sum(v.nbytes for v in sink.values())
                    if iteration >= 5:
                        samples[retain].append(duration * 1000)
            rows.append(
                dict(
                    case=folder.parent.name,
                    candidate=index + 1,
                    samples=len(rendered),
                    retained_array_bytes=retained_bytes,
                    baseline_median_ms=statistics.median(samples[False]),
                    retained_median_ms=statistics.median(samples[True]),
                    fit_identical=True,
                )
            )
    return dict(
        repeats=repeats,
        warmup=5,
        alternating_order=True,
        cases=rows,
        scope="Fit kernel only on original predicted samples; no inference/render/IPC/GUI. "
        "Exploratory timings, not end-to-end latency or accuracy qualification.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("examples", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_bytes((json.dumps(benchmark(args.examples), indent=2) + "\n").encode())
