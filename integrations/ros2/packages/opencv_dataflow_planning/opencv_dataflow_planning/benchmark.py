"""Separate planning latency from simulated pick outcomes; never fill unknowns."""

import argparse
import json
import math
from pathlib import Path

from .contracts import decode, identifier, number


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def summarize(rows):
    if not isinstance(rows, list) or not 1 <= len(rows) <= 100000:
        raise ValueError("Provide 1..100000 episode records")
    groups, identities = {}, set()
    for row in rows:
        expected = {
            "episode_id",
            "case_id",
            "seed",
            "backend",
            "warm",
            "planning_status",
            "planning_wall_s",
            "simulated_outcome",
            "simulated_cycle_s",
        }
        if not isinstance(row, dict) or set(row) != expected:
            raise ValueError("Episode fields must match the documented schema")
        for key in ("episode_id", "case_id", "backend"):
            identifier(row[key])
        if row["episode_id"] in identities:
            raise ValueError("Episode IDs must be unique")
        identities.add(row["episode_id"])
        if type(row["seed"]) is not int or not 0 <= row["seed"] <= 2**32 - 1:
            raise ValueError("Use an explicit uint32 scene seed")
        if type(row["warm"]) is not bool or row["planning_status"] not in (
            "preview",
            "no_plan",
            "timed_out",
            "invalid",
            "cancelled",
        ):
            raise ValueError("Invalid planning status or warm/cold state")
        if row["simulated_outcome"] not in ("success", "failed", "not_run", "unknown"):
            raise ValueError("A planning preview is not a successful pick")
        number(row["planning_wall_s"], 0, 86400)
        if row["simulated_cycle_s"] is not None:
            number(row["simulated_cycle_s"], 1e-9, 86400)
        if row["simulated_outcome"] == "success" and (
            row["planning_status"] != "preview" or row["simulated_cycle_s"] is None
        ):
            raise ValueError("Successful simulated picks require a preview and recorded cycle time")
        key = (row["case_id"], row["backend"], row["warm"])
        groups.setdefault(key, []).append(row)
    result = []
    for (case, backend, warm), episodes in sorted(groups.items()):
        latency = [row["planning_wall_s"] for row in episodes]
        completed = [row for row in episodes if row["simulated_outcome"] in ("success", "failed")]
        success = sum(row["simulated_outcome"] == "success" for row in completed)
        cycle = [
            row["simulated_cycle_s"] for row in completed if row["simulated_cycle_s"] is not None
        ]
        result.append(
            {
                "case_id": case,
                "backend": backend,
                "warm": warm,
                "episodes": len(episodes),
                "seeds": sorted({row["seed"] for row in episodes}),
                "preview_count": sum(row["planning_status"] == "preview" for row in episodes),
                "planning_latency_s": {
                    "p50": percentile(latency, 0.5),
                    "p95": percentile(latency, 0.95),
                },
                "simulated_attempts": len(completed),
                "simulated_successes": success,
                "simulated_success_fraction": success / len(completed) if completed else None,
                "not_run_or_unknown": len(episodes) - len(completed),
                "simulated_cycle_s": {"reported_count": len(cycle), "p50": percentile(cycle, 0.5)},
            }
        )
    return {
        "schema": "opencv_dataflow_planning_benchmark_v1",
        "groups": result,
        "qualification": "Reported simulation/planner evidence; not physical robot acceptance",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes_jsonl", type=Path)
    args = parser.parse_args()
    if args.episodes_jsonl.stat().st_size > 64 * 1024**2:
        parser.error("Episode log exceeds 64 MiB")
    rows = [decode(line) for line in args.episodes_jsonl.read_text(encoding="utf-8").splitlines()]
    print(json.dumps(summarize(rows), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
