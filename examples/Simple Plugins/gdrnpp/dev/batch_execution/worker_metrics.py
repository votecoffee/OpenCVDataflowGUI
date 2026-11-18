"""Read-only owned-process-tree sampling, including Windows venv launcher children."""

import math

import psutil


def snapshot(root_pid):
    rows, problems = [], []
    try:
        root = psutil.Process(root_pid)
        processes = [root, *root.children(recursive=True)]
    except (psutil.Error, OSError) as error:
        return dict(
            complete=False, processes=[], problems=[type(error).__name__], tree_rss_bytes=None
        )
    for process in processes:
        try:
            cpu, memory = process.cpu_times(), process.memory_info()
            seconds = cpu.user + cpu.system
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError("Invalid CPU counter")
            rows.append(
                dict(
                    pid=process.pid,
                    created=process.create_time(),
                    name=process.name(),
                    cpu_seconds=seconds,
                    rss_bytes=memory.rss,
                    peak_working_set_bytes=getattr(memory, "peak_wset", None),
                )
            )
        except (psutil.Error, OSError, ValueError) as error:
            problems.append(type(error).__name__)
    return dict(
        complete=not problems,
        processes=rows,
        problems=problems,
        tree_rss_bytes=sum(row["rss_bytes"] for row in rows) if not problems else None,
    )


def cpu_delta(before, after):
    if not before["complete"] or not after["complete"]:
        return None
    first = {(p["pid"], p["created"]): p["cpu_seconds"] for p in before["processes"]}
    second = {(p["pid"], p["created"]): p["cpu_seconds"] for p in after["processes"]}
    if not first or first.keys() != second.keys():
        return None  # Churn/reused PID: do not invent zero CPU or a comparable interval.
    deltas = [second[key] - first[key] for key in first]
    return sum(deltas) if all(math.isfinite(v) and v >= 0 for v in deltas) else None
