"""Isolated native contour timings/Windows memory peaks; excludes graph/snapshot/JSON cost."""

import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
import time

import cv2
import numpy as np

from opencv_dataflow.run_contours import count_runs


def memory():
    if os.name != "nt":
        return None

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("faults", ctypes.c_ulong),
            *[
                (name, ctypes.c_size_t)
                for name in (
                    "peak_working",
                    "working",
                    "peak_paged",
                    "paged",
                    "peak_nonpaged",
                    "nonpaged",
                    "commit",
                    "peak_commit",
                    "private",
                )
            ],
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(Counters),
        ctypes.c_ulong,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    result = Counters()
    result.cb = ctypes.sizeof(result)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(result), result.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {
        name: getattr(result, name)
        for name in ("peak_working", "working", "peak_commit", "private")
    }


def worker(method, size):
    cv2.setNumThreads(1)

    def extract(image):
        return (
            cv2.findContoursLinkRuns(image)
            if method == "runs"
            else cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        )

    extract(np.zeros((16, 16), np.uint8))
    image = np.empty((size, size), np.uint8)
    image.fill(0)
    side = size // 20
    for y in range(4):
        for x in range(4):
            image[
                (y + 1) * size // 6 : (y + 1) * size // 6 + side,
                (x + 1) * size // 6 : (x + 1) * size // 6 + side,
            ] = 255
    before = memory()
    start = time.perf_counter()
    runs = count_runs(image, 250000)
    preflight_ms = (time.perf_counter() - start) * 1000
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        contours, links = extract(image)
        timings.append((time.perf_counter() - start) * 1000)
    after = memory()
    assert len(contours) == 16
    return {
        "method": method,
        "size": [size, size],
        "input_bytes": image.nbytes,
        "runs": runs,
        "contours": len(contours),
        "points": sum(len(c) for c in contours),
        "polygon_area_sum": sum(cv2.contourArea(c) for c in contours),
        "native_median_ms": statistics.median(timings),
        "preflight_ms": preflight_ms,
        "memory_before": before,
        "memory_after": after,
        "peak_commit_increase_bytes": after["peak_commit"] - before["peak_commit"]
        if after
        else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("dense", "runs"))
    parser.add_argument("--size", type=int, default=6000)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker, args.size)))
        return
    records = []
    for size in (2048, 6000, 8000):
        for method in ("dense", "runs"):
            completed = subprocess.run(
                [sys.executable, __file__, "--worker", method, "--size", str(size)],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            records.append(json.loads(completed.stdout))
    print(
        json.dumps(
            {
                "opencv": cv2.__version__,
                "numpy": np.__version__,
                "records": records,
                "scope": (
                    "Native API only, 3 calls, one OpenCV thread, fresh process per API/size. "
                    "Both workers count runs separately. No graph, input snapshot, measurements, "
                    "JSON or raster-output timing. Memory counters available on Windows only; "
                    "fixture-dependent, not a speed guarantee."
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
