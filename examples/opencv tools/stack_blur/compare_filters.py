"""Original matched-variance smoothing experiment and isolated native timings (MIT)."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np


def fixture(width=1024, height=768):
    clean = np.broadcast_to(
        np.where(np.arange(width) < width // 2, 60, 180), (height, width)
    ).copy()
    noisy = np.clip(np.rint(clean + np.random.default_rng(159).normal(0, 18, clean.shape)), 0, 255)
    return clean.astype(np.uint8), noisy.astype(np.uint8)


def matched_sigma(kernel):
    """Match the variance of a finite Gaussian to a separable triangular stack kernel."""
    radius = kernel // 2
    if radius == 0:
        return 0.0
    target = radius * (radius + 2) / 6
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    low, high = 0.01, float(kernel)
    for _ in range(70):
        sigma = (low + high) / 2
        weights = np.exp(-x * x / (2 * sigma * sigma))
        variance = float(np.sum(weights * x * x) / weights.sum())
        if variance < target:
            low = sigma
        else:
            high = sigma
    return (low + high) / 2


def metrics(image, clean, kernel):
    values = np.asarray(image, np.float64) * 255
    height, width = clean.shape
    edge = width / 2 - 0.5
    margin = kernel // 2 + 8
    columns = np.arange(width)
    selected = (columns > margin) & (columns < width - margin) & (np.abs(columns - edge) > margin)
    rows = slice(height // 4, 3 * height // 4)
    if not selected.any():
        raise ValueError("Use a wider fixture for this kernel")
    residual = (values - clean)[rows][:, selected]
    profile = values[rows].mean(axis=0)
    normalized = (profile - 60) / 120

    def crossing(level):
        candidates = np.flatnonzero(
            (normalized[:-1] <= level) & (normalized[1:] >= level) & (np.diff(normalized) > 0)
        )
        if not len(candidates):
            raise ValueError("No measured edge crossing")
        index = min(candidates, key=lambda i: abs(i - edge))
        return float(
            index + (level - normalized[index]) / (normalized[index + 1] - normalized[index])
        )

    return {
        "flat_rms_dn": float(np.sqrt(np.mean(residual**2))),
        "edge_center_px": crossing(0.5),
        "center_error_px": abs(crossing(0.5) - edge),
        "edge_10_90_width_px": crossing(0.9) - crossing(0.1),
        "peak_gradient_dn_per_px": float(np.max(np.diff(profile))),
        "flat_sample_count": int(residual.size),
    }


def compare(kernel, clean, noisy):
    source = noisy.astype(np.float32) / 255
    sigma = matched_sigma(kernel)
    stack = cv2.stackBlur(source, (kernel, kernel))
    gaussian = cv2.GaussianBlur(
        source,
        (kernel, kernel),
        sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REPLICATE,
        hint=cv2.ALGO_HINT_ACCURATE,
    )
    return {
        "kernel": kernel,
        "matched_gaussian_sigma": sigma,
        "triangular_variance": (kernel // 2) * (kernel // 2 + 2) / 6,
        "input": metrics(source, clean, kernel),
        "stack": metrics(stack, clean, kernel),
        "gaussian": metrics(gaussian, clean, kernel),
    }


def benchmark(width=1920, height=1080, repeats=7):
    clean, noisy = fixture(width, height)
    source = noisy.astype(np.float32) / 255
    previous = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        results = []
        for kernel in (31, 127, 255):
            row = compare(kernel, clean, noisy)
            sigma = row["matched_gaussian_sigma"]
            operations = {
                "stack": lambda: cv2.stackBlur(source, (kernel, kernel)),
                "gaussian": lambda: cv2.GaussianBlur(
                    source,
                    (kernel, kernel),
                    sigma,
                    sigmaY=sigma,
                    borderType=cv2.BORDER_REPLICATE,
                    hint=cv2.ALGO_HINT_ACCURATE,
                ),
            }
            for operation in operations.values():
                for _ in range(2):
                    operation()
            samples = {name: [] for name in operations}
            for iteration in range(repeats):
                # Alternate ordering so one method does not always follow the other.
                order = list(operations) if iteration % 2 else list(reversed(operations))
                for name in order:
                    start = time.perf_counter_ns()
                    operations[name]()
                    samples[name].append((time.perf_counter_ns() - start) / 1_000_000)
            row["timings_ms"] = {
                name: {
                    "median": float(np.median(times)),
                    "p25": float(np.percentile(times, 25)),
                    "p75": float(np.percentile(times, 75)),
                    "samples": times,
                }
                for name, times in samples.items()
            }
            results.append(row)
        return {
            "model": "stack_gaussian_benchmark_v1",
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "platform": platform.system(),
            "width": width,
            "height": height,
            "channels": 1,
            "working_depth": "float32",
            "opencv_threads": 1,
            "repeats": repeats,
            "warmup_runs_per_method": 2,
            "source_sha256": hashlib.sha256(noisy.tobytes()).hexdigest(),
            "scope": "Isolated native calls with allocation; excludes graph copies, validation and "
            "graphics. Local observations, not performance guarantees.",
            "results": results,
        }
    finally:
        cv2.setNumThreads(previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 3 or args.report.exists():
        parser.error("Use at least three repeats and a fresh report path")
    result = benchmark(repeats=args.repeats)
    with args.report.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for row in result["results"]:
        print(
            row["kernel"],
            {name: values["median"] for name, values in row["timings_ms"].items()},
            flush=True,
        )


if __name__ == "__main__":
    main()
