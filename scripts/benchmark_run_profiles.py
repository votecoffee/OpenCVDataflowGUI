"""Compare observation profiles on the same native OpenCV preprocessing chain."""

import argparse
import gc
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Block, Graph, Port, Result
from opencv_dataflow.run_profiles import PROFILES


class Frame(Block):
    type_id = "input.benchmark_frame"
    outputs = (Port("image"),)

    def process(self, inputs):
        return Result({"image": self.image})


def build(image, name):
    graph, registry = Graph(), builtin_registry()
    source = graph.add(Frame())
    source.image = image
    previous = source
    for kind in ("cv.gray", "cv.gaussian_blur", "cv.canny"):
        block = graph.add(registry.create(kind))
        graph.connect(previous.id, "image", block.id, "image")
        previous = block
    graph.set_run_profile(name)
    return graph, previous


def benchmark(width=1280, height=720, iterations=20, rounds=3):
    image = np.random.default_rng(17).integers(0, 256, (height, width, 3), dtype=np.uint8)
    measurements = {name: [] for name in PROFILES}
    digests = set()
    threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        names = list(PROFILES)
        for turn in range(rounds):
            for name in names[turn % len(names) :] + names[: turn % len(names)]:
                graph, last = build(image, name)
                try:
                    for _ in range(3):
                        graph.run()
                    gc.collect()
                    for _ in range(iterations):
                        started = time.perf_counter()
                        graph.run()
                        measurements[name].append((time.perf_counter() - started) * 1000)
                    digests.add(
                        hashlib.sha256(last.last_result.outputs["image"].tobytes()).hexdigest()
                    )
                finally:
                    graph.close()
                del graph, last
                gc.collect()
    finally:
        cv2.setNumThreads(threads)
    if len(digests) != 1:
        raise AssertionError("Run profiles changed the processing output")
    baseline = statistics.median(measurements["debug"])
    return {
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "image_size": [width, height],
        "samples_per_profile": iterations * rounds,
        "opencv_threads": 1,
        "output_sha256": next(iter(digests)),
        "profiles": {
            name: {
                "median_ms": statistics.median(values),
                "speedup_vs_debug": baseline / statistics.median(values),
            }
            for name, values in measurements.items()
        },
        "scope": "Headless graph execution; includes acquisition copy, validation, "
        "payloads and statistics. GUI throttling/pacing gains are excluded.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.iterations, args.rounds, args.width, args.height) < 1:
        parser.error("Counts and image dimensions must be positive")
    if args.output and args.output.exists():
        parser.error("Output exists; choose a new file")
    result = benchmark(args.width, args.height, args.iterations, args.rounds)
    content = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(content)
    print(content)
