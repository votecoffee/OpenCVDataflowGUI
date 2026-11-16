"""Alternating paired preparation benchmark; same candidates and exact crop outputs."""

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from opencv_simple_gdrnpp.preprocess import prepare_frame
from opencv_simple_gdrnpp.workflow import load_capture


def reference_module(path):
    spec = importlib.util.spec_from_file_location("frozen_crop_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def compare(a, b):
    assert len(a) == len(b)
    for first, second in zip(a, b):
        for field in dataclasses.fields(first):
            x, y = getattr(first, field.name), getattr(second, field.name)
            if isinstance(x, np.ndarray):
                assert x.dtype == y.dtype
                np.testing.assert_array_equal(x, y)
            else:
                assert x == y


def run(capture_path, reference, output, *, repeats=9, inner=20):
    if output.exists():
        raise FileExistsError("Choose a fresh preprocessing receipt")
    if not 3 <= repeats <= 15 or not 1 <= inner <= 100:
        raise ValueError("Use 3..15 alternating pairs with 1..100 calls per arm")
    old = reference_module(reference)
    capture = load_capture(capture_path)
    rows = []
    for factor in (1, 2):
        original = capture.frame.rgb
        rgb = (
            original
            if factor == 1
            else cv2.resize(original, (original.shape[1] * factor, original.shape[0] * factor))
        )
        camera = capture.frame.camera.copy()
        camera[:2] *= factor
        boxes = [np.asarray(d.box) * factor for d in capture.detections]

        def reference_call():
            return tuple(old.prepare_crop(rgb, camera, box) for box in boxes)

        def shared_call():
            prepared = prepare_frame(rgb, camera)
            return tuple(prepared.crop(box) for box in boxes)

        compare(reference_call(), shared_call())
        samples = dict(reference=[], shared=[])
        functions = dict(reference=reference_call, shared=shared_call)
        for pair in range(repeats):
            for mode in ("reference", "shared") if pair % 2 == 0 else ("shared", "reference"):
                start = time.perf_counter()
                for _ in range(inner):
                    crops = functions[mode]()
                samples[mode].append((time.perf_counter() - start) / inner)
            compare(crops, reference_call())
        before, after = (statistics.median(samples[k]) for k in ("reference", "shared"))
        row = dict(
            source_size=list(rgb.shape[1::-1]),
            candidate_count=len(boxes),
            derived_resize_factor=factor,
            samples_seconds=samples,
            reference_median_seconds=before,
            shared_median_seconds=after,
            preparation_speedup=before / after,
            all_crop_fields_bit_identical=True,
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    receipt = dict(
        reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        capture_sha256=hashlib.sha256(capture_path.read_bytes()).hexdigest(),
        source_rgb_sha256=hashlib.sha256(capture.frame.rgb.tobytes()).hexdigest(),
        camera=capture.frame.camera.tolist(),
        pairs=repeats,
        calls_per_arm=inner,
        cases=rows,
        scope=(
            "Full-frame preparation and all candidate crops only; excludes model, "
            "ICP/fit, IO, graphics and acquisition. 2x case is a resized copy, "
            "not a new acquisition."
        ),
        attribution="T-LESS, Hodan et al., WACV 2017, CC BY 4.0",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "reference", "new_receipt"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.capture, args.reference, args.new_receipt)
