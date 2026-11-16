"""Paired private-file transport timings: frozen legacy, fast deflate and stored streaming."""

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from opencv_simple_gdrnpp import wire
from opencv_simple_gdrnpp.workflow import load_capture

REFERENCE_SHA256 = "25fd3f6c42f209cd84b2bdf786f9e40a626179238d5a253bc7e553ba83197222"


def reference_module(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise ValueError("Select the frozen, audited transport reference")
    spec = importlib.util.spec_from_file_location("packet_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixtures(examples):
    capture = load_capture(examples / "preprocessing/capture.json")
    yield (
        "real_rgbd_input",
        dict(capture=capture.capture_id),
        dict(
            rgb=capture.frame.rgb,
            depth=capture.frame.depth,
            camera=capture.frame.camera,
            mask=capture.frame.mask,
        ),
    )
    folder = examples / "batch_execution/results"
    raw = (folder / "batch1.ply").read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    count = int(re.search(rb"element vertex (\d+)", raw[:end])[1])
    data = np.frombuffer(raw, dtype=[("xyz", "<f4", (3,)), ("rgb", "u1", (3,))], offset=end)
    assert len(data) == count
    yield (
        "real_overlay_cloud",
        json.loads((folder / "batch1.json").read_bytes()),
        dict(
            overlay=cv2.imread(str(folder / "batch1.png"))[..., ::-1].copy(),
            points=data["xyz"].copy(),
            colors=data["rgb"].copy(),
        ),
    )
    yield (
        "real_strided_rgbd",
        dict(derived="every other source row/column"),
        dict(
            rgb=capture.frame.rgb[::2, ::2],
            depth=capture.frame.depth[::2, ::2],
        ),
    )
    yield "empty_numeric_reply", dict(inspection_performed=False), {}
    rng = np.random.default_rng(13027)
    yield (
        "seeded_noise_stress",
        dict(synthetic="transport-only entropy stress"),
        dict(rgb=rng.integers(0, 256, (1024, 1536, 3), np.uint8)),
    )
    yield (
        "zero_surface_stress",
        dict(synthetic="transport-only compression extreme"),
        dict(depth=np.zeros((1024, 2048), np.float64)),
    )
    yield (
        "fortran_numeric",
        dict(synthetic="transport-only array layout"),
        dict(values=np.asfortranarray(rng.random((1024, 512), dtype=np.float32))),
    )


def run(examples, reference, output, scratch):
    if output.exists():
        raise FileExistsError("Choose a fresh packet-timing receipt")
    scratch.mkdir(parents=True, exist_ok=False)
    scratch = scratch.resolve()
    old = reference_module(reference)
    methods = dict(
        legacy=(old.write_packet, old.read_packet),
        deflated_fast=(
            lambda d, m, a: wire.write_packet(d, m, a, compression="deflated_fast"),
            wire.read_packet,
        ),
        stored=(wire.write_packet, wire.read_packet),
    )
    rows, summary = [], []
    for name, metadata, arrays in fixtures(examples):
        for round_number in range(8):
            order = list(methods) if round_number % 2 == 0 else list(reversed(methods))
            for method in order:
                directory = Path(tempfile.mkdtemp(dir=scratch)).resolve()
                assert directory.is_relative_to(scratch)
                try:
                    write, read = methods[method]
                    started = time.perf_counter()
                    write(directory, metadata, arrays)
                    written = time.perf_counter()
                    actual, returned = read(directory)
                    read_done = time.perf_counter()
                    assert actual == metadata and returned.keys() == arrays.keys()
                    for key in arrays:
                        np.testing.assert_array_equal(returned[key], arrays[key])
                        assert returned[key].dtype == arrays[key].dtype
                        assert returned[key].flags.writeable
                    # New packets must also decode through the unchanged legacy reader.
                    legacy_metadata, legacy = old.read_packet(directory)
                    assert legacy_metadata == metadata
                    for key in arrays:
                        np.testing.assert_array_equal(legacy[key], arrays[key])
                    disk = sum(p.stat().st_size for p in directory.iterdir())
                finally:
                    clean = time.perf_counter()
                    assert directory.is_relative_to(scratch)
                    shutil.rmtree(directory)
                    cleanup = time.perf_counter() - clean
                rows.append(
                    dict(
                        fixture=name,
                        method=method,
                        round=round_number,
                        warmup=round_number == 0,
                        write_seconds=written - started,
                        read_seconds=read_done - written,
                        cleanup_seconds=cleanup,
                        write_read_seconds=read_done - started,
                        numeric_bytes=sum(a.nbytes for a in arrays.values()),
                        disk_bytes=disk,
                        values_dtypes_and_legacy_reader_exact=True,
                    )
                )
        for method in methods:
            samples = [
                r
                for r in rows
                if r["fixture"] == name and r["method"] == method and not r["warmup"]
            ]
            summary.append(
                dict(
                    fixture=name,
                    method=method,
                    samples=len(samples),
                    median_write_ms=statistics.median(r["write_seconds"] for r in samples) * 1000,
                    median_read_ms=statistics.median(r["read_seconds"] for r in samples) * 1000,
                    median_write_read_ms=statistics.median(r["write_read_seconds"] for r in samples)
                    * 1000,
                    minimum_write_read_ms=min(r["write_read_seconds"] for r in samples) * 1000,
                    maximum_write_read_ms=max(r["write_read_seconds"] for r in samples) * 1000,
                    disk_bytes=samples[0]["disk_bytes"],
                    numeric_bytes=samples[0]["numeric_bytes"],
                )
            )
        print(name, json.dumps(summary[-3:]), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(
            dict(
                summary=summary,
                samples=rows,
                reference_sha256=REFERENCE_SHA256,
                candidate_wire_sha256=hashlib.sha256(Path(wire.__file__).read_bytes()).hexdigest(),
                scope=(
                    "One warmup and seven alternating measured rounds per fixture/method. "
                    "Private local files, encode/decode timing only; "
                    "correctness and cleanup separate. "
                    "Real T-LESS inputs/results plus explicitly synthetic transport stress arrays. "
                    "Not whole-cycle, capacity, scene accuracy or proof "
                    "of eliminating OS/file stalls."
                ),
                attribution="T-LESS, Hodan et al., WACV 2017, CC BY 4.0",
            ),
            stream,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("examples", "reference", "new_output", "new_scratch"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.examples, args.reference, args.new_output, args.new_scratch)
