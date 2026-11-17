"""Replay three retained examples from the selected wheel and check raw map parity."""

import argparse
import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np


def verify(wheel, studies, source=None):
    wheel = wheel.resolve()
    with zipfile.ZipFile(wheel) as archive:
        modules = {
            name: archive.read(name)
            for name in archive.namelist()
            if name.startswith("opencv_simple_fringe_scan/") and name.endswith(".py")
        }
        assert {"__init__", "patterns", "phase", "decode", "geometry", "cli", "render"} <= {
            Path(name).stem for name in modules
        }
        assert any(name.endswith("/LICENSE") for name in archive.namelist())
        entry = next(name for name in archive.namelist() if name.endswith("/entry_points.txt"))
        assert b"simple-fringe-scan = opencv_simple_fringe_scan.cli:main" in archive.read(entry)
        if source:
            for name, raw in modules.items():
                assert (source / name).read_bytes() == raw, name
    sys.path.insert(0, str(wheel))
    import opencv_simple_fringe_scan as package
    from opencv_simple_fringe_scan.cli import main

    assert str(wheel) in package.__file__
    checks = []
    with tempfile.TemporaryDirectory(prefix="fringe_package_") as temporary:
        for case in json.loads((studies / "summary.json").read_bytes()):
            name = case["case"]
            output = Path(temporary) / name
            main([str(studies / name / "capture.json"), "--output", str(output), "--no-graphics"])
            exact = True
            with (
                np.load(output / "maps.npz", allow_pickle=False) as actual,
                np.load(studies / name / "result/maps.npz", allow_pickle=False) as expected,
            ):
                assert set(actual.files) == set(expected.files)
                for key in expected.files:
                    left, right = actual[key], expected[key]
                    if left.dtype.kind == "f":
                        np.testing.assert_allclose(left, right, atol=1e-6, rtol=0, equal_nan=True)
                        exact &= np.array_equal(left, right, equal_nan=True)
                    else:
                        np.testing.assert_array_equal(left, right)
            report = json.loads((output / "report.json").read_bytes())
            assert report["valid_pixels"] == case["report"]["valid_pixels"]
            assert report["metric"] == case["report"]["metric"]
            assert (output / "accepted.ply").exists() == report["metric"]
            checks.append(
                dict(
                    case=name,
                    raw_maps=len(expected.files),
                    exact_maps_including_nan=bool(exact),
                    absolute_float_tolerance=1e-6,
                    discrete_reasons_exact=True,
                    cli_reexecuted=True,
                    metric=report["metric"],
                    passed=True,
                )
            )
    return dict(
        wheel=wheel.name,
        sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
        runtime_modules=len(modules),
        source_bytes_checked=source is not None,
        controls=checks,
        scope="Software replay, not acquired-camera accuracy",
        passed=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("studies", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    result = verify(args.wheel, args.studies, args.source)
    with args.receipt.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(cases=len(result["controls"]), passed=True)))
