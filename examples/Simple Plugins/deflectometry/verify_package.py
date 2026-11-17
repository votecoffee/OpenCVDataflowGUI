"""Replay all controls from two selected wheels and compare every raw map."""

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from run_capture import select


def verify(wheel, fringe_wheel, studies, *, source=None, fringe_source=None):
    modules = {}
    for package, path, tree in (
        ("opencv_simple_deflectometry", wheel, source),
        ("opencv_simple_fringe_scan", fringe_wheel, fringe_source),
    ):
        with zipfile.ZipFile(path) as archive:
            files = [
                name
                for name in archive.namelist()
                if name.startswith(package + "/") and name.endswith(".py")
            ]
            assert files and any(name.endswith("/LICENSE") for name in archive.namelist())
            if tree:
                for name in files:
                    assert (tree / name).read_bytes() == archive.read(name)
            modules[package] = len(files)
    select(wheel, fringe_wheel)
    from opencv_simple_deflectometry.cli import main

    checks = []
    with tempfile.TemporaryDirectory(prefix="deflectometry_package_") as temporary:
        for row in json.loads((studies / "summary.json").read_bytes()):
            case = row["case"]
            output = Path(temporary) / case
            main(
                [
                    str(studies / row["reference"]),
                    str(studies / row["capture"]),
                    "--minimum-coverage",
                    ".95",
                    "--maximum-change-rad",
                    ".45",
                    "--no-graphics",
                    "--output",
                    str(output),
                ]
            )
            with (
                np.load(studies / case / "result/maps.npz", allow_pickle=False) as expected,
                np.load(output / "maps.npz", allow_pickle=False) as actual,
            ):
                assert set(expected.files) == set(actual.files)
                for key in expected.files:
                    np.testing.assert_array_equal(actual[key], expected[key])
                count = len(expected.files)
            report = json.loads((output / "report.json").read_bytes())
            assert report == json.loads((studies / case / "result/report.json").read_bytes())
            assert report["passed"] == row["expected_pass"]
            assert not list(output.glob("*.ply")) and not list(output.glob("*.png"))
            checks.append(
                dict(
                    case=case,
                    raw_maps=count,
                    exact_maps_and_report=True,
                    expected_pass=report["passed"],
                    comparable=report["comparable_pixels"],
                    actual_cli_reexecuted=True,
                    metric_height=False,
                )
            )
    return dict(
        wheel=wheel.name,
        sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
        fringe_wheel=fringe_wheel.name,
        fringe_sha256=hashlib.sha256(fringe_wheel.read_bytes()).hexdigest(),
        runtime_modules=modules,
        source_bytes_checked=bool(source and fringe_source),
        controls=checks,
        scope="Software replay, not physical measurement accuracy",
        passed=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("fringe_wheel", type=Path)
    parser.add_argument("studies", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--fringe-source", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    result = verify(
        args.wheel,
        args.fringe_wheel,
        args.studies,
        source=args.source,
        fringe_source=args.fringe_source,
    )
    args.receipt.write_bytes((json.dumps(result, indent=2) + "\n").encode())
    print(json.dumps(dict(cases=len(result["controls"]), passed=True)))
