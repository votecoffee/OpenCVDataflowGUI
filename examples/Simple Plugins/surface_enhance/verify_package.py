"""Replay retained controls through the wheel without installing into the host."""

import argparse
import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np


def verify(wheel, studies, *, source=None):
    wheel, studies = wheel.resolve(), studies.resolve()
    with zipfile.ZipFile(wheel) as archive:
        modules = {
            name: archive.read(name)
            for name in archive.namelist()
            if name.startswith("opencv_simple_surface_enhance/") and name.endswith(".py")
        }
        assert {"__init__", "inputs", "solver", "cli", "render"} <= {
            Path(name).stem for name in modules
        }, "Missing runtime modules"
        assert any(name.endswith("/LICENSE") for name in archive.namelist())
        entry = next(name for name in archive.namelist() if name.endswith("/entry_points.txt"))
        assert b"simple-surface-enhance = opencv_simple_surface_enhance.cli:main" in (
            archive.read(entry)
        )
        if source is not None:
            for name, raw in modules.items():
                assert (source / name).read_bytes() == raw, name
    sys.path.insert(0, str(wheel))
    import opencv_simple_surface_enhance as package
    from opencv_simple_surface_enhance.cli import main

    assert str(wheel) in package.__file__, "Not running the selected wheel"
    cases = json.loads((studies / "summary.json").read_bytes())["cases"]
    checks = []
    with tempfile.TemporaryDirectory(prefix="simple_surface_verify_") as temporary:
        for case in cases:
            folder = studies / case["case"]
            output = Path(temporary) / case["case"]
            main([str(folder / "capture.json"), "--output", str(output), "--no-graphics"])
            exact = True
            with (
                np.load(folder / "result/maps.npz", allow_pickle=False) as expected,
                np.load(output / "maps.npz", allow_pickle=False) as actual,
            ):
                assert set(actual.files) == set(expected.files)
                for name in expected.files:
                    left, right = expected[name], actual[name]
                    assert left.dtype == right.dtype and left.shape == right.shape, name
                    if left.dtype.kind == "f":
                        np.testing.assert_allclose(left, right, rtol=0, atol=1e-6, equal_nan=True)
                        exact &= np.array_equal(left, right, equal_nan=True)
                    else:
                        np.testing.assert_array_equal(left, right)
            report = json.loads((output / "report.json").read_bytes())
            assert report["valid_pixels"] == case["valid_pixels"]
            assert report["reason_counts"] == case["reason_counts"]
            checks.append(
                dict(
                    case=case["case"],
                    numerical_maps=sorted(expected.files),
                    absolute_float_tolerance=1e-6,
                    exact_equal_including_nan=bool(exact),
                    reason_and_support_exact=True,
                    cli_reexecuted=True,
                    passed=True,
                )
            )
    return dict(
        schema="simple_surface_package_verification/v1",
        scope="Rendered-input replay; not acquired-camera or physical accuracy qualification",
        wheel=wheel.name,
        sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
        source_bytes_checked=source is not None,
        modules={name: hashlib.sha256(raw).hexdigest() for name, raw in modules.items()},
        controls=checks,
        passed=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("studies", type=Path)
    parser.add_argument(
        "--source", type=Path, help="Optional source package parent for byte checks"
    )
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    result = verify(args.wheel, args.studies, source=args.source)
    with args.receipt.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(wheel=result["wheel"], cases=len(result["controls"]), passed=True)))
