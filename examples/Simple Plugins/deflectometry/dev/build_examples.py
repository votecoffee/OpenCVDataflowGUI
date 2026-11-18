"""Generate independent phase controls and replay an optional attributed real capture."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from opencv_simple_deflectometry.cli import read_capture, write_result
from opencv_simple_deflectometry.core import CompiledReference


def write_capture(output, name, images, *, provenance):
    directory = output / name
    directory.mkdir()
    shifts = np.arange(6) * 2 * np.pi / 6
    axes, frames = {}, []
    for axis, stack in zip("xy", images):
        axes[axis] = dict(
            frame_ids=[f"{axis}:{i}" for i in range(6)],
            phase_angles=shifts.tolist(),
            period_px=32,
            origin_rad=0,
            direction=1,
        )
        for i, plane in enumerate(stack):
            file = f"{axis}_{i:02d}.npy"
            np.save(directory / file, plane.astype(np.float32))
            frames.append(dict(id=f"{axis}:{i}", image=file))
    data = dict(
        schema="simple_deflectometry_capture/v1",
        capture_id=name,
        sequence=dict(
            setup_id="analytic-phase-control",
            pixel_grid="original-96x72",
            exposure_s=0.01,
            axes=axes,
        ),
        source_encoding="linear",
        input_scale=1,
        frames=frames,
        provenance=provenance,
    )
    (directory / "capture.json").write_bytes((json.dumps(data, indent=2) + "\n").encode())
    return directory / "capture.json"


def build(output, *, sofast=None):
    output.mkdir(parents=True, exist_ok=False)
    y, x = np.indices((72, 96))
    phase = np.stack((2 * np.pi * x / 32 + 0.3, 2 * np.pi * y / 32 + 0.5))
    bump = np.exp(-((x - 48) ** 2 / 160 + (y - 36) ** 2 / 100))
    shifts = np.arange(6) * 2 * np.pi / 6

    def signals(changes):
        return np.stack(
            [
                np.stack([0.4 + 0.2 * np.cos(p + d + s) for s in shifts])
                for p, d in zip(phase, changes)
            ]
        )

    reference_path = write_capture(
        output,
        "reference",
        signals((0, 0)),
        provenance=dict(kind="Analytic phase control; not physical shape"),
    )
    reference, reference_data = read_capture(reference_path)
    compiled = CompiledReference(reference)
    rows = []
    for case in ("analytic_clean", "analytic_change", "analytic_missing"):
        delta = (
            np.stack((0.8 * bump, -0.5 * bump))
            if case == "analytic_change"
            else np.zeros_like(phase)
        )
        images = signals(delta)
        if case == "analytic_missing":
            images[0, 0, 5:25, 10:30] = np.nan
            images[1, 1, 40:55, 50:70] = 1
        manifest = write_capture(
            output,
            case,
            images,
            provenance=dict(
                kind="Analytic phase/missing-signal control; no acquired or metric shape truth",
            ),
        )
        capture, capture_data = read_capture(manifest)
        result = compiled.compare(capture, minimum_coverage=0.95, maximum_change_rad=0.45)
        expected = case == "analytic_clean"
        assert result["report"]["passed"] == expected
        valid = result["maps"]["valid"]
        np.testing.assert_allclose(
            result["maps"]["change_rad"][valid], delta.transpose(1, 2, 0)[valid], rtol=0, atol=2e-7
        )
        np.savez_compressed(output / case / "truth.npz", change_rad=delta.transpose(1, 2, 0))
        write_result(
            output / case / "result",
            result,
            source_records=dict(
                reference=reference_data["provenance"], test=capture_data["provenance"]
            ),
        )
        rows.append(
            dict(
                case=case,
                reference="reference/capture.json",
                capture=f"{case}/capture.json",
                expected_pass=expected,
                actual_pass=result["report"]["passed"],
                comparable=result["report"]["comparable_pixels"],
                coverage=result["report"]["coverage"],
                independent_phase_truth=True,
                physical_height_truth=False,
            )
        )
    if sofast:
        capture, metadata = read_capture(sofast / "capture.json")
        compiled = CompiledReference(capture)
        result = compiled.compare(capture, minimum_coverage=0.95, maximum_change_rad=0.45)
        directory = output / "sofast_self_review"
        directory.mkdir()
        write_result(
            directory / "result",
            result,
            source_records=dict(reference=metadata["provenance"], test=metadata["provenance"]),
        )
        np.testing.assert_array_equal(result["maps"]["change_rad"][result["maps"]["valid"]], 0)
        path = Path(os.path.relpath(sofast / "capture.json", output)).as_posix()
        rows.append(
            dict(
                case="sofast_self_review",
                reference=path,
                capture=path,
                expected_pass=result["report"]["passed"],
                actual_pass=result["report"]["passed"],
                comparable=result["report"]["comparable_pixels"],
                coverage=result["report"]["coverage"],
                independent_phase_truth=False,
                physical_height_truth=False,
                scope="Same real exported capture, code diagnostics",
            )
        )
    (output / "summary.json").write_bytes((json.dumps(rows, indent=2) + "\n").encode())
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sofast", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.output, sofast=args.sofast), indent=2))
