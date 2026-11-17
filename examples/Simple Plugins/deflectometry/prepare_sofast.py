"""Inspect only numeric/image members from the pinned official SOFAST ZIP; no code execution."""

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ZIP_SHA256 = "ce027803717c3e97cc4e4481b53e7ac6a90e822c151d1c850f4023233e24b5ff"
PREFIX = "NSTTF_Facet_SOFAST_Landscape/SOFAST_Fringe/"
MASK = "DataSofastCalculation/general/CalculationImageProcessingGeneral/mask_raw"


def prepare(archive, license_file, output):
    # h5py is a preparation-only dependency; estimator and published reviews do not use it.
    import h5py

    if archive.stat().st_size != 83154541:
        raise ValueError("Official sample archive size differs")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ZIP_SHA256:
        raise ValueError("Official sample archive hash differs")
    license_raw = license_file.read_bytes()
    if hashlib.sha1(license_raw).hexdigest() != "63b5eeee74062d1ea1fb194bfcce0a81afe58f74":
        raise ValueError("Official separate data-license bytes differ")
    output.mkdir(parents=True, exist_ok=False)
    originals = output / "originals"
    originals.mkdir()
    (originals / "OpenCSP_Box_license.txt").write_bytes(license_raw)
    selected = []
    with zipfile.ZipFile(archive) as zipped:

        def member(relative, *, retained=True):
            info = zipped.getinfo(PREFIX + relative)
            if info.file_size > 80_000_000:
                raise ValueError("Selected numeric/image member exceeds preparation bound")
            raw = zipped.read(info)
            selected.append(
                dict(
                    member=info.filename,
                    bytes=len(raw),
                    crc32=f"{info.CRC:08x}",
                    sha256=hashlib.sha256(raw).hexdigest(),
                    retained_original=retained,
                )
            )
            return raw

        for folder in ("1_images_fringes_projected", "2_images_captured"):
            directory = originals / folder
            directory.mkdir()
            for axis in "xy":
                for i in range(4):
                    name = f"{axis}_{i:02d}.png"
                    (directory / name).write_bytes(member(folder + "/" + name))
        for i in range(2):
            name = f"mask_{i:02d}.png"
            (originals / "2_images_captured" / name).write_bytes(
                member("2_images_captured/" + name)
            )
        (originals / "README.txt").write_bytes(member("README.txt"))
        raw = member("3_processed_data/data_sofast_processed.h5", retained=False)
        with h5py.File(io.BytesIO(raw), "r") as handle:
            group = handle
            for key in MASK.split("/"):
                if not isinstance(group.get(key, getlink=True), h5py.HardLink):
                    raise ValueError("Only direct numeric HDF5 data is allowed")
                group = group[key]
            if group.shape != (1236, 1626) or group.dtype != bool:
                raise ValueError("Upstream raw-region mask shape/type differs")
            mask = group[...]
    np.save(output / "upstream_mask_raw.npy", mask)
    reduced_mask = mask.reshape(618, 2, 813, 2).all(axis=(1, 3))
    np.save(output / "mask.npy", reduced_mask)
    command_checks = {}
    for axis in "xy":
        commands = np.stack(
            [
                np.asarray(
                    Image.open(originals / "1_images_fringes_projected" / f"{axis}_{i:02d}.png"),
                    dtype=float,
                )
                for i in range(4)
            ]
        )
        phase = np.arctan2(commands[1] - commands[3], commands[0] - commands[2])
        amplitude = 0.5 * np.hypot(commands[0] - commands[2], commands[1] - commands[3])
        predicted = np.stack(
            [commands.mean(axis=0) + amplitude * np.cos(phase - i * np.pi / 2) for i in range(4)]
        )
        error = float(np.max(np.abs(predicted - commands)))
        line = phase[0] if axis == "x" else phase[:, 0]
        cycles = float((np.unwrap(line)[-1] - np.unwrap(line)[0]) / (2 * np.pi))
        assert error <= 1 and -0.91 < cycles < -0.89
        command_checks[axis] = dict(
            maximum_four_step_error_codes=error, inferred_cycles_in_thumbnail=cycles
        )
    frames = []
    for axis in "xy":
        for i in range(4):
            source = originals / "2_images_captured" / f"{axis}_{i:02d}.png"
            codes = np.asarray(Image.open(source))
            if codes.shape != (1236, 1626) or codes.dtype != np.uint8:
                raise ValueError("Retained source frame shape/type differs")
            # Explicit code-value area average, not a calibrated radiometric conversion.
            reduced = (codes.astype(np.float32) / 255).reshape(618, 2, 813, 2).mean(axis=(1, 3))
            ceiling = (codes == 255).reshape(618, 2, 813, 2).any(axis=(1, 3))
            reduced[ceiling] = 1.0
            file = f"{axis}_{i:02d}.npy"
            np.save(output / file, reduced)
            frames.append(dict(id=f"{axis}:{i}", image=file))
    provenance = dict(
        creator="OpenCSP Team, Sandia National Laboratories NSTTF",
        title="SOFAST Landscape measurement of an NSTTF heliostat facet",
        project="https://opencsp.sandia.gov/",
        license="CC BY-SA 4.0",
        source="https://sandia-csp.app.box.com/s/iftmhdkhgjnmfgefsfmkid011dtv8n3g/file/1951928463619",
        archive_sha256=ZIP_SHA256,
        archive_publisher_sha1_verified=True,
        selected_members=selected,
        mask_dataset=MASK,
        transformation=dict(
            source_size=[1626, 1236],
            derived_size=[813, 618],
            source_from_derived=[[2, 0, 0.5], [0, 2, 0.5], [0, 0, 1]],
            operation="Normalize byte codes by 255; 2x2 mean; any source ceiling remains ceiling",
            mask="Upstream raw region, 2x2 all-true reduction; not defect ground truth",
        ),
        sequence_scope="First four exported X/Y coarse-frequency frames. Negative pi/2 shifts "
        "verified against retained command thumbnails and upstream export/generator.",
        command_checks=command_checks,
        limitations="Unknown exposure/response and original physical screen pixel pitch. "
        "Code-value diagnostics only. Same capture self-comparison, not a held-out defect test. "
        "No independently measured surface reference or metric reconstruction.",
    )
    sequence = dict(
        setup_id="sofast-landscape-exported-coarse",
        pixel_grid="sofast-area2-code-values",
        exposure_s=None,
        response="uncalibrated_codes",
        axes={
            axis: dict(
                frame_ids=[f"{axis}:{i}" for i in range(4)],
                phase_angles=[float(-i * np.pi / 2) for i in range(4)],
                period_px=None,
                origin_rad=None,
                direction=-1,
            )
            for axis in "xy"
        },
    )
    manifest = dict(
        schema="simple_deflectometry_capture/v1",
        capture_id="sofast-exported-facet",
        sequence=sequence,
        source_encoding="uncalibrated_codes",
        input_scale=1,
        mask="mask.npy",
        frames=frames,
        provenance=provenance,
    )
    (output / "capture.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode())
    (output / "provenance.json").write_bytes((json.dumps(provenance, indent=2) + "\n").encode())
    return dict(
        retained_original_images=18,
        active_pixels=int(reduced_mask.sum()),
        derived_size=[813, 618],
        license="CC BY-SA 4.0",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("license", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.license, args.output)))
