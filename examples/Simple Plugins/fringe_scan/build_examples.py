"""Fixed analytic ray controls and an attributed external simulated six-step scan."""

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from opencv_simple_fringe_scan import PatternPlan
from opencv_simple_fringe_scan.cli import export, read_manifest, run


def write_case(root, name, data, images, *, truth=None):
    folder = root / name
    folder.mkdir()
    frames = []
    identifiers = data.pop("frame_ids")
    for index, (identifier, image) in enumerate(zip(identifiers, images)):
        filename = f"frame_{index:02d}.npy"
        np.save(folder / filename, image, allow_pickle=False)
        frames.append(dict(id=identifier, image=filename))
    data["frames"] = frames
    (folder / "capture.json").write_bytes((json.dumps(data, indent=2) + "\n").encode())
    definition, loaded, mask = read_manifest(folder / "capture.json")
    result = run(definition, loaded, mask)
    export(folder / "result", result)
    report = dict(case=name, report=result["report"], graphics_views=sorted(result["views"]))
    if truth is not None:
        np.save(folder / "xyz_truth_m.npy", truth, allow_pickle=False)
        valid = result["maps"]["geometry_valid"]
        error = np.linalg.norm(result["maps"]["geometry_xyz_m"][valid] - truth[valid], axis=-1)
        report["mean_render_reference_error_m"] = float(np.mean(error))
        report["maximum_render_reference_error_m"] = float(np.max(error))
        report["reference_scope"] = (
            "Independent analytic ray construction, not physical gauge accuracy"
        )
    return report


def analytic():
    plan = PatternPlan((128, 96), period_px=(16, 16), phase_steps=6)
    yy, xx = np.indices((72, 96))
    xn, yn = (xx - 48) / 120, (yy - 36) / 120
    depth = 0.75 / (1 + 0.12 * xn + 0.05 * yn)
    xyz = np.stack((xn * depth, yn * depth, depth), -1)
    projector_x = 120 * (xyz[:, :, 0] - 0.08) / depth + 64
    projector_y = 120 * xyz[:, :, 1] / depth + 48
    raw = [np.full(depth.shape, 0.08), np.full(depth.shape, 0.82)]
    for coordinate, extent in ((projector_x, 128), (projector_y, 96)):
        integer = np.floor(coordinate + 0.5).astype(np.uint16)
        gray = integer ^ (integer >> 1)
        for bit in range((extent - 1).bit_length() - 1, -1, -1):
            signal = ((gray >> bit) & 1) * 0.74
            raw.extend((0.08 + signal, 0.82 - signal))
    for coordinate in (projector_x, projector_y):
        for shift in plan.phase_angles:
            raw.append(0.45 + 0.32 * np.cos(2 * np.pi * coordinate / 16 + shift))
    data = dict(
        schema="simple_fringe_capture/v1",
        frame_ids=plan.pattern_ids,
        plan=asdict(plan),
        capture_id="analytic_plane",
        pixel_grid="rendered_camera_original",
        exposure_s=0.01,
        input_scale=1,
        source_encoding="linear",
        calibration=dict(
            camera=dict(
                matrix=[[120, 0, 48], [0, 120, 36], [0, 0, 1]], image_size=[96, 72], frame="camera"
            ),
            projector=dict(
                matrix=[[120, 0, 64], [0, 120, 48], [0, 0, 1]],
                image_size=[128, 96],
                frame="projector",
            ),
            rotation_cp=np.eye(3).tolist(),
            translation_cp_m=[-0.08, 0, 0],
        ),
        provenance=dict(
            source="Original analytic tilted plane; one first visible surface per camera ray",
            scope="Rendered software control, not acquired-camera or physical accuracy evidence",
        ),
    )
    return data, np.asarray(raw, np.float32), xyz


def build(output, warsaw):
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for damaged in (False, True):
        data, images, truth = analytic()
        name = "analytic_missing" if damaged else "analytic_clean"
        data["capture_id"] = name
        if damaged:
            images[4, 12:22, 14:25] = np.nan
            images[6, 30:38, 20:35] = 1
            images[1, 45:54, 50:63] = images[0, 45:54, 50:63]
            first_phase = 2 + len(PatternPlan(**data["plan"]).gray_ids)
            images[first_phase : first_phase + 6, 20:30, 60:75] = 0.45
        rows.append(write_case(output, name, data, images, truth=truth))
    if warsaw:
        originals = output / "warsaw_source"
        originals.mkdir()
        shutil.copytree(warsaw / "crown", originals / "crown")
        shutil.copyfile(warsaw / "provenance.json", originals / "provenance.json")
        patterns = warsaw.parent / "warsaw_patterns"
        shutil.copytree(patterns, originals / "projector_patterns")
        # Preserve small scene declarations; never execute third-party renderer/code files.
        base = warsaw / "scanner-simulation"
        for relative in (
            "include/cameras/film-2592-1944.pbrt",
            "include/camera.pbrt",
            "scenes-2592-1944-640-480/crown/0.pbrt",
        ):
            target = originals / "scene" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(base / relative, target)
        planes = []
        for index in range(6):
            code = cv2.imread(str(warsaw / "crown" / f"{index}.png"), cv2.IMREAD_UNCHANGED)
            assert code.shape == (1944, 2592, 3) and code.dtype == np.uint8
            green = code[:, :, 1].astype(np.float32) / 255
            linear = np.where(green <= 0.04045, green / 12.92, ((green + 0.055) / 1.055) ** 2.4)
            plane = linear.reshape(648, 3, 864, 3).mean(axis=(1, 3))
            clipped = (green >= 1).reshape(648, 3, 864, 3).any(axis=(1, 3))
            plane[clipped] = 1  # Preserve any original input-ceiling state through binning.
            planes.append(plane)
        transform = dict(
            source_size=[2592, 1944],
            derived_size=[864, 648],
            channel="green",
            source_encoding="Declared PBRT-v4 standard PNG sRGB output, inverse applied",
            radiometry_scope="Inference from supplied PBRT rgb Film/PNG scene and standard writer; "
            "exact renderer build/photometric qualification unavailable",
            operation="Linearize then 3x3 area average; any original ceiling propagates as ceiling",
            source_from_derived=[[3, 0, 1], [0, 3, 1], [0, 0, 1]],
            phase_angles_rad=(2 * np.pi * (np.arange(6) + 2) / 6).tolist(),
            phase_origin="Six source projector commands: +pi/3 step, initial 2pi/3, period6px in X",
            geometry_scope="Wrapped phase only; reconstruction calibration not imported",
        )
        (originals / "transform.json").write_bytes(
            (json.dumps(transform, indent=2) + "\n").encode()
        )
        data = dict(
            schema="simple_fringe_phase/v1",
            frame_ids=[f"phase:{i}" for i in range(6)],
            capture_id="warsaw_crown_phase",
            pixel_grid="warsaw_perspective_green_area3",
            exposure_s=1.0,
            input_scale=1,
            source_encoding="linear",
            phase_angles=transform["phase_angles_rad"],
            provenance=dict(
                source="Michal Wlasiuk / Robert Sitnik; "
                "Simulated structured light 3D scanner datasets",
                record="https://zenodo.org/records/17826191",
                license="CC BY 4.0",
                image_kind="External PBRT simulated images, not acquired camera captures",
                transformation=transform,
            ),
        )
        rows.append(write_case(output, "warsaw_crown_phase", data, np.stack(planes)))
    (output / "summary.json").write_bytes((json.dumps(rows, indent=2) + "\n").encode())
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--warsaw", type=Path)
    args = parser.parse_args()
    records = build(args.output, args.warsaw)
    print(
        json.dumps(
            [
                dict(
                    case=r["case"],
                    coverage=r["report"]["coverage"],
                    valid_pixels=r["report"]["valid_pixels"],
                    metric=r["report"]["metric"],
                    error_m=r.get("maximum_render_reference_error_m"),
                )
                for r in records
            ]
        )
    )
