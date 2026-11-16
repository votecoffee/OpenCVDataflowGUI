"""Fixed analytic and CC0 material-map diffuse-render controls, with reference errors."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from opencv_simple_surface_enhance import SurfaceEstimator
from opencv_simple_surface_enhance.cli import export_result, read_capture


def write_json(path, data):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + "\n")


def case(folder, normal, albedo, *, seed, contamination=False, count=8, source):
    folder.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    rows, height, width = [], *albedo.shape
    mask = np.ones((height, width), np.uint8)
    mask[:2] = mask[-2:] = 0
    mask[:, :2] = mask[:, -2:] = 0
    for index in range(count):
        angle = index * 2 * np.pi / count
        direction = [0.6 * np.cos(angle), 0.6 * np.sin(angle), -0.8]
        gain, exposure = 0.8 + 0.3 * index / (count - 1), 0.2 + 0.2 * index / (count - 1)
        signal = albedo * np.maximum(np.einsum("hwi,i->hw", normal, direction), 0)
        image = 0.02 + gain * exposure * signal
        image += rng.normal(0, 0.0001, image.shape)
        image = np.clip(image, 0, 1).astype(np.float32)
        if contamination and index == 0:
            image[height // 5 : 4 * height // 5, width // 5 : 4 * width // 5] += 0.11
        if contamination and count == 8:
            if index == 1:
                region = np.s_[height // 3 : 2 * height // 3, : width // 6]
                image[region] = 0.02 + (image[region] - 0.02) * 0.1
            image[height // 2 : height // 2 + 10, width // 8 : width // 8 + 10] = np.nan
            if index < 5:
                image[height // 8 : height // 8 + 10, 3 * width // 4 : 3 * width // 4 + 10] = 1.0
        filename = f"lamp{index}.npy"
        with (folder / filename).open("xb") as stream:
            np.save(stream, image, allow_pickle=False)
        rows.append(
            dict(
                id=f"lamp{index}",
                image=filename,
                direction=direction,
                relative_intensity=gain,
                exposure_s=exposure,
            )
        )
    for filename, value in (
        ("mask.npy", mask),
        ("normal_truth.npy", normal),
        ("albedo_truth.npy", albedo),
    ):
        with (folder / filename).open("xb") as stream:
            np.save(stream, value, allow_pickle=False)
    manifest = dict(
        schema="simple_surface_capture/v1",
        capture_id=folder.name,
        pixel_grid="rendered_reference_grid",
        response="linear",
        input_scale=1,
        frame="camera",
        black_level=0.02,
        input_ceiling=1,
        mask="mask.npy",
        lights=rows,
        source=source,
        reference_scope=(
            "Rendering inputs are reference normals/albedo; not physical gauge evidence"
        ),
    )
    write_json(folder / "capture.json", manifest)
    stack, settings, first, virtual = read_capture(folder / "capture.json")
    result = SurfaceEstimator(stack.rig, settings).estimate(stack)
    export_result(folder / "result", result, stack, first, virtual)
    valid = result.valid
    if valid.any():
        a, b = result.normals[valid].astype(np.float64), normal[valid].astype(np.float64)
        cosine = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        error = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        reliable_albedo = valid & (albedo > 1e-8)
        relative = (
            np.abs(result.albedo[reliable_albedo] - albedo[reliable_albedo])
            / albedo[reliable_albedo]
        )
    else:
        error, relative = np.empty(0), np.empty(0)
    metrics = dict(
        case=folder.name,
        source=source,
        image_kind="Rendered diffuse control, not acquired light-stack images",
        valid_pixels=int(valid.sum()),
        coverage=result.report["coverage"],
        reason_counts=result.report["reason_counts"],
        mean_normal_error_deg=float(error.mean()) if len(error) else None,
        p95_normal_error_deg=float(np.percentile(error, 95)) if len(error) else None,
        mean_relative_albedo_error_percent=float(relative.mean() * 100) if len(relative) else None,
        rejected_observations=result.report["rejected_observations"],
        settings=result.report["settings"],
        compiled_seed_count=result.report["compiled_triplet_seeds"],
    )
    write_json(folder / "evaluation.json", metrics)
    # The clear provenance caption is part of the derived overview, not source coordinates.
    image = cv2.imread(str(folder / "result/overview.png"))
    image = cv2.copyMakeBorder(image, 0, 34, 0, 0, cv2.BORDER_CONSTANT, value=(28, 28, 28))
    credit = "RENDERED CONTROL | " + source
    cv2.putText(
        image,
        credit,
        (12, image.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    assert cv2.imwrite(str(folder / "result/overview.png"), image)
    return metrics


def analytic():
    y, x = np.mgrid[-1:1:128j, -1:1:160j]
    bump = np.exp(-((x - 0.2) ** 2 + (y + 0.15) ** 2) / 0.08)
    hollow = np.exp(-((x + 0.45) ** 2 + (y - 0.2) ** 2) / 0.04)
    dx = -0.1 * 2 * (x - 0.2) / 0.08 * bump + 0.055 * 2 * (x + 0.45) / 0.04 * hollow
    dy = -0.1 * 2 * (y + 0.15) / 0.08 * bump + 0.055 * 2 * (y - 0.2) / 0.04 * hollow
    normal = np.stack((dx, dy, -np.ones_like(x)), axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    albedo = 0.42 + 0.16 * np.sin(10 * x) * np.cos(6 * y)
    return normal.astype(np.float32), albedo.astype(np.float32)


def build(output, polyhaven):
    output.mkdir(parents=True, exist_ok=False)
    metrics = []
    normal, albedo = analytic()
    for name, bad in (("analytic_clean", False), ("analytic_contaminated", True)):
        metrics.append(
            case(
                output / name,
                normal,
                albedo,
                seed=7601,
                contamination=bad,
                source="Original analytic bump/hollow and varying albedo",
            )
        )
    flat = np.zeros_like(normal)
    flat[:, :, 2] = -1
    metrics.append(
        case(
            output / "four_light_ambiguity",
            flat,
            np.full_like(albedo, 0.5),
            seed=7601,
            contamination=True,
            count=4,
            source="Original flat reference; four lights with one competing observation",
        )
    )
    originals = output / "polyhaven_source"
    originals.mkdir()
    provenance = json.loads((polyhaven / "provenance.json").read_bytes())
    for name, entry in provenance["files"].items():
        raw = (polyhaven / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("Material source bytes changed")
        shutil.copyfile(polyhaven / name, originals / name)
    shutil.copyfile(polyhaven / "provenance.json", originals / "provenance.json")
    # Curation-only EXR decoder; runtime capture manifests use PNG/NPY planes.
    import OpenEXR

    with OpenEXR.File(str(originals / "diffuse.exr")) as exr:
        diffuse = exr.channels()["RGB"].pixels.astype(np.float32)
    encoded_normal = cv2.imread(str(originals / "normal_gl.png"), cv2.IMREAD_UNCHANGED)
    if (
        diffuse is None
        or encoded_normal is None
        or diffuse.shape != encoded_normal.shape
        or diffuse.dtype != np.float32
        or encoded_normal.dtype != np.uint16
    ):
        raise ValueError("Expected pinned linear EXR and uint16 RGB normal texture")
    diffuse = cv2.resize(diffuse, (256, 256), interpolation=cv2.INTER_AREA)
    normal = 2 * encoded_normal[:, :, ::-1].astype(np.float64) / 65535 - 1
    normal = cv2.resize(normal, (256, 256), interpolation=cv2.INTER_AREA)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    normal[:, :, 1:] *= -1  # OpenGL tangent axes -> camera X right, Y down, Z forward.
    albedo = diffuse[:, :, 1]
    if not np.isfinite(albedo).all() or albedo.min() < 0 or albedo.max() > 1:
        raise ValueError("Pinned linear diffuse green channel exceeds normalized reflectance range")
    write_json(
        originals / "transform.json",
        dict(
            size=[256, 256],
            resampling="Area average; decoded normal vectors renormalized",
            normal_decode="OpenGL 2*uint16/65535-1, then [Nx,-Ny,-Nz] into camera axes",
            albedo="Linear EXR green channel; no per-frame normalization",
            rendering=(
                "Local diffuse cosine only; not measured illumination or a full optical simulation"
            ),
        ),
    )
    for name, bad in (("polyhaven_clean", False), ("polyhaven_contaminated", True)):
        metrics.append(
            case(
                output / name,
                normal.astype(np.float32),
                albedo.astype(np.float32),
                seed=7601,
                contamination=bad,
                source="Powered by Poly Haven / Charlotte Baglioni / Anti Slip Concrete / CC0",
            )
        )
    write_json(
        output / "summary.json",
        dict(
            scope=(
                "Two rendered control families; acquired real multi-light validation remains open"
            ),
            cases=metrics,
        ),
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--polyhaven", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.polyhaven)))
