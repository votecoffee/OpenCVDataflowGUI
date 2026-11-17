"""Audit recorded depth/XYZ conversions against per-image BOP camera declarations."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from opencv_dataflow.cloud_io import read_cloud


def audit(root, output):
    output.mkdir(parents=True, exist_ok=False)
    cases = [
        (3, root / "tless/frame000003", root / "tless/source"),
        (13, root / "tless/frame000013", root / "tless/source"),
        (255, root / "locator_chain/reconstructed", root / "locator_chain/source"),
    ]
    rows = []
    for number, case, source in cases:
        folder = source / "test_primesense/000002"
        metadata = json.loads((folder / "scene_camera.json").read_bytes())[str(number)]
        raw_path = folder / f"depth/{number:06}.png"
        raw = cv2.imread(str(raw_path), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.uint16:
            raise ValueError("Retain the original uint16 BOP depth capture")
        depth_path = case / "assets/depth.npy"
        depth = np.load(depth_path, allow_pickle=False)
        records = json.loads((case / "assets/records.json").read_bytes())
        declaration = records["camera"]
        actual_k = np.array(declaration.get("matrix", declaration.get("camera_matrix")))
        expected_k = np.array(metadata["cam_K"]).reshape(3, 3)
        if not np.array_equal(actual_k, expected_k):
            raise ValueError("Retained intrinsics differ from this image's camera declaration")
        metric = raw.astype(np.float64) * metadata["depth_scale"] * 0.001
        if depth.dtype != np.float32 or not np.array_equal(depth, metric.astype(np.float32)):
            raise ValueError("Stored depth differs from the declared per-image depth conversion")
        if not np.array_equal(depth == 0, raw == 0):
            raise ValueError("Stored depth filled or removed unknown source samples")
        row = dict(
            frame=number,
            source_size=[raw.shape[1], raw.shape[0]],
            depth_scale_mm_per_sample=metadata["depth_scale"],
            per_image_intrinsics_exact=True,
            missing_pixels_unchanged=True,
            stored_depth_exact_declared_float32_conversion=True,
            maximum_float32_depth_roundoff_m=float(np.max(abs(depth.astype(float) - metric))),
            raw_depth_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            stored_depth_sha256=hashlib.sha256(depth_path.read_bytes()).hexdigest(),
        )
        cloud_path = case / "assets/full.ply"
        if cloud_path.exists():
            cloud = read_cloud(cloud_path)
            if cloud.frame != declaration["frame"] or cloud.units not in ("m", "mm"):
                raise ValueError("Measured cloud must use the declared metric camera frame")
            points = cloud.points.astype(np.float64) * (1000 if cloud.units == "m" else 1)
            projected = points @ expected_k.T
            pixels_float = projected[:, :2] / projected[:, 2:3]
            pixels = np.rint(pixels_float).astype(int)
            if (
                not len(points)
                or np.any(points[:, 2] <= 0)
                or np.any(pixels < 0)
                or np.any(pixels >= [raw.shape[1], raw.shape[0]])
            ):
                raise ValueError("Measured cloud does not map to valid source depth pixels")
            u, v = pixels.T
            z = raw[v, u].astype(float) * metadata["depth_scale"]
            expected = np.c_[
                (u - expected_k[0, 2]) * z / expected_k[0, 0],
                (v - expected_k[1, 2]) * z / expected_k[1, 1],
                z,
            ]
            errors = abs(points - expected)
            # Representation roundoff only, independent of product pose/contact tolerances.
            roundoff = 4 * np.finfo(np.float32).eps * np.maximum(1, abs(expected))
            if np.any(z == 0) or np.any(errors > roundoff):
                raise ValueError("Measured XYZ differs from source rays beyond float32 roundoff")
            row["full_cloud"] = dict(
                measured_points=len(points),
                valid_source_rays=len(np.unique(v * raw.shape[1] + u)),
                maximum_xyz_roundoff_mm=errors.max(0).tolist(),
                maximum_reprojection_roundoff_pixels=float(abs(pixels_float - pixels).max()),
                source_ray_backprojection_within_float32_roundoff=True,
                sha256=hashlib.sha256(cloud_path.read_bytes()).hexdigest(),
            )
        rows.append(row)
    report = dict(
        model="recorded_depth_conversion_audit_v1",
        cases=rows,
        format_source="https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md",
        scope="Declared-input conversion only; not physical calibration or CAD/pose accuracy",
        conclusion="Conversions match declarations; CAD/depth disagreement remains unresolved",
        no_acceptance_or_clearance_threshold_changed=True,
    )
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit(args.root, args.output)
