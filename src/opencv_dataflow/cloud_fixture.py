"""Reproducible point-cloud fixtures with paired ground truth."""

import cv2
import numpy as np

from .core import Port, Result
from .geometry3d_common import Tool3D, cloud_graphics, pose_matrix, pose_record, transformed_cloud
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE

_MAX_ROWS = 4096
_SEED = NumericRange(0, 2**32 - 1, whole=True)


def _vector(value, field, low, high):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValidationError("Use three numeric components", field)
    rule = NumericRange(low, high)
    return np.array([rule.check(component, field, field) for component in value], dtype=np.float64)


class PerturbCloudFixture(Tool3D):
    type_id = "3d.cloud_perturb"
    title, subtype = "Perturb point-cloud fixture", "Point clouds"
    description = "Make a seeded scan with an exact pose, a clean counterpart and row matches."
    payload_reference = "cloud"
    preview_safe = parallel_safe = process_inputs_readonly = True
    inputs = (
        Port("cloud", "cloud"),
        Port("seed", "number", required=False, required_when_connected=True),
    )
    outputs = (
        Port("cloud", "cloud", required=False),
        Port("clean", "cloud", required=False),
        Port("pose", "data", required=False, visual_type="pose"),
        Port("pairs", "data", required=False),
        Port("report", "data"),
    )
    defaults = {
        "seed": 0,
        "pose_mode": "fixed",
        "rotation_vector_deg": [0.0, 0.0, 0.0],
        "max_rotation_deg": 20.0,
        "translation": [0.0, 0.0, 0.0],
        "target_frame": "fixture_target",
        "noise_distribution": "gaussian",
        "noise_scale": [0.0, 0.0, 0.0],
        "dropout_fraction": 0.0,
    }
    parameter_choices = {
        "pose_mode": ("fixed", "seeded"),
        "noise_distribution": ("gaussian", "uniform"),
    }
    parameter_ranges = {
        "seed": _SEED,
        "max_rotation_deg": NumericRange(0, 180),
        "dropout_fraction": NumericRange(0, 0.99),
    }
    parameter_help = {
        "seed": "Private random seed. The connected seed takes precedence.",
        "pose_mode": "Use an explicit pose or draw one within the configured bounds.",
        "rotation_vector_deg": "Fixed axis-angle vector. Its magnitude sets the angle in degrees.",
        "max_rotation_deg": "Seeded pose draws a uniform angle from zero to this limit.",
        "translation": "Fixed XYZ offset or seeded +/- bounds in cloud units.",
        "target_frame": "Named coordinate frame of the two generated clouds.",
        "noise_distribution": "Independent Gaussian or uniform draws in the target XYZ axes.",
        "noise_scale": "Gaussian standard deviation or uniform half-width for each target axis.",
        "dropout_fraction": "Fraction of source rows to remove at random, rounded down.",
    }
    input_help = {"cloud": "Supply 1 to 4,096 source rows. Sample larger clouds before this block."}
    output_help = {
        "cloud": "Transformed retained points with the configured XYZ noise.",
        "clean": "Noise-free transformed points in the same order as the perturbed cloud.",
        "pose": "Exact rigid transform from the original cloud frame to the fixture frame.",
        "pairs": "Zero-based [original source row, generated row] pairs for both output clouds.",
        "report": "Seed, pose and noise settings, retained counts and realized displacement.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        target = p["target_frame"]
        if not isinstance(target, str) or not target.strip() or len(target) > 128:
            raise ValidationError(
                "Use a named target frame of up to 128 characters", "target_frame"
            )
        rotation = _vector(p["rotation_vector_deg"], "rotation_vector_deg", -180, 180)
        if np.linalg.norm(rotation) > 180:
            raise ValidationError(
                "Rotation vector magnitude must be at most 180 degrees", "rotation_vector_deg"
            )
        _vector(p["translation"], "translation", 0 if p["pose_mode"] == "seeded" else -1e6, 1e6)
        _vector(p["noise_scale"], "noise_scale", 0, 1e6)

    def process(self, inputs):
        source, p = inputs["cloud"], self.parameters
        size = len(source.points)
        if not 1 <= size <= _MAX_ROWS:
            raise ValidationError(
                "Supply 1 to 4,096 fixture rows; sample larger clouds first", "cloud"
            )
        seed = int(_SEED.check(inputs.get("seed", p["seed"]), "Seed", "seed"))
        if self._cancelled():
            return self._cancelled_fixture(seed)

        pose_rng, noise_rng, row_rng = (
            np.random.Generator(np.random.PCG64(stream))
            for stream in np.random.SeedSequence(seed).spawn(3)
        )
        translation = np.asarray(p["translation"], np.float64)
        rotation_vector = np.deg2rad(p["rotation_vector_deg"])
        if p["pose_mode"] == "seeded":
            axis = pose_rng.standard_normal(3)
            rotation_vector = (
                axis / np.linalg.norm(axis) * pose_rng.uniform(0, np.deg2rad(p["max_rotation_deg"]))
            )
            translation = pose_rng.uniform(-translation, translation)

        rotation, _ = cv2.Rodrigues(rotation_vector)
        transform = pose_matrix(rotation, translation)
        pose = pose_record(transform, source.frame, p["target_frame"], source.units)

        dropped = int(np.floor(size * p["dropout_fraction"]))
        kept = size - dropped
        rows = np.sort(row_rng.choice(size, kept, replace=False)) if dropped else np.arange(size)
        source_rows = source.subset(rows)
        clean = transformed_cloud(source_rows, transform, p["target_frame"])
        noisy = clean.subset(slice(None))
        noise_scale = np.asarray(p["noise_scale"], np.float64)
        if noise_scale.any():
            # Draw for every original row so changing dropout leaves each surviving draw intact.
            noise = (
                noise_rng.standard_normal((size, 3))
                if p["noise_distribution"] == "gaussian"
                else noise_rng.uniform(-1.0, 1.0, (size, 3))
            )
            perturbed = clean.points.astype(np.float64) + noise[rows] * noise_scale
            if not np.isfinite(perturbed).all() or np.any(
                np.abs(perturbed) > np.finfo(np.float32).max
            ):
                raise ValidationError("Perturbed coordinates exceed the float32 cloud range")
            noisy.points = perturbed.astype(np.float32)

        for cloud, role in ((clean, "clean"), (noisy, "perturbed")):
            cloud.metadata["synthetic_fixture"] = {
                "role": role,
                "seed": seed,
                "original_frame": source.frame,
            }

        delta = noisy.points.astype(np.float64) - clean.points
        displacements = np.linalg.norm(delta, axis=1)
        report = {
            "model": "perturbed_cloud_fixture_v1",
            "status": "ok",
            "seed": seed,
            "rng": "PCG64",
            "pose_mode": p["pose_mode"],
            "source_frame": source.frame,
            "target_frame": clean.frame,
            "units": clean.units,
            "input_points": size,
            "retained_points": kept,
            "dropped_points": dropped,
            "noise_distribution": p["noise_distribution"],
            "noise_scale": noise_scale.tolist(),
            "noise_mean_xyz": delta.mean(axis=0).tolist(),
            "noise_stddev_xyz": delta.std(axis=0).tolist(),
            "displacement_rms": float(np.sqrt(np.mean(displacements**2))),
            "displacement_max": float(displacements.max()),
        }
        if self._cancelled():
            return self._cancelled_fixture(seed)
        return Result(
            {
                "cloud": noisy,
                "clean": clean,
                "pose": pose,
                "pairs": np.column_stack((rows, np.arange(kept))).tolist(),
                "report": report,
            },
            graphics=cloud_graphics(self, noisy),
            message=(
                f"{kept:,} of {size:,} rows, "
                f"noise RMS {report['displacement_rms']:.4g} {source.units}"
            ),
        )

    def _cancelled_fixture(self, seed):
        return Result(
            {
                "cloud": NO_VALUE,
                "clean": NO_VALUE,
                "pose": NO_VALUE,
                "pairs": NO_VALUE,
                "report": {
                    "model": "perturbed_cloud_fixture_v1",
                    "status": "cancelled",
                    "seed": seed,
                },
            },
            passed=False,
            fail_reasons=["cloud_fixture_cancelled"],
            message="Fixture generation cancelled",
        )
