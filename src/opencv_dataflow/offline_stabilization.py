"""Centered camera-trajectory smoothing for a finite batch of recorded frames."""

from uuid import uuid4

import numpy as np

from .core import Block, Port, Result
from .images import ImageArray, image_format
from .parameter_validation import NumericRange, ValidationError
from .payloads import (
    NO_VALUE,
    FlowGroup,
    ValuePacket,
    check_spatial,
    frame,
    image_with_payload,
    metadata,
)
from .processing import gray_image
from .selection import selection_mask
from .stabilization import (
    StabilizeCameraMotion,
    _estimate_pose,
    _features,
    _pose_matrix,
    _transform_payload,
    _warp_correction,
)

_OUTPUTS = ("corrected", "coverage", "transforms")
_MAX_BATCH_BYTES = 128 * 1024**2
_SHARED_PARAMETERS = (
    "motion_model",
    "max_features",
    "feature_quality",
    "feature_spacing",
    "tracking_window",
    "max_pyramid_level",
    "forward_backward_pixels",
    "max_patch_error",
    "inlier_pixels",
    "min_inliers",
    "min_inlier_ratio",
    "max_step_pixels",
    "max_step_degrees",
    "max_step_scale_ratio",
    "min_coverage_fraction",
    "interpolation",
)


def _frames(group):
    if not isinstance(group, FlowGroup) or not 2 <= len(group.items) <= 64:
        raise ValidationError("Frames requires an ordered group of two to 64 images")
    if any(
        not isinstance(entry.value, np.ndarray) or entry.value.ndim not in (2, 3)
        for entry in group.items
    ):
        raise ValidationError("Every stabilization entry must be an image")
    first = group.items[0].value
    height, width = first.shape[:2]
    if min(height, width) < 8 or height * width > 16_777_216:
        raise ValidationError("Use at least 8 × 8 pixels and at most 16,777,216 pixels")
    bytes_used = sum(entry.value.nbytes + height * width for entry in group.items)
    if bytes_used > _MAX_BATCH_BYTES:
        raise ValidationError("Corrected batch and coverage exceed the 128 MiB array limit")
    contexts = [metadata(entry) for entry in group.items]
    images = [
        image_with_payload(entry.value, context)
        for entry, context in zip(group.items, contexts, strict=True)
    ]
    encoding = image_format(images[0])
    for image, context in zip(images, contexts, strict=True):
        if (
            image.shape != first.shape
            or image.dtype != first.dtype
            or image_format(image) != encoding
        ):
            raise ValidationError("All frames must match in size, pixel depth and encoding")
        check_spatial(context, [width, height])
    if encoding not in ("gray", "bgr", "rgb", "bgra", "rgba"):
        raise ValidationError("Stabilization requires grayscale or RGB/BGR encoding")
    return images, contexts, encoding


class StabilizeRecordedSequence(Block):
    type_id, title = "sequence.stabilize_batch", "Stabilize recorded sequence"
    category, block_type, subtype = "Sequence", "Sequence", "Temporal processing"
    process_inputs_readonly = True
    payload_reference = "frames"
    inputs = (
        Port("frames", "any"),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (*tuple(Port(name, "any") for name in _OUTPUTS), Port("report", "data"))
    flow_primary = _OUTPUTS
    defaults = {
        "radius_frames": 4,
        "sigma_frames": 2.0,
        **{name: StabilizeCameraMotion.defaults[name] for name in _SHARED_PARAMETERS},
    }
    parameter_choices = {
        name: choices
        for name, choices in StabilizeCameraMotion.parameter_choices.items()
        if name in _SHARED_PARAMETERS
    }
    parameter_ranges = {
        "radius_frames": NumericRange(1, 63, whole=True),
        "sigma_frames": NumericRange(0.1, 63),
        **{
            name: limits
            for name, limits in StabilizeCameraMotion.parameter_ranges.items()
            if name in _SHARED_PARAMETERS
        },
    }
    parameter_help = {
        **{
            name: description
            for name, description in StabilizeCameraMotion.parameter_help.items()
            if name in _SHARED_PARAMETERS
        },
        "radius_frames": "Maximum number of neighbors on each side of the current frame.",
        "sigma_frames": "Standard deviation of the centered Gaussian weights, in frames.",
        "min_coverage_fraction": "Withhold the batch if any frame falls below this coverage.",
    }
    input_help = {
        "frames": "Ordered batch of two to 64 same-grid frames from one recording.",
        "mask": "Optional static-background mask applied in each frame's raw grid.",
        "region": "Optional static-background region applied in each frame's raw grid.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if int(p["min_inliers"]) > int(p["max_features"]):
            raise ValidationError(
                "Minimum inliers exceeds maximum features", "min_inliers", "max_features"
            )

    def _estimate_path(self, images, inputs, center, report):
        p = self.parameters
        path = np.zeros((len(images), 4))
        starts = report["segment_starts"]
        previous, points = None, None
        for index, image in enumerate(images):
            if self._cancelled():
                return None
            gray = gray_image(image, (np.uint8,), auto_depth=True)
            mask = selection_mask({**inputs, "image": image}, image)
            details = {"index": index, "tracked": 0, "inliers": 0, "inlier_ratio": 0.0}
            reason = "initial_frame"
            if previous is not None:
                observed, _, reason = _estimate_pose(
                    previous, gray, points, mask, center, path[index - 1], p, details
                )
                if reason is None:
                    path[index] = observed
            if reason is not None:
                starts.append(index)
            details.update(segment=len(starts) - 1, reason=reason or "tracked")
            report["frames"].append(details)
            previous, points = gray, _features(gray, mask, p, "max_features")
        return path

    @staticmethod
    def _reject(report, reason):
        report["status"] = reason
        return Result(
            {**dict.fromkeys(_OUTPUTS, NO_VALUE), "report": report},
            passed=False,
            fail_reasons=[f"offline_stabilization_{reason}"],
            message=reason.replace("_", " "),
        )

    def process(self, inputs):
        group = inputs["frames"]
        images, contexts, encoding = _frames(group)
        size = images[0].shape[1::-1]
        center = (np.asarray(size) - 1) / 2
        p = self.parameters
        report = {
            "model": "offline_stabilization_v1",
            "frame_count": len(images),
            "segment_starts": [],
            "frames": [],
        }
        path = self._estimate_path(images, inputs, center, report)
        if path is None:
            return self._reject(report, "cancelled")

        radius = int(p["radius_frames"])
        weights = np.exp(-0.5 * (np.arange(-radius, radius + 1) / float(p["sigma_frames"])) ** 2)
        corrected, coverage, transforms = [], [], []
        starts = report["segment_starts"]
        root = f"{self.id}:offline:{uuid4().hex}"
        for segment, (start, end) in enumerate(
            zip(starts, [*starts[1:], len(images)], strict=True)
        ):
            for index in range(start, end):
                if self._cancelled():
                    return self._reject(report, "cancelled")
                lo, hi = max(start, index - radius), min(end, index + radius + 1)
                target = np.average(
                    path[lo:hi], axis=0, weights=weights[lo - index + radius : hi - index + radius]
                )
                target_pose = _pose_matrix(target, center)
                correction = target_pose @ np.linalg.inv(_pose_matrix(path[index], center))
                output, valid, covered = _warp_correction(images[index], correction, p)
                report["frames"][index].update(
                    window=[lo, hi],
                    observed_pose=path[index].tolist(),
                    smoothed_pose=target.tolist(),
                    coverage_fraction=covered,
                )
                if output is None:
                    report["failed_frame"] = index
                    return self._reject(report, "insufficient_coverage")
                spatial = frame(size, f"{root}:{segment}")
                spatial["to_root"] = np.linalg.inv(target_pose).tolist()
                context = {**contexts[index], "spatial": spatial}
                context.pop("pyramid", None)
                transform = _transform_payload(correction, size, contexts[index], spatial)
                corrected.append(ValuePacket(ImageArray(output, encoding), context))
                coverage.append(ValuePacket(valid, context))
                transforms.append(ValuePacket(transform, context))

        report["status"] = "complete"
        return Result(
            {
                "corrected": FlowGroup(tuple(corrected), group.labels),
                "coverage": FlowGroup(tuple(coverage), group.labels),
                "transforms": FlowGroup(tuple(transforms), group.labels),
                "report": report,
            },
            message=f"Stabilized {len(images)} frames in {len(starts)} segments",
        )
