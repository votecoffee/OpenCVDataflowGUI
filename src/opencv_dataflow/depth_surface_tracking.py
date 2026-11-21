"""Track and reconstruct visible deforming surfaces on a fixed RGB-D grid."""

import cv2
import numpy as np

from .calibration_revisions import APPLIED, CAPTURE, digest, is_full_grid
from .core import Port
from .geometry3d_common import camera_for_image, require_pinhole
from .images import DepthArray, ImageArray, preview_rgb
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, frame
from .pointclouds import PointCloud
from .processing import gray_image
from .sequence import (
    SEQUENCE_CHOICES,
    SEQUENCE_DEFAULTS,
    SEQUENCE_HELP,
    SequenceBlock,
    SequenceStep,
)
from .stabilization import StabilizeCameraMotion, _correspondences
from .stereo3d import metric_depth
from .tickets import ticket_key

_MAX_IMAGE_PIXELS = 4_194_304
_MAX_TRACKS = 65_536
_MAX_LK_WORK = 134_217_728
_LK_FIELDS = ("tracking_window", "max_pyramid_level", "forward_backward_pixels", "max_patch_error")
_LK = StabilizeCameraMotion


def _camera_xyz(pixels, z, intrinsics):
    return np.column_stack(
        (
            (pixels[:, 0] - intrinsics[0, 2]) * z / intrinsics[0, 0],
            (pixels[:, 1] - intrinsics[1, 2]) * z / intrinsics[1, 1],
            z,
        )
    )


def _surface_samples(z, valid, pixels, max_span):
    """Bilinear depth with complete support and a span limit across nonzero taps."""
    x, y = pixels.T
    x0, y0 = np.floor(pixels).astype(np.int32).T
    x1, y1 = np.minimum(x0 + 1, z.shape[1] - 1), np.minimum(y0 + 1, z.shape[0] - 1)
    xs, ys = np.stack((x0, x1, x0, x1)), np.stack((y0, y0, y1, y1))
    u, v = x - x0, y - y0
    weights = np.stack(((1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v))
    used = weights > 0
    depths = z[ys, xs]
    lowest = np.where(used, depths, np.inf).min(axis=0)
    highest = np.where(used, depths, -np.inf).max(axis=0)
    usable = np.all(~used | valid[ys, xs], axis=0) & (highest - lowest <= max_span)
    values = (np.where(used, depths, 0) * weights).sum(axis=0)
    return usable, values[usable]


def _aligned_guide(guide, depth_payload, guide_payload, size, optical_frame):
    if guide.shape[:2][::-1] != size:
        raise ValidationError("Guide and depth must share the calibrated pixel dimensions")
    if guide_payload.get("optical_frame", optical_frame) != optical_frame:
        raise ValidationError("Guide must be registered to the depth optical frame")
    if "spatial" in guide_payload and not is_full_grid(guide_payload, size):
        raise ValidationError("Guide must use its full registered pixel grid")
    a, b = (payload.get("ticket") for payload in (depth_payload, guide_payload))
    if a is not None and b is not None and ticket_key(a) != ticket_key(b):
        raise ValidationError("Guide and depth must come from the same ticket")
    a, b = (payload.get("recorded_rgbd") for payload in (depth_payload, guide_payload))
    if a is not None or b is not None:
        if not all(isinstance(stamp, dict) for stamp in (a, b)) or any(
            a.get(key) != b.get(key) or a.get(key) is None for key in ("clock", "pair_index")
        ):
            raise ValidationError("Guide and depth must come from the same recorded RGB-D pair")
    return gray_image(guide, (np.uint8,))


class TrackDeformingDepthSurface(SequenceBlock):
    type_id, title = "3d.track_deforming_surface", "Track deforming depth surface"
    category = block_type = "3D vision"
    subtype = "Depth and RGB-D"
    inputs = (
        Port("image", "depth"),
        Port("camera", "data"),
        Port("guide", required=False, required_when_connected=True),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, visual_type="region", required_when_connected=True),
        Port("reset", "data", required=False),
        Port("stream", "data", required=False),
    )
    data_outputs = (
        Port("cloud", "cloud", required=False),
        Port("surface_depth", "depth", required=False),
        Port("displacement", required=False),
    )
    defaults = {
        **SEQUENCE_DEFAULTS,
        **{key: _LK.defaults[key] for key in _LK_FIELDS},
        "correspondence": "tracked_guide",
        "input_units": "mm",
        "units_per_sample": 1.0,
        "sample_step": 4,
        "minimum_points": 20,
        "minimum_fraction": 0.55,
        "minimum_depth_m": 0.05,
        "maximum_depth_m": 5.0,
        "max_depth_span_m": 0.03,
        "max_motion_px": 36.0,
        "max_displacement_m": 0.25,
    }
    parameter_choices = {
        **SEQUENCE_CHOICES,
        "correspondence": ("tracked_guide", "fixed_pixel"),
        "input_units": ("m", "mm"),
    }
    parameter_ranges = {
        **{key: _LK.parameter_ranges[key] for key in _LK_FIELDS},
        "units_per_sample": NumericRange(1e-9, 1000),
        "sample_step": NumericRange(1, 128, whole=True),
        "minimum_points": NumericRange(1, _MAX_TRACKS, whole=True),
        "minimum_fraction": NumericRange(0.001, 1),
        "minimum_depth_m": NumericRange(0.000001, 1000),
        "maximum_depth_m": NumericRange(0.000001, 1000),
        "max_depth_span_m": NumericRange(0, 100),
        "max_motion_px": NumericRange(0, 2048),
        "max_displacement_m": NumericRange(0.000001, 1000),
    }
    parameter_help = {
        **SEQUENCE_HELP,
        "tracking_window": "Odd side length of each Lucas–Kanade window in pixels.",
        "max_pyramid_level": "Highest optical-flow pyramid level.",
        "forward_backward_pixels": _LK.parameter_help["forward_backward_pixels"],
        "max_patch_error": _LK.parameter_help["max_patch_error"],
        "correspondence": "Track a registered guide, or compare camera rays at fixed pixels.",
        "input_units": "Units for an untagged input. A depth tag supplies its own units.",
        "units_per_sample": "Depth units per integer sample. Float depth uses those units.",
        "sample_step": "Reference-grid spacing in pixels. Increase it for large depth images.",
        "minimum_points": "Minimum valid points needed to capture or track the surface.",
        "minimum_fraction": "Minimum fraction of reference samples needed to emit a surface.",
        "minimum_depth_m": "Nearest usable depth in meters.",
        "maximum_depth_m": "Farthest usable depth in meters.",
        "max_depth_span_m": "Reject interpolation across larger depth jumps between active taps.",
        "max_motion_px": "Maximum image displacement from the reference pixel in tracked mode.",
        "max_displacement_m": "Maximum 3D displacement from a reference sample.",
    }
    input_help = {
        "image": "Full camera-Z depth frame on the calibrated pinhole grid.",
        "camera": "Fixed, undistorted pinhole intrinsics on the input depth grid.",
        "guide": "Same exposure's 8-bit image already registered to depth.",
        "mask": "Valid area in the live depth image. Also limits the reference on capture.",
        "region": "Optional selected area in the live depth image.",
        "reset": "True clears this stream's reference before processing the incoming depth.",
    }
    payload_reference = "image"
    process_inputs_readonly = True
    preview_safe = False

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["minimum_depth_m"] >= p["maximum_depth_m"]:
            raise ValidationError(
                "Set increasing depth limits", "minimum_depth_m", "maximum_depth_m"
            )

    def validate_sequence_image(self, image):
        if (
            not isinstance(image, np.ndarray)
            or image.ndim != 2
            or image.dtype not in (np.uint16, np.float32, np.float64)
            or not 0 < image.size <= _MAX_IMAGE_PIXELS
        ):
            raise ValidationError("Use a scalar depth plane with up to 4,194,304 pixels")

    def extra_context(self, inputs):
        image, camera = inputs["image"], inputs["camera"]
        capture = self.input_payloads["image"].get("recorded_rgbd")
        return {
            "camera": digest(camera),
            "recording_clock": capture.get("clock") if isinstance(capture, dict) else None,
            "depth_units": getattr(image, "units", self.parameters["input_units"]),
        }

    def _result(self, state, status, observed, **counts):
        return {
            "surface_state": status,
            "correspondence": self.parameters["correspondence"],
            "reference_points": len(state["reference_xy"]) if "reference_xy" in state else 0,
            "accepted_points": observed,
            "reference_ticket": state.get("reference_ticket"),
            "reference_recording": state.get("reference_recording"),
            "current_recording": self.input_payloads["image"].get("recorded_rgbd"),
            **counts,
        }

    def _waiting(self, state, status, observed=0, learned=False, **counts):
        return SequenceStep(
            dict.fromkeys(self.flow_primary),
            ready=False,
            learned=learned,
            details=self._result(state, status, observed, **counts),
        )

    def _anchor(self, state, z, valid, gray, intrinsics):
        spacing = int(self.parameters["sample_step"])
        height, width = z.shape
        if ((height + spacing - 1) // spacing) * ((width + spacing - 1) // spacing) > _MAX_TRACKS:
            raise ValidationError("Reference grid exceeds 65,536 samples. Increase Sample Step.")
        y, x = np.mgrid[:height:spacing, :width:spacing]
        keep = valid[y, x]
        x, y = x[keep], y[keep]
        if len(x) < int(self.parameters["minimum_points"]):
            return self._waiting(state, "insufficient_reference", len(x))
        pixels = np.column_stack((x, y)).astype(np.float32)
        payload = self.input_payloads["image"]
        state.update(
            reference_xy=pixels,
            reference_xyz=_camera_xyz(pixels, z[y, x], intrinsics),
            reference_indices=(y * width + x).astype(np.int32),
            reference_guide=gray.copy() if gray is not None else None,
            reference_ticket=payload.get("ticket"),
            reference_recording=payload.get("recorded_rgbd"),
            material_grid=frame((width, height), f"{self.id}:material:{state['generation']}"),
        )
        return self._waiting(state, "anchored", len(x), learned=True)

    def step(self, inputs, state, mask):
        depth, camera = inputs["image"], inputs["camera"]
        intrinsics, distortion, size = require_pinhole(camera)
        camera_for_image(camera, depth)
        if np.any(distortion):
            raise ValidationError("Use undistorted, registered input for fixed-view reconstruction")
        optical_frame = camera.get("frame", "camera")
        payload = self.input_payloads["image"]
        if payload.get("optical_frame", optical_frame) != optical_frame:
            raise ValidationError("Depth frame differs from the camera optical frame")
        if "spatial" in payload and not is_full_grid(payload, size):
            raise ValidationError("Depth must use its full calibrated pixel grid")
        p = self.parameters
        z = metric_depth(depth, float(p["units_per_sample"]), p["input_units"])
        valid = (z >= float(p["minimum_depth_m"])) & (z <= float(p["maximum_depth_m"]))
        if mask is not None:
            valid &= mask != 0
        gray = None
        guide = inputs.get("guide")
        if p["correspondence"] == "tracked_guide":
            if guide is None:
                raise ValidationError("Connect a registered guide or choose Fixed Pixel")
            gray = _aligned_guide(guide, payload, self.input_payloads["guide"], size, optical_frame)
        if "reference_xy" not in state:
            return self._anchor(state, z, valid, gray, intrinsics)

        reference = state["reference_xy"]
        if not valid.any():
            return self._waiting(
                state,
                "tracking_lost",
                image_tracks=0,
                depth_supported_points=0,
                tracked_fraction=0.0,
            )
        if gray is None:
            ids, pixels = np.arange(len(reference)), reference
        else:
            window, levels = int(p["tracking_window"]), int(p["max_pyramid_level"]) + 1
            if len(reference) * window**2 * levels > _MAX_LK_WORK:
                raise ValidationError("Tracking work exceeds the limit. Increase Sample Step.")
            ids, pixels = _correspondences(
                state["reference_guide"], gray, reference[:, None], mask, p
            )
            within = np.linalg.norm(pixels - reference[ids], axis=1) <= float(p["max_motion_px"])
            ids, pixels = ids[within], pixels[within]
        image_tracks = len(ids)
        usable, sampled = _surface_samples(z, valid, pixels, float(p["max_depth_span_m"]))
        xyz = _camera_xyz(pixels[usable], sampled, intrinsics)
        ids, pixels = ids[usable], pixels[usable]
        motion = np.linalg.norm(xyz - state["reference_xyz"][ids], axis=1)
        safe = motion <= float(p["max_displacement_m"])
        fraction = float(np.count_nonzero(safe)) / len(reference)
        counts = {
            "image_tracks": image_tracks,
            "depth_supported_points": len(ids),
            "tracked_fraction": fraction,
        }
        kept = int(np.count_nonzero(safe))
        if kept < int(p["minimum_points"]) or fraction < float(p["minimum_fraction"]):
            return self._waiting(state, "tracking_lost", kept, **counts)

        indices = state["reference_indices"][ids[safe]]
        observed = np.rint(pixels[safe]).astype(np.int32)
        colors = preview_rgb(guide)[observed[:, 1], observed[:, 0]] if gray is not None else None
        cloud = PointCloud(
            xyz[safe],
            "m",
            optical_frame,
            colors=colors,
            pixels=observed,
            source_size=size,
            attributes={"source_index": indices},
        )
        material_depth = np.full(z.shape, np.nan, np.float32)
        material_depth.flat[indices] = sampled[safe]
        displacement = np.full(z.shape, np.nan, np.float32)
        displacement.flat[indices] = motion[safe]
        graphics = {}
        if self.generate_graphics:
            visual = np.zeros(z.shape, np.uint8)
            visual.flat[indices] = np.rint(motion[safe] * 255 / float(p["max_displacement_m"]))
            heatmap = cv2.applyColorMap(visual, cv2.COLORMAP_TURBO)
            heatmap[~np.isfinite(displacement)] = 0
            graphics["Reference-grid displacement"] = ImageArray(heatmap, "bgr")
        return SequenceStep(
            {
                "cloud": cloud,
                "surface_depth": DepthArray(
                    material_depth, "m", float(p["minimum_depth_m"]), float(p["maximum_depth_m"])
                ),
                "displacement": ImageArray(displacement, "gray"),
            },
            details=self._result(
                state,
                "tracked",
                kept,
                **counts,
                rms_displacement_m=float(np.sqrt(np.mean(motion[safe] ** 2))),
                maximum_displacement_m=float(np.max(motion[safe])),
            ),
            output_frames={
                "surface_depth": state["material_grid"],
                "displacement": state["material_grid"],
            },
            graphics=graphics,
        )

    def finalize_result(self, result, inputs):
        status = result.outputs["status"]
        if status["surface_state"] in ("insufficient_reference", "tracking_lost"):
            result.passed = False
            result.fail_reasons = [f"deforming_surface_{status['surface_state']}"]
            result.message = f"Surface {status['surface_state'].replace('_', ' ')}"
        result.score = status.get("tracked_fraction", 0.0)
        for name in ("surface_depth", "displacement"):
            if result.outputs[name] is not NO_VALUE:
                # These samples are indexed by reference material pixels.
                context = result.payloads[name]
                for key in (CAPTURE, APPLIED, "optical_frame"):
                    context.pop(key, None)
                context["surface_layout"] = {
                    "domain": "reference_material_pixels",
                    "camera_z_units": "m",
                    "reference_ticket": status["reference_ticket"],
                }
        return result
