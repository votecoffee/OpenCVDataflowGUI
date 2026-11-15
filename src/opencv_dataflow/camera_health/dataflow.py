"""Camera health block with an explicitly staged immutable reference."""

import re
from pathlib import Path

import cv2
import numpy as np

from ..core import Block, Port, Result
from ..graphics_layers import GraphicDrawing, graphic_object
from ..images import image_format
from ..parameter_validation import NumericRange, ValidationError
from ..selection import rasterize, selection_mask
from .analysis import assess
from .model import TRAIN_DEFAULTS, load_reference, normalized

COLORS = {"healthy": (80, 210, 80), "warning": (0, 190, 255), "alarm": (60, 60, 240)}


class CameraHealth(Block):
    type_id, title = "inspect.camera_health", "Monitor camera health"
    category = block_type = "Inspection"
    subtype = "Camera monitoring"
    inputs = (Port("image"), Port("foreground_mask", required=False, required_when_connected=True))
    outputs = (
        Port("image"),
        Port("state", "text"),
        Port("healthy", "boolean"),
        Port("warning", "boolean"),
        Port("alarm", "boolean"),
        Port("report", "data"),
        Port("foreground_mask", visual_type="region"),
        Port("background_mask", visual_type="region"),
        Port("transform", "data", visual_type="pose"),
    )
    defaults = {
        "reference_path": "",
        "reference_sha256": "",
        "check_position": True,
        "check_lighting": True,
        "check_focus": True,
        "position_method": "background",
        "position_warning": 1.0,
        "position_alarm": 3.0,
        "rotation_warning": 0.25,
        "rotation_alarm": 1.0,
        "scale_warning": 0.5,
        "scale_alarm": 2.0,
        "lighting_warning": 10.0,
        "lighting_alarm": 20.0,
        "focus_warning": 20.0,
        "focus_alarm": 40.0,
        "auto_foreground": True,
        "foreground_threshold": 0.12,
        "foreground_min_area": 32,
        "foreground_dilation": 2,
        "minimum_region_coverage": 0.6,
        "lighting_floor": 0.02,
        "max_features": 600,
        "feature_quality": 0.01,
        "minimum_tracks": 12,
        "tracking_window": 21,
        "pyramid_levels": 3,
        "forward_backward_error": 1.0,
        "ransac_error": 1.5,
        "minimum_inlier_fraction": 0.5,
        "minimum_track_coverage": 0.03,
        "fiducial_search_margin": 32,
        "fiducial_min_score": 0.85,
        "fiducial_peak_margin": 0.02,
        **TRAIN_DEFAULTS,
    }
    parameter_choices = {
        "position_method": ("background", "fiducials"),
        "focus_method": ("tenengrad", "laplacian_variance"),
        "focus_aperture": ("1", "3", "5", "7"),
    }
    parameter_ranges = {
        **{
            f"{key}_{level}": NumericRange(0, 10000)
            for key in ("position", "rotation", "scale", "lighting", "focus")
            for level in ("warning", "alarm")
        },
        "foreground_threshold": NumericRange(0.001, 1),
        "foreground_min_area": NumericRange(1, 1_000_000, whole=True),
        "foreground_dilation": NumericRange(0, 32, whole=True),
        "minimum_region_coverage": NumericRange(0.01, 1),
        "lighting_floor": NumericRange(1e-6, 1),
        "max_features": NumericRange(16, 5000, whole=True),
        "feature_quality": NumericRange(0.0001, 0.5),
        "minimum_tracks": NumericRange(3, 5000, whole=True),
        "tracking_window": NumericRange(5, 101, whole=True, odd=True),
        "pyramid_levels": NumericRange(0, 5, whole=True),
        "forward_backward_error": NumericRange(0.05, 10),
        "ransac_error": NumericRange(0.1, 10),
        "minimum_inlier_fraction": NumericRange(0.1, 1),
        "minimum_track_coverage": NumericRange(0, 1),
        "fiducial_search_margin": NumericRange(1, 512, whole=True),
        "fiducial_min_score": NumericRange(0.1, 1),
        "fiducial_peak_margin": NumericRange(0, 1),
        "float_min": NumericRange(-1e12, 1e12),
        "float_max": NumericRange(-1e12, 1e12),
        "stability_threshold": NumericRange(0, 1),
    }
    parameter_help = {
        "reference_path": "Project-contained reference/history asset staged by the trainer.",
        "reference_sha256": "Pinned training bytes. Retrain/stage to change the baseline.",
        "check_position": "Check the fixed background or trained static fiducials.",
        "check_lighting": "Check visible corresponding channel means in each lighting region.",
        "check_focus": "Check sharpness loss in each focus region against the saved reference.",
        "position_method": "Stable background tracking or trained static-region matching.",
        "auto_foreground": "Reject coherent changes after rough motion/lighting normalization.",
        "foreground_threshold": "Normalized difference on [0,1] after lighting normalization.",
        "foreground_min_area": "Minimum coherent changed component area in pixels.",
        "foreground_dilation": "Extra foreground margin in pixels before tracking and scoring.",
        "minimum_region_coverage": "Required visible/stable fraction; missing evidence alarms.",
        "lighting_floor": "Minimum normalized mean used as denominator near black.",
        "max_features": "Maximum Shi-Tomasi reference corners.",
        "feature_quality": "Minimum corner quality relative to the strongest corner.",
        "minimum_tracks": "Minimum background inlier count after foreground removal.",
        "tracking_window": "Odd Lucas-Kanade window width/height in pixels.",
        "pyramid_levels": "Maximum LK pyramid level; zero disables pyramids.",
        "forward_backward_error": "Largest allowed round-trip feature error in pixels.",
        "ransac_error": "Inlier reprojection error in pixels for the similarity transform.",
        "minimum_inlier_fraction": "Minimum inlier fraction among eligible final matches.",
        "minimum_track_coverage": "Inlier convex-hull area / trained position selection area.",
        "fiducial_search_margin": "Search distance around each trained region, in pixels.",
        "fiducial_min_score": "Minimum masked normalized correlation for every fiducial.",
        "fiducial_peak_margin": "Required score separation from competing locations.",
        "focus_method": "Training metric. Changes require retraining.",
        "focus_aperture": "Derivative kernel; keep fixed between training and inspection.",
        "normalize_focus": "Divide sharpness by region variance to reduce gain sensitivity.",
        "float_min": "Lower bound for floating samples; changes require retraining.",
        "float_max": "Upper bound for floating samples, usually one or 255; retrain to change.",
        "stability_threshold": "Largest stable-pixel history range on [0,1]; retrain to change.",
        **{
            f"{key}_{limit}": f"{limit.title()} at absolute {description}. Warning < alarm."
            for key, description in (
                ("position", "image-center displacement in pixels"),
                ("rotation", "rotation in degrees (background or >=2 fiducials)"),
                ("scale", "scale change in percent (two fiducials or background)"),
                ("lighting", "per-channel mean change in percent"),
                ("focus", "sharpness loss in percent"),
            )
            for limit in ("warning", "alarm")
        },
    }
    input_help = {
        "image": "Current image on the trained pixel grid, with matching depth/encoding.",
        "foreground_mask": "Current-image uint8 mask: nonzero means moving/occluded pixels.",
    }
    plugin_dialog_label = "Train camera reference…"
    plugin_dialog_modeless = True
    project_asset_parameters = ("reference_path",)
    payload_reference = "image"
    parallel_safe = preview_safe = process_inputs_readonly = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = ["image", "state", "warning", "alarm", "report"]
        self.reset_runtime()

    def reset_runtime(self):
        self._reference = self._reference_key = None

    def copy_runtime_from(self, other):
        self.reset_runtime()

    def create_plugin_dialog(self, parent, parameters, context):
        from .dialog import CameraHealthTrainer

        return CameraHealthTrainer(parent, parameters, context)

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for key in (
            "check_position",
            "check_lighting",
            "check_focus",
            "normalize_focus",
            "auto_foreground",
        ):
            if type(p[key]) is not bool:
                raise ValidationError("Use a boolean setting", key)
        if not any(p[k] for k in ("check_position", "check_lighting", "check_focus")):
            raise ValidationError("Enable at least one camera health check")
        for key in ("position", "rotation", "scale", "lighting", "focus"):
            if p[f"{key}_warning"] >= p[f"{key}_alarm"]:
                raise ValidationError(
                    "Warning must be below alarm", f"{key}_warning", f"{key}_alarm"
                )
        if p["float_min"] >= p["float_max"]:
            raise ValidationError("Float min must be below Float max", "float_min", "float_max")
        if p["minimum_tracks"] > p["max_features"]:
            raise ValidationError(
                "Minimum tracks exceeds Max features", "minimum_tracks", "max_features"
            )
        if not isinstance(p["reference_path"], str) or not isinstance(p["reference_sha256"], str):
            raise ValidationError("Use a trained reference path and hash")
        if p["reference_path"] and not re.fullmatch(r"[0-9a-f]{64}", p["reference_sha256"]):
            raise ValidationError("Train/stage to obtain the reference SHA-256", "reference_sha256")

    def process(self, inputs):
        p, image = self.parameters, inputs["image"]
        if not p["reference_path"]:
            raise ValidationError(
                "Train and stage a camera reference before running", "reference_path"
            )
        path = Path(p["reference_path"])
        if not path.is_absolute():
            if self.project_dir is None:
                raise ValidationError("Save the project before loading its camera reference")
            path = Path(self.project_dir) / path
        stat = path.stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, p["reference_sha256"])
        if key != self._reference_key:
            self._reference = load_reference(path, p["reference_sha256"])
            self._reference_key = key
        model = self._reference
        if any(p[k] != model.info["settings"][k] for k in TRAIN_DEFAULTS):
            raise ValidationError("Training settings changed; retrain the camera reference")
        if (
            list(image.shape) != model.info["input_shape"]
            or image.dtype.name != model.info["dtype"]
        ):
            raise ValidationError("Current image size/depth differs from training")
        if image_format(image) != model.info["encoding"]:
            raise ValidationError("Current image color encoding differs from training")
        old = model.info["captures"][0].get("spatial")
        current = self.input_payloads["image"]["spatial"]
        if old and (
            old["root_size"] != current["root_size"]
            or not np.allclose(old["to_root"], current["to_root"], atol=1e-9, rtol=0)
        ):
            raise ValidationError("Current crop/scale grid differs from training")
        values = normalized(image, p)
        mask = selection_mask({"mask": inputs.get("foreground_mask")}, image)
        external = np.zeros(image.shape[:2], bool) if mask is None else mask != 0
        report, changed = assess(model, values, external, p)
        report["reference_sha256"] = p["reference_sha256"]
        report["thresholds"] = {k: v for k, v in p.items() if k.endswith(("_warning", "_alarm"))}
        state, matrix = report["state"], report["transform"]
        failures = [
            {
                "code": f"camera_{r['check']}_{r.get('reason', 'alarm')}",
                "message": f"{r['name']}: {r.get('reason', 'alarm threshold reached')}",
            }
            for r in report["checks"]
            if r["state"] == "alarm"
        ]
        background = np.zeros(image.shape[:2], np.uint8)
        if matrix is not None:
            background = cv2.warpAffine(
                model.stable, np.asarray(matrix), image.shape[1::-1], flags=cv2.INTER_NEAREST
            )
            background[changed] = 0
        result = Result(
            {
                "image": image,
                "state": state,
                "healthy": state == "healthy",
                "warning": state == "warning",
                "alarm": state == "alarm",
                "report": report,
                "foreground_mask": changed.astype(np.uint8) * 255,
                "background_mask": background,
                "transform": {
                    "model": "image_affine_v1",
                    "source_to_target": np.vstack([matrix, [0, 0, 1]]).tolist(),
                    "source_size": model.info["image_size"],
                    "target_size": model.info["image_size"],
                }
                if matrix is not None
                else None,
            },
            passed=state != "alarm",
            score={"healthy": 1.0, "warning": 0.5, "alarm": 0.0}[state],
            message=f"Camera health: {state}",
            fail_reasons=failures,
            event_data={"camera_health": report},
        )
        if state != "healthy":
            self.logger.warning(
                f"Camera health {state}: "
                + "; ".join(r["name"] for r in report["checks"] if r["state"] != "healthy")
            )
        if self.generate_graphics:
            objects = []
            for role in ("lighting", "focus", "fiducials"):
                for row in model.info["regions"][role]:
                    region = rasterize(row["region"], image.shape)
                    if matrix is not None:
                        region = cv2.warpAffine(
                            region, np.asarray(matrix), image.shape[1::-1], flags=cv2.INTER_NEAREST
                        )
                    contours, _ = cv2.findContours(
                        region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    status = next(
                        (
                            r["state"]
                            for r in report["checks"]
                            if r["name"] == row["name"] and r["check"] == role
                        ),
                        state,
                    )
                    objects.extend(
                        graphic_object("polyline", c.reshape(-1, 2), COLORS[status], closed=True)
                        for c in contours
                    )
            objects.append(
                graphic_object("text", [(8, 22)], COLORS[state], text=f"Camera: {state}")
            )
            result.drawings["camera health"] = GraphicDrawing(tuple(objects))
        return result
