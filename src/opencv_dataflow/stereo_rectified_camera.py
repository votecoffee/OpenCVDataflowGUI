"""Expose the rectified left pinhole and its pose in the original camera frame."""

import numpy as np

from .core import Port, Result
from .geometry3d_common import Tool3D, camera, camera_record, image_size, matrix, pose_record
from .parameter_validation import ValidationError


class StereoRectifiedCamera(Tool3D):
    type_id, title, subtype = (
        "3d.stereo_rectified_camera",
        "Rectified stereo camera geometry",
        "Stereo",
    )
    inputs = (Port("rectification", "data"), Port("left_camera", "data"))
    outputs = (Port("camera", "data"), Port("transform", "data", visual_type="pose"))
    input_help = {
        "rectification": "Calibration from Rectify stereo images, including P1 and R1.",
        "left_camera": "Original left camera used to make this stereo rectification.",
    }
    parallel_safe = preview_safe = process_inputs_readonly = True

    def process(self, inputs):
        rec, original = inputs["rectification"], inputs["left_camera"]
        camera(original)
        source_frame = original.get("frame", "camera")
        if (
            not isinstance(rec, dict)
            or rec.get("model") != "stereo_rectification_v1"
            or rec.get("units") not in ("m", "mm")
            or rec.get("units") != original.get("units")
            or rec.get("frame") != f"{source_frame}:rectified"
        ):
            raise ValidationError("Rectification must describe the original metric left camera")
        size = image_size(rec.get("image_size"))
        projection = matrix(rec.get("P1"), (3, 4), "Rectified left projection")
        if not np.allclose(projection[:, 3], 0, atol=1e-10, rtol=0):
            raise ValidationError("Rectified left projection must be centered at its camera origin")
        rectified = camera_record(projection[:, :3], [], size, rec["units"], rec["frame"])
        transform = np.eye(4)
        # R1 maps original left camera XYZ to rectified XYZ. Map back with its inverse.
        transform[:3, :3] = matrix(rec.get("R1"), (3, 3), "Left rectification rotation").T
        pose = pose_record(transform, rec["frame"], source_frame, rec["units"])
        return Result({"camera": rectified, "transform": pose}, message=f"Rectified {source_frame}")
