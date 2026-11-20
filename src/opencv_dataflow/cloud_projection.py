"""Rasterize calibrated point samples and classify their visibility on the target grid."""

import cv2
import numpy as np

from .calibration_revisions import REVISION, applied_payload
from .core import Port, Result
from .geometry3d_common import Tool3D, camera_for_image, require_pinhole, rigid
from .images import DepthArray, ImageArray
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, frame, image_with_payload
from .selection import selection_mask
from .stereo3d import metric_depth

METERS = {"m": 1.0, "mm": 0.001}
MAX_PIXELS = 8_388_608


def camera_coordinates(cloud, target_frame, pose):
    if cloud.units not in METERS:
        raise ValidationError("Point-cloud projection requires metric cloud units (m or mm)")
    xyz = cloud.points.astype(np.float64) * METERS[cloud.units]
    if pose is None:
        if cloud.frame != target_frame:
            raise ValidationError("Connect a transform from the point-cloud frame to the camera")
        return xyz
    if (
        not isinstance(pose, dict)
        or pose.get("source_frame") != cloud.frame
        or pose.get("target_frame") != target_frame
        or pose.get("units") not in METERS
    ):
        raise ValidationError("Transform must map the cloud frame to the camera in metric units")
    transform = rigid(pose)
    return xyz @ transform[:3, :3].T + transform[:3, 3] * METERS[pose["units"]]


def pixel_cells(uv, size):
    """Return in-grid rows and their nearest pixel cells using half-open borders."""
    inside = np.isfinite(uv).all(axis=1) & (uv >= -0.5).all(axis=1)
    inside &= (uv < [size[0] - 0.5, size[1] - 0.5]).all(axis=1)
    x, y = np.floor(uv[inside] + 0.5).astype(np.int32).T
    return inside, y * size[0] + x


def raster_indices(xyz, candidates, k, distortion, size):
    """Use nearest pixel centers with bounded native scratch."""
    cells = np.full(len(candidates), -1, np.int32)
    zero = np.zeros(3, np.float64)
    for start in range(0, len(candidates), 16384):
        band = slice(start, start + 16384)
        uv, _ = cv2.projectPoints(xyz[candidates[band]], zero, zero, k, distortion)
        inside, indices = pixel_cells(uv.reshape(-1, 2), size)
        band_cells = cells[band]
        band_cells[inside] = indices
    return cells


def as_mask(size, cells):
    mask = np.zeros(size[::-1], np.uint8)
    mask.flat[cells] = 255
    return mask


class ProjectCloudVisibility(Tool3D):
    type_id, title, subtype = (
        "3d.cloud_project",
        "Project cloud and check visibility",
        "Point clouds",
    )
    parallel_safe = True
    process_inputs_readonly = True
    payload_reference = "cloud"
    inputs = (
        Port("cloud", "cloud"),
        Port("camera", "data"),
        Port("transform", "data", required=False, required_when_connected=True),
        Port("reference_depth", "depth", required=False, required_when_connected=True),
        Port("mask", required=False, required_when_connected=True),
        Port("region", "data", required=False, required_when_connected=True),
    )
    outputs = (
        Port("depth", "depth"),
        Port("coverage_mask", visual_type="region"),
        Port("color", required=False),
        Port("visible_mask", visual_type="region"),
        Port("occluded_mask", visual_type="region"),
        Port("unverified_mask", visual_type="region"),
        Port("visible_cloud", "cloud"),
        Port("occluded_cloud", "cloud"),
        Port("unverified_cloud", "cloud"),
        Port("report", "data"),
    )
    defaults = {
        "minimum_z_m": 0.001,
        "maximum_z_m": 10.0,
        "surface_tolerance_m": 0.0,
        "reference_tolerance_m": 0.01,
        "reference_units": "m",
        "reference_units_per_sample": 0.001,
    }
    parameter_choices = {"reference_units": ("m", "mm")}
    parameter_ranges = {
        "minimum_z_m": NumericRange(1e-6, 100_000),
        "maximum_z_m": NumericRange(1e-6, 100_000),
        "surface_tolerance_m": NumericRange(0, 100),
        "reference_tolerance_m": NumericRange(0, 100),
        "reference_units_per_sample": NumericRange(1e-9, 1000),
    }
    parameter_help = {
        "minimum_z_m": "Ignore samples nearer than this positive camera Z.",
        "maximum_z_m": "Ignore samples farther than this camera Z.",
        "surface_tolerance_m": "Maximum Z behind the nearest sample to share its visible surface.",
        "reference_tolerance_m": "Maximum Z difference for agreement with observed target depth.",
        "reference_units": "Length units for a depth reference without declared units.",
        "reference_units_per_sample": "Physical units per raw uint16 reference count.",
    }
    input_help = {
        "camera": "Target pinhole model and image dimensions, including applicable distortion.",
        "transform": "Rigid cloud-to-camera pose. Omit only when the cloud is in the camera frame.",
        "reference_depth": "Optional measured camera Z on this exact target grid.",
        "mask": "Limit results and visibility assessment to these target pixels.",
        "region": "Select target pixels, intersecting the mask when supplied.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if float(p["maximum_z_m"]) <= float(p["minimum_z_m"]):
            raise ValidationError("Maximum Z must exceed minimum Z", "maximum_z_m", "minimum_z_m")

    def process(self, inputs):
        cloud, camera, p = inputs["cloud"], inputs["camera"], self.parameters
        k, distortion, size = require_pinhole(camera)
        width, height = size
        if width * height > MAX_PIXELS:
            raise ValidationError("Target camera grid exceeds the 8,388,608 pixel projection limit")
        camera_frame = camera.get("frame", "camera")
        xyz = camera_coordinates(cloud, camera_frame, inputs.get("transform"))
        z = xyz[:, 2]
        lower, upper = float(p["minimum_z_m"]), float(p["maximum_z_m"])
        candidates = np.flatnonzero((z >= lower) & (z <= upper))

        reference = inputs.get("reference_depth")
        target = {"spatial": frame(size, f"{self.id}:target")}
        observed = None
        if reference is not None:
            camera_for_image(camera, reference)
            target.update(self.input_payloads["reference_depth"])
            observed = metric_depth(
                reference, p["reference_units_per_sample"], p["reference_units"]
            )
        target = applied_payload(target, camera.get(REVISION))

        cells = raster_indices(xyz, candidates, k, distortion, size)
        inside = cells >= 0
        points, cells = candidates[inside], cells[inside]
        in_frame_count = len(points)
        if "mask" in inputs or "region" in inputs:
            grid = image_with_payload(np.empty((height, width), np.uint8), target)
            selected = selection_mask(inputs, grid).ravel()[cells] != 0
            points, cells = points[selected], cells[selected]

        z_points = z[points]
        order = np.lexsort((points, z_points, cells))
        sorted_cells = cells[order]
        first = np.empty(len(order), bool)
        first[:1] = True
        first[1:] = sorted_cells[1:] != sorted_cells[:-1]
        unique_cells = sorted_cells[first]
        owners = points[order[first]]
        nearest_z = z[owners]
        depth = np.full((height, width), np.nan, np.float32)
        depth.flat[unique_cells] = nearest_z
        nearest_per_sample = nearest_z[np.searchsorted(unique_cells, cells)]
        self_hidden = z_points - nearest_per_sample > float(p["surface_tolerance_m"])

        missing = np.zeros(len(points), bool)
        in_front = missing.copy()
        reference_hidden = missing.copy()
        if observed is not None:
            measured = observed.ravel()[cells]
            remaining = ~self_hidden
            delta = z_points - measured
            tolerance = float(p["reference_tolerance_m"])
            missing = remaining & ~np.isfinite(measured)
            in_front = remaining & (delta < -tolerance)
            reference_hidden = remaining & (delta > tolerance)

        hidden = self_hidden | reference_hidden
        unverified = missing | in_front
        visible = ~(hidden | unverified)
        visible_count = int(np.count_nonzero(visible))
        behind_count = int(np.count_nonzero(z <= 0))

        def subset(status):
            return cloud.subset(points[status])

        color = NO_VALUE
        if cloud.colors is not None:
            color = np.zeros((height, width, 3), np.uint8)
            color.reshape(-1, 3)[unique_cells] = cloud.colors[owners]
            color = ImageArray(color, "rgb")
        outputs = {
            "depth": DepthArray(depth, "m", lower, upper),
            "coverage_mask": as_mask(size, unique_cells),
            "color": color,
            "visible_mask": as_mask(size, cells[visible]),
            "occluded_mask": as_mask(size, cells[hidden]),
            "unverified_mask": as_mask(size, cells[unverified]),
            "visible_cloud": subset(visible),
            "occluded_cloud": subset(hidden),
            "unverified_cloud": subset(unverified),
            "report": {
                "camera_frame": camera_frame,
                "units": "m",
                "image_size": list(size),
                "compared_to_reference": observed is not None,
                "cloud_points": len(z),
                "behind_camera_points": behind_count,
                "clipped_z_points": len(z) - len(candidates) - behind_count,
                "outside_camera_grid_points": len(candidates) - in_frame_count,
                "outside_selection_points": in_frame_count - len(points),
                "projected_points": len(points),
                "covered_pixels": len(unique_cells),
                "visible_points": visible_count,
                "self_occluded_points": int(np.count_nonzero(self_hidden)),
                "reference_occluded_points": int(np.count_nonzero(reference_hidden)),
                "missing_reference_points": int(np.count_nonzero(missing)),
                "in_front_of_reference_points": int(np.count_nonzero(in_front)),
            },
        }
        graphics = {}
        if self.capture_graphics:
            # BGR channels independently show unknown, visible and occluded support.
            graphics["Visibility"] = cv2.merge(
                (outputs["unverified_mask"], outputs["visible_mask"], outputs["occluded_mask"])
            )
        return Result(
            outputs,
            graphics=graphics,
            payloads={
                key: target for key, value in outputs.items() if isinstance(value, np.ndarray)
            },
            passed=bool(len(points)),
            score=visible_count / len(points) if len(points) else 0.0,
            fail_reasons=[] if len(points) else ["no_projected_cloud_samples"],
            message=f"{len(unique_cells)} covered pixels · {visible_count} visible points",
        )
