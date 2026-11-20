"""Seeded metric pose tracking against an object's projected silhouette."""

import copy
from dataclasses import dataclass

import cv2
import numpy as np

from .core import Port, Result
from .geometry3d_common import Tool3D, camera_for_image, pose_matrix, pose_record, rigid
from .graphics_layers import GraphicDrawing, graphic_object
from .meshes import TriangleMesh
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .processing import gray_image
from .tickets import ticket_key

_MESH_LIMIT = 4096
_METERS = {"m": 1, "mm": 0.001}


def _mesh(value):
    if isinstance(value, dict):
        for name in ("vertices", "faces"):
            if (
                not isinstance(value.get(name), (list, np.ndarray))
                or len(value[name]) > _MESH_LIMIT
            ):
                raise ValidationError(f"Supply mesh {name} with at most {_MESH_LIMIT} entries")
        value = TriangleMesh(
            value["vertices"], value["faces"], value.get("units"), value.get("frame")
        )
    if not isinstance(value, TriangleMesh) or value.units not in _METERS:
        raise ValidationError("Supply a triangle mesh with meter or millimeter units")
    if len(value.vertices) > _MESH_LIMIT or len(value.faces) > _MESH_LIMIT:
        raise ValidationError(f"Tracker mesh limit is {_MESH_LIMIT} vertices and triangles")
    value.validate()
    vertices = value.vertices * _METERS[value.units]
    faces = value.faces
    triangles = vertices[faces].astype(np.float64)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    if np.any(np.linalg.norm(normals, axis=1) == 0):
        raise ValidationError("Silhouette mesh includes zero-area triangles")
    return value.frame, vertices.reshape(-1, 1, 3), faces.reshape(-1, 1, 3)


def _seed(value, object_frame, camera_frame):
    if (
        not isinstance(value, dict)
        or value.get("model") != "pose_3d_v1"
        or value.get("source_frame") != object_frame
        or value.get("target_frame") != camera_frame
        or value.get("units") not in _METERS
    ):
        raise ValidationError("Initial pose must transform the named metric object into the camera")
    transform = rigid(value)
    rvec = cv2.Rodrigues(transform[:3, :3])[0].ravel()
    return np.r_[rvec, transform[:3, 3] * _METERS[value["units"]]]


def _projection(vertices, pose, camera, size, radius):
    if not np.isfinite(pose).all():
        return None
    rotation = cv2.Rodrigues(pose[:3])[0]
    xyz = vertices.reshape(-1, 3)
    depths = xyz @ rotation[2, :] + pose[5]
    if np.any(depths <= 0):
        return None
    pixels = cv2.projectPoints(vertices, pose[:3], pose[3:], camera, None)[0].reshape(-1, 2)
    margin = radius + 1
    if (
        not np.isfinite(pixels).all()
        or np.any(pixels < margin)
        or np.any(pixels >= np.array(size) - margin)
    ):
        return None
    return pixels


def _edge_matches(gray, controls, radius, edge_min):
    # Integer centers keep the dominant-axis span exactly 2 * radius + 1 pixels.
    centers = np.rint(controls).reshape(-1, 2)
    tangent = np.roll(centers, -1, axis=0) - np.roll(centers, 1, axis=0)
    if np.any(np.all(tangent == 0, axis=1)):
        return None
    bundle, locations = cv2.rapid.extractLineBundle(radius, centers.reshape(-1, 1, 2), gray)
    columns, response = cv2.rapid.findCorrespondencies(bundle)
    best = columns.ravel()
    selected = (response.ravel() >= edge_min) & (best > 0) & (best < bundle.shape[1] - 1)
    best, locations = best[selected], locations[selected]
    samples = np.clip(best[:, None] + [-1, 0, 1], 1, bundle.shape[1] - 2)
    rows = np.arange(len(best))[:, None]
    selected_bundle = bundle[selected].astype(np.float64)
    weights = np.abs(selected_bundle[rows, samples + 1] - selected_bundle[rows, samples - 1])
    observed = (locations[rows, samples] * weights[:, :, None]).sum(1) / weights.sum(1)[:, None]
    directions = (locations[:, -1] - locations[:, 0]).astype(np.float64)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return selected, observed, directions


def _fit_normal(pose, xyz, observed, directions, camera, iterations):
    from scipy.optimize import least_squares

    def residual(candidate):
        projected, _ = cv2.projectPoints(xyz, candidate[:3], candidate[3:], camera, None)
        return np.einsum("ij,ij->i", projected.reshape(-1, 2) - observed, directions)

    def jacobian(candidate):
        _, jac = cv2.projectPoints(xyz, candidate[:3], candidate[3:], camera, None)
        return np.einsum("ijk,ij->ik", jac[:, :6].reshape(-1, 2, 6), directions)

    fit = least_squares(residual, pose, jac=jacobian, method="lm", max_nfev=iterations)
    scales = np.linalg.norm(fit.jac, axis=0)
    if np.any(scales == 0) or np.linalg.cond(fit.jac / scales) > 1000:
        return None
    return fit.x, float(np.sqrt(np.mean(fit.fun**2)))


def _drawings(pixels, faces, observed):
    # Draw a bounded wireframe, plus small marks at the matched image edges.
    edges = np.concatenate([faces[:, :, (0, 1)], faces[:, :, (1, 2)], faces[:, :, (0, 2)]])
    edges = np.unique(np.sort(edges.reshape(-1, 2), axis=1), axis=0)
    indices = np.linspace(0, len(edges) - 1, min(256, len(edges)), dtype=int)
    wire = tuple(
        graphic_object("polyline", pixels[edge], (80, 235, 125), thickness=1)
        for edge in edges[indices]
    )
    marks = tuple(
        graphic_object("polyline", ((x - 1, y), (x + 1, y)), (45, 200, 255), thickness=1)
        for x, y in observed
    )
    return {"Tracked mesh": GraphicDrawing(wire), "Matched edges": GraphicDrawing(marks)}


@dataclass
class _Track:
    context: tuple
    estimate: np.ndarray | None = None
    lost: bool = False


class TrackObjectSilhouette(Tool3D):
    """Refine a known pose using the edge normal along a projected triangle model."""

    type_id, title, subtype = (
        "3d.track_silhouette",
        "Track metric object silhouette",
        "Pose tracking",
    )
    inputs = (
        Port("image"),
        Port("camera", "data"),
        Port("mesh", "mesh", required=False),
        Port("initial_pose", "data", required=False, visual_type="pose"),
        Port("reset", "data", required=False),
    )
    outputs = (
        Port("pose", "data", required=False, visual_type="pose"),
        Port("tracked", "data"),
        Port("report", "data"),
    )
    defaults = {
        "model": {},
        "initial_pose": {},
        "search_lines": 120,
        "search_radius_px": 10,
        "minimum_edge_response": 30,
        "minimum_match_fraction": 0.7,
        "maximum_reprojection_rms_px": 1.25,
        "pose_updates": 4,
        "fit_iterations": 24,
    }
    parameter_ranges = {
        "search_lines": NumericRange(16, 512, whole=True),
        "search_radius_px": NumericRange(2, 96, whole=True),
        "minimum_edge_response": NumericRange(1, 255, whole=True),
        "minimum_match_fraction": NumericRange(0.1, 1),
        "maximum_reprojection_rms_px": NumericRange(0.05, 32),
        "pose_updates": NumericRange(1, 20, whole=True),
        "fit_iterations": NumericRange(1, 80, whole=True),
    }
    parameter_help = {
        "model": "Object-space mesh with vertices, triangle faces, units (m/mm) and frame.",
        "initial_pose": "pose_3d_v1 record for the first frame or after an explicit reset.",
        "search_lines": "Number of normal searches spaced along the projected silhouette.",
        "search_radius_px": "Distance in pixels searched on each side of the predicted silhouette.",
        "minimum_edge_response": "Minimum intensity difference across an edge sample.",
        "minimum_match_fraction": "Minimum fraction of the requested searches with a valid edge.",
        "maximum_reprojection_rms_px": "Maximum RMS error along the edge normals.",
        "pose_updates": "Number of silhouette correspondence and pose-fit updates per frame.",
        "fit_iterations": "Maximum least-squares evaluations per pose update.",
    }
    input_help = {
        "image": "Consecutive uint8 frames from one stream, with a visible object boundary.",
        "camera": "Pinhole intrinsics for these undistorted frames.",
        "mesh": "Metric outward-wound triangle mesh. Overrides Model.",
        "initial_pose": "Seed for startup or reset. Overrides the configured pose.",
        "reset": "True clears tracking and tries the seed on this frame.",
    }
    flow_primary = ("pose",)
    payload_reference = "image"
    sequence_memory = requires_serial_execution = process_inputs_readonly = True
    parallel_safe = preview_safe = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset_runtime()

    def reset_runtime(self):
        self._track = None

    def copy_runtime_from(self, other):
        self._track = copy.deepcopy(other._track)

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["model"] != {}:
            _mesh(p["model"])
        if p["initial_pose"] != {}:
            rigid(p["initial_pose"])

    def _result(self, status, report, pose=NO_VALUE, drawings=None):
        report.update(model="silhouette_pose_tracking_v1", status=status)
        tracked = status == "tracked"
        reason = status.replace("_", " ")
        return Result(
            {"pose": pose, "tracked": tracked, "report": report},
            passed=tracked,
            score=report["match_fraction"],
            fail_reasons=[] if tracked else [{"code": f"silhouette_{status}", "message": reason}],
            event_data=report,
            drawings=drawings or {},
            message="Silhouette tracked" if tracked else f"Silhouette tracking · {reason}",
        )

    def _lost(self, reason, report):
        self._track.lost = True
        self._track.estimate = None
        return self._result(reason, report)

    def process(self, inputs):
        p = self.parameters
        native = getattr(cv2, "rapid", None)
        if not all(
            callable(getattr(native, name, None))
            for name in ("extractControlPoints", "extractLineBundle", "findCorrespondencies")
        ):
            raise RuntimeError("Install an OpenCV contrib build with RAPID silhouette support")
        image, record = inputs["image"], inputs["camera"]
        if image.shape[0] * image.shape[1] > 8_388_608:
            raise ValidationError("Silhouette tracking accepts at most 8,388,608 image pixels")
        k, distortion, size = camera_for_image(record, image)
        if record.get("lens", "pinhole") != "pinhole" or np.any(distortion):
            raise ValidationError("Silhouette tracking requires undistorted pinhole frames")
        frame, vertices, faces = _mesh(inputs.get("mesh", p["model"]))
        target = record.get("frame", "camera")
        if target == frame:
            raise ValidationError("The object and optical camera frames must have distinct names")
        image_payload = self.input_payloads.get("image", {})
        if image_payload.get("optical_frame") not in (None, target):
            raise ValidationError("Input optical frame differs from the camera model")
        reset = inputs.get("reset", False)
        if not isinstance(reset, (bool, np.bool_)):
            raise ValidationError("Reset must be a boolean")

        count, radius = int(p["search_lines"]), int(p["search_radius_px"])
        context = (
            frame,
            target,
            vertices.tobytes(),
            faces.tobytes(),
            k.tobytes(),
            size,
            image_payload.get("video_frame", {}).get("filename"),
            tuple(p[key] for key in self.parameter_ranges),
        )
        if self._track is None or self._track.context != context or reset:
            self._track = _Track(context)
        state = self._track
        report = {
            "search_lines": count,
            "matched_lines": 0,
            "match_fraction": 0.0,
            "reprojection_rms_px": None,
            "updates": 0,
            "seeded_this_frame": False,
        }
        if state.lost:
            return self._result("reset_required", report)
        if state.estimate is None:
            seed = inputs.get("initial_pose", p["initial_pose"])
            if isinstance(seed, dict) and not seed:
                return self._result("initial_pose_required", report)
            ports = ("image", "camera", "mesh", "initial_pose")
            tickets = {
                ticket_key(self.input_payloads[name]["ticket"])
                for name in ports
                if "ticket" in self.input_payloads.get(name, {})
            }
            if len(tickets) > 1:
                raise ValidationError(
                    "Frame and initialization inputs must share a ticket when supplied"
                )
            estimate = _seed(seed, frame, target)
            report["seeded_this_frame"] = True
        else:
            estimate = state.estimate
        gray = gray_image(image, (np.uint8,))
        for _ in range(int(p["pose_updates"])):
            if _projection(vertices, estimate, k, size, radius) is None:
                return self._lost("projection_outside_search_area", report)
            try:
                controls, xyz = cv2.rapid.extractControlPoints(
                    count, radius, vertices, estimate[:3], estimate[3:], k, size, faces
                )
            except cv2.error as exc:
                if exc.code != -215:
                    raise
                return self._lost("silhouette_unavailable", report)
            if (
                controls is None
                or xyz is None
                or len(controls) < 8
                or not np.isfinite(controls).all()
                or not np.isfinite(xyz).all()
            ):
                return self._lost("silhouette_unavailable", report)
            matches = _edge_matches(gray, controls, radius, p["minimum_edge_response"])
            if matches is None:
                return self._lost("silhouette_unavailable", report)
            selected, observed, directions = matches
            matched = len(observed)
            fraction = matched / count
            report.update(matched_lines=matched, match_fraction=fraction, reprojection_rms_px=None)
            if matched < 8 or fraction < float(p["minimum_match_fraction"]):
                return self._lost("insufficient_edges", report)
            fit = _fit_normal(
                estimate,
                xyz.reshape(-1, 3)[selected].astype(np.float64),
                observed,
                directions,
                k,
                int(p["fit_iterations"]),
            )
            if fit is None:
                return self._lost("pose_unobservable", report)
            estimate, rms = fit
            if not np.isfinite(rms):
                return self._lost("nonfinite_reprojection", report)
            report.update(reprojection_rms_px=rms, updates=report["updates"] + 1)
        pixels = _projection(vertices, estimate, k, size, radius)
        if pixels is None:
            return self._lost("projection_outside_search_area", report)
        if rms > float(p["maximum_reprojection_rms_px"]):
            return self._lost("reprojection_limit_exceeded", report)

        state.estimate = estimate
        pose = pose_record(pose_matrix(cv2.Rodrigues(estimate[:3])[0], estimate[3:]), frame, target)
        drawings = _drawings(pixels, faces, observed) if self.generate_graphics else None
        return self._result("tracked", report, pose, drawings)
