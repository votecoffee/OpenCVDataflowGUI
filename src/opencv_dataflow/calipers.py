"""Pose-relative intensity scans for paired-edge width and continuity inspection."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .features import overlay_image
from .payloads import NO_VALUE
from .poses import pose_for_image
from .processing import finite, gray_image, integer
from .selection import SELECTION_INPUTS, selection_mask


def paired_edges(profile, step, polarity, minimum):
    gradient = polarity * np.diff(profile) / step
    rising = np.flatnonzero((gradient > 0) & (gradient >= minimum))
    falling = np.flatnonzero((gradient < 0) & (-gradient >= minimum))
    if not rising.size or not falling.size:
        return None
    strength = np.minimum(gradient[rising, None], -gradient[None, falling])
    strength[rising[:, None] >= falling] = 0
    i, j = np.unravel_index(np.argmax(strength), strength.shape)
    return np.array([rising[i] + 0.5, falling[j] + 0.5]) if strength[i, j] > 0 else None


class BeadCalipers(Block):
    type_id, title = "inspect.bead_calipers", "Paired-edge bead calipers"
    category, block_type, subtype = "Inspection", "Inspection", "Measurements"
    inputs = (*SELECTION_INPUTS, Port("pose", "data", required=False))
    outputs = (
        Port("measurements", "data"),
        Port("points", "data"),
        Port("mean_width", "number", required=False),
        Port("pass_fraction", "number"),
        Port("longest_gap", "number"),
        Port("overlay", required=False),
    )
    defaults = {
        "start": [20.0, 50.0],
        "end": [80.0, 50.0],
        "scan_count": 5,
        "scan_span": 20.0,
        "samples": 121,
        "smoothing_sigma": 0.8,
        "polarity": "bright",
        "interpolation": "linear",
        "min_gradient": 10.0,
        "min_width": 0.0,
        "max_width": 0.0,
        "max_center_offset": -1.0,
        "min_pass_fraction": 1.0,
        "maximum_gap": -1,
        "draw_overlay": True,
    }
    parameter_choices = {"polarity": ("bright", "dark"), "interpolation": ("linear", "nearest")}
    parameter_help = {
        "start": "Base scan start [x,y] in image pixels or local pose pixels.",
        "end": "Base scan end [x,y] in the same frame as start.",
        "scan_span": "Total spacing from first to last scan, perpendicular to the base scan.",
        "smoothing_sigma": "Gaussian smoothing along each scan, in sample units. Zero disables.",
        "min_gradient": "Minimum intensity change per local pixel for each edge.",
        "max_width": "Maximum accepted edge spacing. Zero disables the upper bound.",
        "max_center_offset": "Maximum deviation from scan midpoint. Negative disables the check.",
        "maximum_gap": "Maximum consecutive scans with missing measurements. -1 disables.",
    }

    def process(self, inputs):
        source = inputs["image"]
        image = gray_image(source, (np.uint8, np.uint16, np.float32))
        height, width = image.shape
        if max(height, width) >= 32767:
            raise ValueError("Caliper remapping requires image axes below 32767 pixels")
        p = self.parameters
        endpoints = np.asarray([p["start"], p["end"]], np.float64)
        if (
            endpoints.shape != (2, 2)
            or not np.isfinite(endpoints).all()
            or np.max(np.abs(endpoints)) > 1_000_000
        ):
            raise ValueError("Caliper endpoints must be finite [x,y] positions")
        delta = endpoints[1] - endpoints[0]
        length = float(np.linalg.norm(delta))
        if length <= 1e-6:
            raise ValueError("Caliper endpoints must differ")
        samples = integer(p["samples"], "Samples", 3, 1024)
        scans = integer(p["scan_count"], "Scan count", 1, 128)
        span = finite(p["scan_span"], "Scan span", 0)
        sigma = finite(p["smoothing_sigma"], "Smoothing sigma", 0)
        if span > 1_000_000 or sigma > 100:
            raise ValueError("Scan span and smoothing sigma exceed their limits")
        minimum_gradient = finite(p["min_gradient"], "Minimum gradient", 0)
        low = finite(p["min_width"], "Minimum width", 0)
        high = finite(p["max_width"], "Maximum width", 0)
        offset_limit = finite(p["max_center_offset"], "Center offset limit", -1)
        minimum_fraction = finite(p["min_pass_fraction"], "Minimum pass fraction", 0)
        if (high and high < low) or minimum_fraction > 1:
            raise ValueError("Invalid width range or pass fraction")
        maximum_gap = integer(p["maximum_gap"], "Maximum gap", -1, 128)
        distances = np.linspace(0, length, samples)
        step = length / (samples - 1)
        normal = np.array([-delta[1], delta[0]]) / length
        offsets = np.linspace(-span / 2, span / 2, scans) if scans > 1 else np.zeros(1)
        local = endpoints[0] + offsets[:, None, None] * normal + distances[:, None] * delta / length
        pose = inputs.get("pose")
        matrix = (
            np.eye(3)
            if pose is None
            else np.asarray(pose_for_image(pose, source)["source_to_target"])
        )
        homogeneous = np.concatenate((local, np.ones((*local.shape[:2], 1))), axis=-1)
        projected = homogeneous @ matrix.T
        if not np.isfinite(projected).all() or np.any(projected[..., 2] <= 1e-9):
            raise ValueError("Caliper scan crosses the pose projection horizon")
        locations = projected[..., :2] / projected[..., 2:]
        in_image = np.all((locations >= 0) & (locations <= [width - 1, height - 1]), axis=(1, 2))
        maps = np.where(in_image[:, None, None], locations, 0).astype(np.float32)
        method = cv2.INTER_LINEAR if p["interpolation"] == "linear" else cv2.INTER_NEAREST
        profiles = cv2.remap(image.astype(np.float32, copy=False), maps, None, method)
        profiles = profiles.astype(np.float64)
        selection = selection_mask(inputs, image)
        in_selection = (
            np.ones(scans, bool)
            if selection is None
            else np.all(cv2.remap(selection.astype(np.float32), maps, None, method) == 255, axis=1)
        )
        if sigma:
            window = 2 * int(np.ceil(3 * sigma)) + 1
            profiles = cv2.GaussianBlur(
                profiles, (window, 1), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE
            )
        overlay = overlay_image(source) if p["draw_overlay"] else NO_VALUE
        polarity = 1 if p["polarity"] == "bright" else -1
        rows, centers, measured = [], [], []
        passing, gap, longest_gap = 0, 0, 0
        for index, profile in enumerate(profiles):
            row = {"scan": index, "scan_offset": float(offsets[index])}
            if not in_image[index]:
                reason, edges = "scan_outside_image", None
            elif not in_selection[index]:
                reason, edges = "scan_outside_selection", None
            else:
                reason = "edge_pair_missing"
                edges = paired_edges(profile, step, polarity, minimum_gradient)
            if edges is None:
                gap += 1
                longest_gap = max(longest_gap, gap)
            else:
                gap = 0
                bead_width = float((edges[1] - edges[0]) * step)
                center_offset = float(edges.mean() * step - length / 2)
                indices = np.append(edges, edges.mean())
                # Project sub-sample positions from the local scan to preserve perspective.
                sample_points = local[index, 0] + indices[:, None] * delta / (samples - 1)
                a, b, center = cv2.perspectiveTransform(sample_points[None], matrix)[0]
                measured.append(bead_width)
                centers.append(center.tolist())
                if bead_width < low or (high and bead_width > high):
                    reason = "width_out_of_tolerance"
                elif offset_limit >= 0 and abs(center_offset) > offset_limit:
                    reason = "center_out_of_tolerance"
                else:
                    reason = None
                    passing += 1
                row.update(
                    width=bead_width, center_offset=center_offset, edges=[a.tolist(), b.tolist()]
                )
            row["reason"] = reason
            rows.append(row)
            if overlay is not NO_VALUE and in_image[index]:
                color = (60, 80, 240) if reason else (70, 220, 140)
                start, end = np.rint(locations[index, [0, -1]]).astype(np.int32)
                cv2.line(overlay, tuple(start), tuple(end), color, 1)
                if edges is not None:
                    cv2.line(
                        overlay,
                        tuple(np.rint(a).astype(int)),
                        tuple(np.rint(b).astype(int)),
                        color,
                        2,
                    )
        fraction = passing / scans
        reasons = []
        if not measured or fraction < minimum_fraction:
            reasons.append("bead_pass_fraction_below_minimum")
        if maximum_gap >= 0 and longest_gap > maximum_gap:
            reasons.append("bead_gap_above_maximum")
        return Result(
            {
                "measurements": rows,
                "points": centers,
                "mean_width": float(np.mean(measured)) if measured else NO_VALUE,
                "pass_fraction": fraction,
                "longest_gap": longest_gap,
                "overlay": overlay,
            },
            graphics={"bead calipers": overlay} if overlay is not NO_VALUE else {},
            passed=not reasons,
            score=fraction,
            message=f"{passing} of {scans} caliper scans pass, longest gap {longest_gap}",
            fail_reasons=reasons,
        )
