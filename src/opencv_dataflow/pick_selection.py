"""Filter image-plane pick candidates and rank the ones with visible clearance."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .features import overlay_image
from .payloads import NO_VALUE
from .poses import pose_for_image
from .processing import finite, integer
from .selection import contains, selection_mask


def projected_rectangle(rectangle, matrix, margin=0):
    x, y, width, height = rectangle
    low = np.array([x - margin, y - margin])
    high = np.array([x + width + margin, y + height + margin])
    corners = np.array([low, [high[0], low[1]], high, [low[0], high[1]]])
    if np.any(corners @ matrix[2, :2] + matrix[2, 2] <= 1e-9):
        raise ValueError("Pick clearance rectangle crosses the pose projection horizon")
    polygon = cv2.perspectiveTransform(corners[None], matrix)[0].astype(np.float32)
    if (
        not np.isfinite(polygon).all()
        or np.max(np.abs(polygon)) > 1_000_000
        or not cv2.isContourConvex(polygon)
        or cv2.contourArea(polygon) <= 0
    ):
        raise ValueError("Pick clearance projection must produce a finite convex rectangle")
    return polygon


def ring_occupancy(mask, outer, footprint):
    lower = np.floor(outer.min(axis=0)).astype(int)
    upper = np.ceil(outer.max(axis=0)).astype(int) + 1
    x, y = lower
    right, bottom = upper
    ring = np.zeros((bottom - y, right - x), np.uint8)
    for polygon, value in ((outer, 255), (footprint, 0)):
        vertices = np.rint((polygon - lower) * 256).astype(np.int32)
        cv2.fillConvexPoly(ring, vertices, value, shift=8)
    return int(np.count_nonzero(ring & mask[y:bottom, x:right]))


class SelectPickCandidate(Block):
    type_id, title = "inspect.select_pick", "Select pick candidate"
    category, block_type, subtype = "Inspection", "Inspection", "Pick selection"
    inputs = (
        Port("image"),
        Port("matches", "data"),
        Port("occupancy", required=False, visual_type="mask"),
        Port("region", "data", required=False),
    )
    outputs = (
        Port("pose", "data", required=False),
        Port("selected", "data", required=False),
        Port("candidates", "data"),
        Port("count", "number"),
        Port("overlay", required=False),
    )
    defaults = {
        "footprint": [-16.0, -16.0, 32.0, 32.0],
        "clearance": 8.0,
        "min_score": 0.8,
        "target_angle": 0.0,
        "angle_tolerance": 180.0,
        "angle_period": "360",
        "ranking": "downstream",
        "downstream_direction": [0.0, 1.0],
        "maximum_occupied_pixels": 0,
        "require_occupancy": False,
        "draw_overlay": True,
    }
    parameter_choices = {"angle_period": ("360", "180", "90"), "ranking": ("downstream", "score")}
    parameter_help = {
        "footprint": "Part rectangle [x, y, width, height] relative to each supplied pose.",
        "clearance": "Extra margin on all sides of the footprint in local pose pixels.",
        "target_angle": "Desired image pose angle, counterclockwise in degrees.",
        "angle_period": "Set 180 or 90 for a part with twofold or fourfold rotational symmetry.",
        "downstream_direction": "Image-space vector [dx, dy], with positive Y down.",
        "maximum_occupied_pixels": "Maximum nonzero occupancy pixels outside the part footprint.",
        "require_occupancy": "Require an obstacle mask in addition to match footprints.",
    }

    def process(self, inputs):
        image, matches = inputs["image"], inputs["matches"]
        if isinstance(matches, dict) and matches.get("model") == "pattern_matches_v1":
            if not isinstance(matches.get("items"), list) or matches["count"] != len(
                matches["items"]
            ):
                raise ValueError("Pattern match count differs from its items")
            matches = matches["items"]
        if not isinstance(matches, list) or len(matches) > 64:
            raise ValueError("Pick selection requires up to 64 pattern or template match records")
        p = self.parameters
        rectangle = np.asarray(p["footprint"], np.float64)
        if (
            rectangle.shape != (4,)
            or not np.isfinite(rectangle).all()
            or np.max(np.abs(rectangle)) > 10_000
            or np.any(rectangle[2:] <= 0)
        ):
            raise ValueError("Footprint needs finite [x, y, width, height] and positive dimensions")
        margin = finite(p["clearance"], "Clearance", 0)
        if margin > 10_000:
            raise ValueError("Clearance must be at most 10000 local pixels")
        score_min = finite(p["min_score"], "Minimum score", 0)
        tolerance = finite(p["angle_tolerance"], "Angle tolerance", 0)
        if score_min > 1 or tolerance > 180:
            raise ValueError("Minimum score must be at most 1 and angle tolerance at most 180")
        target = finite(p["target_angle"], "Target angle") % 360
        period = int(p["angle_period"])
        direction = np.asarray(p["downstream_direction"], np.float64)
        if direction.shape != (2,) or not np.isfinite(direction).all() or not np.any(direction):
            raise ValueError("Downstream direction must be a finite nonzero [dx,dy] vector")
        direction /= np.max(np.abs(direction))
        direction /= np.linalg.norm(direction)
        maximum_occupied = integer(
            p["maximum_occupied_pixels"], "Maximum occupied pixels", 0, 2**31 - 1
        )
        occupancy = selection_mask({"image": image, "mask": inputs.get("occupancy")}, image)
        if p["require_occupancy"] and occupancy is None:
            raise ValueError("Pick selection requires an occupancy mask")
        region = selection_mask({"image": image, "region": inputs.get("region")}, image)
        candidates, footprints, clearances = [], [], []
        for index, match in enumerate(matches):
            pose = pose_for_image(match.get("pose"), image)
            score = finite(match.get("score"), "Match score", 0)
            if score > 1:
                raise ValueError("Pick match scores must be between 0 and 1")
            matrix = np.asarray(pose["source_to_target"])
            footprint = projected_rectangle(rectangle, matrix)
            outer = projected_rectangle(rectangle, matrix, margin)
            angle_error = abs((pose["angle_degrees"] - target + period / 2) % period - period / 2)
            candidates.append(
                {
                    "index": index,
                    "pose": pose,
                    "score": score,
                    "angle_error": angle_error,
                    "downstream_position": float(direction @ pose["origin"]),
                    "footprint": footprint.tolist(),
                    "clearance": outer.tolist(),
                }
            )
            footprints.append(footprint)
            clearances.append(outer)

        overlay = overlay_image(image) if p["draw_overlay"] else NO_VALUE
        height, width = image.shape[:2]
        eligible = []
        for index, (candidate, outer) in enumerate(zip(candidates, clearances, strict=True)):
            reasons = []
            if candidate["score"] < score_min:
                reasons.append("score_below_minimum")
            if candidate["angle_error"] > tolerance:
                reasons.append("angle_out_of_range")
            if not contains(region, candidate["pose"]["origin"]):
                reasons.append("outside_pick_region")
            in_frame = bool(np.all(outer >= 0) and np.all(outer <= [width - 1, height - 1]))
            if not in_frame:
                reasons.append("clearance_outside_image")
            blockers = [
                other
                for other, footprint in enumerate(footprints)
                if other != index and abs(cv2.intersectConvexConvex(outer, footprint)[0]) > 1e-6
            ]
            if blockers:
                reasons.append("neighbor_in_clearance")
            occupied = (
                ring_occupancy(occupancy, outer, footprints[index])
                if occupancy is not None and in_frame
                else None
            )
            if occupied is not None and occupied > maximum_occupied:
                reasons.append("occupancy_in_clearance")
            candidate.update(reasons=reasons, blocking_indices=blockers, occupied_pixels=occupied)
            if not reasons:
                eligible.append(candidate)
            if overlay is not NO_VALUE:
                color = (60, 80, 240) if reasons else (70, 220, 140)
                cv2.polylines(overlay, [np.rint(outer).astype(np.int32)], True, color, 1)
        rank = (
            ("downstream_position", "score")
            if p["ranking"] == "downstream"
            else ("score", "downstream_position")
        )
        selected = max(
            eligible,
            key=lambda item: (item[rank[0]], item[rank[1]], -item["index"]),
            default=NO_VALUE,
        )
        found = selected is not NO_VALUE
        if found and overlay is not NO_VALUE:
            outline = np.rint(clearances[selected["index"]]).astype(np.int32)
            cv2.polylines(overlay, [outline], True, (0, 255, 255), 2)
        return Result(
            {
                "pose": selected["pose"] if found else NO_VALUE,
                "selected": selected,
                "candidates": candidates,
                "count": len(eligible),
                "overlay": overlay,
            },
            graphics={"pick clearance": overlay} if overlay is not NO_VALUE else {},
            passed=found,
            score=selected["score"] if found else 0.0,
            message=f"{len(eligible)} of {len(candidates)} candidates eligible",
            fail_reasons=[] if found else ["no_pick_candidate"],
        )
