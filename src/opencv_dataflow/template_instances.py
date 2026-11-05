"""Rank repeated fixed-scale template detections and emit pose lists."""

import cv2
import numpy as np

from .core import Port, Result
from .features import overlay_image
from .graphics_layers import GraphicDrawing, graphic_box
from .payloads import NO_VALUE
from .poses import POSE_HELP, POSE_OFFSETS, local_offset, make_pose
from .processing import MatchTemplate, finite, integer
from .result_graphics import ACCEPTED, REJECTED, review_label, review_legend


class FindTemplateInstances(MatchTemplate):
    type_id, title = "inspect.template_instances", "Find template instances"
    outputs = (
        Port("poses", "data"),
        Port("matches", "data"),
        Port("count", "number"),
        Port("response"),
        Port("overlay", required=False),
    )
    defaults = {
        **MatchTemplate.defaults,
        "maximum_matches": 25,
        "minimum_matches": 1,
        "maximum_iou": 0.25,
        "draw_overlay": True,
        **POSE_OFFSETS,
    }
    parameter_choices = {"method": ("ccoeff_normed", "ccorr_normed", "sqdiff_normed")}
    parameter_help = {
        **MatchTemplate.parameter_help,
        "maximum_iou": "Suppress candidates that overlap an accepted match above this IoU.",
        **POSE_HELP,
    }

    def process(self, inputs):
        maximum = integer(self.parameters["maximum_matches"], "Maximum matches", 1, 64)
        minimum = integer(self.parameters["minimum_matches"], "Minimum matches", 0, maximum)
        iou = finite(self.parameters["maximum_iou"], "Maximum IoU", 0)
        if iou > 1:
            raise ValueError("Maximum IoU must be at most 1")
        template, response, quality, allowed, method, threshold = self.response_maps(inputs)
        candidates = np.clip(1 - quality if method == "sqdiff_normed" else quality, 0, 1)
        # Restrict ranking to local peaks within the legal search domain.
        if allowed is not None:
            candidates[allowed == 0] = -1
        peaks = cv2.compare(
            candidates, cv2.dilate(candidates, np.ones((3, 3), np.uint8)), cv2.CMP_EQ
        )
        candidates[peaks == 0] = -1
        source = inputs["image"]
        size = [source.shape[1], source.shape[0]]
        payload = self.input_payloads.get("image", {})
        offset = local_offset(*(self.parameters[key] for key in POSE_OFFSETS))
        overlay = overlay_image(source) if self.parameters["draw_overlay"] else NO_VALUE
        height, width = template.shape
        matches, poses, rejected = [], [], []
        review = min(maximum, 32) if self.generate_graphics and overlay is not NO_VALUE else 0
        for _ in range(maximum + review):
            _, score, _, (x, y) = cv2.minMaxLoc(candidates)
            if score < 0:
                break
            if score >= threshold and len(matches) < maximum:
                top_left = self.refined_top_left(response, allowed, method, x, y)
                cx, cy = top_left if top_left is not None else (x, y)
                center = [cx + (width - 1) / 2, cy + (height - 1) / 2]
                pose_matrix = np.eye(3)
                pose_matrix[:2, 2] = center
                pose = make_pose(pose_matrix @ offset, size, payload)
                matches.append(
                    {
                        "bounds": [x, y, width, height],
                        **({"subpixel_top_left": top_left} if top_left is not None else {}),
                        "center": center,
                        "score": float(score),
                        "pose": pose,
                    }
                )
                poses.append(pose)
                if overlay is not NO_VALUE:
                    cv2.rectangle(
                        overlay, (x, y), (x + width - 1, y + height - 1), (70, 220, 140), 2
                    )
            elif score > 0 and len(rejected) < review:
                rejected.append((x, y, score, "score" if score < threshold else "limit"))
            else:
                break
            x0, x1 = max(0, x - width + 1), min(candidates.shape[1], x + width)
            y0, y1 = max(0, y - height + 1), min(candidates.shape[0], y + height)
            intersection = (width - np.abs(np.arange(x0, x1) - x))[None, :] * (
                height - np.abs(np.arange(y0, y1) - y)
            )[:, None]
            overlap = intersection / (2 * width * height - intersection)
            candidates[y0:y1, x0:x1][overlap > iou] = -1
            candidates[y, x] = -1
        count = len(matches)
        graphics = {}
        drawings = {}
        if self.generate_graphics and overlay is not NO_VALUE:
            graphic = overlay.copy()
            for x, y, score, reason in rejected:
                cv2.rectangle(graphic, (x, y), (x + width - 1, y + height - 1), REJECTED, 2)
                review_label(graphic, f"{score:.2f} {reason}", (x, y + height // 2), REJECTED)
            for match in matches:
                x, y, _, _ = match["bounds"]
                cv2.rectangle(graphic, (x, y), (x + width - 1, y + height - 1), ACCEPTED, 2)
                review_label(graphic, f"{match['score']:.2f}", (x, y + height // 2), ACCEPTED)
            review_legend(graphic, count, len(rejected))
            graphics["template instances"] = graphic
            objects = [graphic_box(x, y, width, height, REJECTED) for x, y, _, _ in rejected]
            objects.extend(graphic_box(*match["bounds"], ACCEPTED) for match in matches)
            drawings["template instances"] = GraphicDrawing(tuple(objects))
        return Result(
            {
                "poses": poses,
                "matches": matches,
                "count": count,
                "response": response,
                "overlay": overlay,
            },
            graphics=graphics,
            drawings=drawings,
            passed=count >= minimum,
            score=matches[0]["score"] if matches else 0.0,
            message=f"Found {count} template instances (minimum {minimum})",
            fail_reasons=[] if count >= minimum else ["template_count_below_minimum"],
        )
