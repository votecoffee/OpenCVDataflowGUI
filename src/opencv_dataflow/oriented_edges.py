"""Oriented gradient-band edge detection with measured continuity and thickness."""

import math
from bisect import bisect_right

import cv2
import numpy as np

from .core import Block, Port, Result
from .features import overlay_image
from .objects import collection, line_item
from .parameters import BORDERS
from .processing import finite, gray_image, integer
from .result_graphics import ACCEPTED, REJECTED, review_legend
from .selection import SELECTION_INPUTS, selection_mask


def angular_error(a, b):
    return abs((a - b + 90) % 180 - 90)


def join_components(parts, gap, normal_distance):
    parents = list(range(len(parts)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    if gap:
        ordered = sorted(range(len(parts)), key=lambda i: parts[i]["start"])
        starts = [parts[i]["start"] for i in ordered]
        comparisons = 0
        for index, part in enumerate(parts):
            a = bisect_right(starts, part["end"] + 1e-6)
            b = bisect_right(starts, part["end"] + gap + 1)
            for other in ordered[a:b]:
                comparisons += 1
                if comparisons > 1_000_000:
                    raise ValueError(
                        "Too many gap-join comparisons; reduce ROI or strengthen edge threshold"
                    )
                if abs(parts[other]["normal"] - part["normal"]) <= normal_distance:
                    parents[root(other)] = root(index)
    groups = {}
    for index in range(len(parts)):
        groups.setdefault(root(index), []).append(index)
    return list(groups.values())


def interval_coverage(intervals):
    ordered = sorted((a - 0.5, b + 0.5) for a, b in intervals)
    start, end = ordered[0]
    total, lower = 0.0, start
    for a, b in ordered[1:]:
        if a <= end:
            end = max(end, b)
        else:
            total += end - start
            start, end = a, b
    total += end - start
    return min(1.0, total / (end - lower))


class FindOrientedEdges(Block):
    type_id, title, category = "inspect.oriented_edges", "Find oriented edges", "Inspection"
    block_type, subtype = "Inspection", "Edges"
    inputs = SELECTION_INPUTS
    outputs = (
        Port("collection", "data"),
        Port("segments", "data"),
        Port("count", "number"),
        Port("mask"),
        Port("response"),
        Port("overlay"),
    )
    defaults = {
        "orientation": "vertical",
        "angle_degrees": 0.0,
        "angle_tolerance": 10.0,
        "polarity": "either",
        "blur_kernel": 3,
        "blur_sigma": 1.0,
        "gradient_aperture": "3",
        "border": "reflect",
        "response_threshold": 10.0,
        "strength_statistic": "mean",
        "min_strength": 10.0,
        "max_strength": 0.0,
        "min_thickness": 0.0,
        "max_thickness": 0.0,
        "min_length": 20.0,
        "min_continuous_length": 20.0,
        "min_intensity": 0.0,
        "max_intensity": 255.0,
        "max_gap": 0,
        "gap_normal_distance": 2.0,
        "min_coverage": 0.0,
        "min_contrast": 0.0,
        "max_contrast": 0.0,
        "contrast_offset": 2.0,
    }
    parameter_choices = {
        "orientation": ("vertical", "horizontal", "angled"),
        "polarity": ("either", "rising", "falling"),
        "gradient_aperture": ("3", "5", "7"),
        "border": tuple(BORDERS),
        "strength_statistic": ("mean", "median", "peak"),
    }
    parameter_help = {
        "angle_degrees": "Tangent degrees: 0 horizontal, 90 vertical; used for angled.",
        "polarity": "Along normal (tangent-90) mod 180: right for vertical, down for horizontal.",
        "min_thickness": "Gradient-band width normal to the fitted edge, not bright-stripe width.",
        "min_continuous_length": "Longest original connected run, independent of gap joining.",
        "response_threshold": "Pixel gradient threshold in grayscale intensity/pixel.",
        "min_intensity": "Mean original grayscale intensity over measured edge pixels (0..255).",
        "max_gap": "Join separated collinear runs; zero disables joining.",
        "contrast_offset": "Sample this far beyond half band width on each side for contrast.",
    }

    def process(self, inputs):
        image = gray_image(inputs["image"], (np.uint8,))
        if image.size > 64_000_000 or max(image.shape) > 16384:
            raise ValueError("Edge image exceeds 64M pixels / 16384 pixels per axis")
        p = self.parameters
        angle = {
            "vertical": 90.0,
            "horizontal": 0.0,
            "angled": finite(p["angle_degrees"], "Angle"),
        }[p["orientation"]] % 180
        tolerance = finite(p["angle_tolerance"], "Angle tolerance", 0)
        if not 0 < tolerance < 90:
            raise ValueError("Angle tolerance must be in (0,90)")
        response_threshold = finite(p["response_threshold"], "Response threshold", 0)
        (
            strength_min,
            strength_max,
            thick_min,
            thick_max,
            length_min,
            continuous_min,
            intensity_min,
            intensity_max,
            coverage_min,
            join_distance,
        ) = (
            finite(p[k], k, 0)
            for k in (
                "min_strength",
                "max_strength",
                "min_thickness",
                "max_thickness",
                "min_length",
                "min_continuous_length",
                "min_intensity",
                "max_intensity",
                "min_coverage",
                "gap_normal_distance",
            )
        )
        if (
            response_threshold <= 0
            or (strength_max and strength_max < strength_min)
            or (thick_max and thick_max < thick_min)
            or not intensity_min <= intensity_max <= 255
            or coverage_min > 1
        ):
            raise ValueError(
                "Invalid edge threshold, thickness, strength, coverage or intensity limits"
            )
        gap = integer(p["max_gap"], "Maximum gap", 0, 1000)
        contrast_min, contrast_max, contrast_offset = (
            finite(p[k], k, 0) for k in ("min_contrast", "max_contrast", "contrast_offset")
        )
        if contrast_max and contrast_max < contrast_min:
            raise ValueError("Maximum contrast must be at least minimum contrast")
        kernel = integer(p["blur_kernel"], "Blur kernel", 1, 63, odd=True)
        smooth = cv2.GaussianBlur(image, (kernel, kernel), finite(p["blur_sigma"], "Blur sigma", 0))
        aperture = int(p["gradient_aperture"])
        scale = 1 / 2.0 ** (2 * aperture - 3)
        gx = cv2.Sobel(
            smooth, cv2.CV_32F, 1, 0, ksize=aperture, scale=scale, borderType=BORDERS[p["border"]]
        )
        gy = cv2.Sobel(
            smooth, cv2.CV_32F, 0, 1, ksize=aperture, scale=scale, borderType=BORDERS[p["border"]]
        )
        normal_angle = math.radians((angle - 90) % 180)
        normal = np.array([math.cos(normal_angle), math.sin(normal_angle)])
        tangent = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
        signed = gx * normal[0] + gy * normal[1]
        response = np.abs(signed).astype(np.float32)
        magnitude = cv2.magnitude(gx, gy)
        gate = (response >= response_threshold) & (
            response >= magnitude * math.cos(math.radians(tolerance))
        )
        selection = selection_mask(inputs, image)
        if selection is not None:
            gate &= selection != 0
        signs = (
            (1, -1) if p["polarity"] == "either" else ((1,) if p["polarity"] == "rising" else (-1,))
        )
        items, segments, rejected = [], [], []
        intensity_float = image.astype(np.float32)
        accepted_mask = np.zeros_like(image)
        overlay = overlay_image(inputs["image"])
        for sign in signs:
            binary = (gate & (signed * sign > 0)).astype(np.uint8)
            total, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            if total > 10001:
                raise ValueError(
                    "Too many raw edge components; increase response threshold or restrict ROI"
                )
            parts = []
            for label in range(1, total):
                x, y, w, h, area = stats[label]
                if area < 2:
                    continue
                points = np.argwhere(labels[y : y + h, x : x + w] == label)[:, ::-1] + [x, y]
                along = points @ tangent
                parts.append(
                    {
                        "points": points,
                        "start": float(along.min()),
                        "end": float(along.max()),
                        "normal": float(np.mean(points @ normal)),
                    }
                )
            for group in join_components(parts, gap, join_distance):
                points = np.vstack([parts[i]["points"] for i in group])
                center = points.mean(axis=0)
                _, vectors = np.linalg.eigh(np.cov(points.T))
                direction = vectors[:, -1]
                fitted_angle = math.degrees(math.atan2(direction[1], direction[0])) % 180
                if angular_error(fitted_angle, angle) > tolerance:
                    continue
                direction = np.array(
                    [math.cos(math.radians(fitted_angle)), math.sin(math.radians(fitted_angle))]
                )
                across = np.array([-direction[1], direction[0]])
                t, n = (points - center) @ direction, (points - center) @ across
                length, thickness = float(np.ptp(t)), float(np.ptp(n) + 1)
                intervals = [((parts[i]["points"] - center) @ direction) for i in group]
                continuous = max(float(np.ptp(values)) for values in intervals)
                coverage = interval_coverage([(float(v.min()), float(v.max())) for v in intervals])
                values = response[points[:, 1], points[:, 0]]
                strength = float(
                    {"mean": np.mean, "median": np.median, "peak": np.max}[p["strength_statistic"]](
                        values
                    )
                )
                intensity = float(image[points[:, 1], points[:, 0]].mean())
                samples = (
                    center + np.linspace(float(t.min()), float(t.max()), 64)[:, None] * direction
                )
                a_samples, b_samples = (
                    samples + across * (thickness / 2 + contrast_offset),
                    samples - across * (thickness / 2 + contrast_offset),
                )
                valid = np.all((a_samples >= 0) & (b_samples >= 0), axis=1)
                valid &= (
                    (a_samples[:, 0] < image.shape[1])
                    & (b_samples[:, 0] < image.shape[1])
                    & (a_samples[:, 1] < image.shape[0])
                    & (b_samples[:, 1] < image.shape[0])
                )
                contrast = None
                if valid.any():

                    def sample(values):
                        return cv2.remap(
                            intensity_float,
                            values[valid, 0].astype(np.float32).reshape(-1, 1),
                            values[valid, 1].astype(np.float32).reshape(-1, 1),
                            cv2.INTER_LINEAR,
                        ).mean()

                    contrast = float(abs(sample(a_samples) - sample(b_samples)))
                if (
                    length < length_min
                    or continuous < continuous_min
                    or thickness < thick_min
                    or (thick_max and thickness > thick_max)
                    or strength < strength_min
                    or (strength_max and strength > strength_max)
                    or not intensity_min <= intensity <= intensity_max
                    or coverage < coverage_min
                    or (contrast_min and (contrast is None or contrast < contrast_min))
                    or (contrast_max and (contrast is None or contrast > contrast_max))
                ):
                    if self.generate_graphics and len(rejected) < 256:
                        rejected.append(
                            np.array([center + direction * t.min(), center + direction * t.max()])
                        )
                    continue
                endpoints = np.array([center + direction * t.min(), center + direction * t.max()])
                item = line_item(endpoints, len(items), width=thickness)
                item["metrics"].update(
                    thickness=thickness,
                    continuous_length=continuous,
                    strength=strength,
                    intensity=intensity,
                    coverage=coverage,
                    contrast=contrast,
                )
                item["polarity"] = "rising" if sign > 0 else "falling"
                item["normal_degrees"] = math.degrees(normal_angle)
                items.append(item)
                segments.append(endpoints.reshape(-1).tolist())
                accepted_mask[points[:, 1], points[:, 0]] = 255
                a, b = np.rint(endpoints).astype(int)
                cv2.line(overlay, tuple(a), tuple(b), ACCEPTED, 2)
        graphic = overlay
        if self.generate_graphics:
            graphic = overlay.copy()
            for endpoints in rejected:
                a, b = np.rint(endpoints).astype(int)
                cv2.line(graphic, tuple(a), tuple(b), REJECTED, 2)
            for segment in segments:
                a, b = np.rint(np.reshape(segment, (2, 2))).astype(int)
                cv2.line(graphic, tuple(a), tuple(b), ACCEPTED, 2)
            review_legend(graphic, len(items), len(rejected))
        return Result(
            {
                "collection": collection("lines", items, image, self.input_payloads["image"]),
                "segments": segments,
                "count": len(items),
                "mask": accepted_mask,
                "response": response,
                "overlay": overlay,
            },
            graphics={"oriented edges": graphic} if self.generate_graphics else {},
            message=f"{len(items)} oriented edges near {angle:g} degrees",
        )
