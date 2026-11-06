"""Bound OpenCV Voronoi facets to the reference image."""

import math

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import ValuePacket, require_same_frame
from .result_graphics import ACCEPTED
from .triangulation import SITE_INPUTS, read_sites


def _clip_facet(facet, half):
    # Clip in double precision because outer facet vertices can be far from the image.
    polygon = list(facet.astype(np.float64))
    for axis, limit, sign in (
        (0, -half[0], -1),
        (0, half[0], 1),
        (1, -half[1], -1),
        (1, half[1], 1),
    ):
        if not polygon:
            break
        clipped = []
        start = polygon[-1]
        distance = sign * (start[axis] - limit)
        for end in polygon:
            next_distance = sign * (end[axis] - limit)
            if distance * next_distance < 0:
                crossing = start + (end - start) * (distance / (distance - next_distance))
                crossing[axis] = limit
                clipped.append(crossing)
            if next_distance <= 0:
                clipped.append(end)
            start, distance = end, next_distance
        polygon = clipped
    return np.asarray(list(dict.fromkeys(map(tuple, polygon))))


def _cells(sites, size):
    origin = (np.asarray(size) - 1) / 2
    scale = math.ldexp(1.0, 10 - math.frexp(max(size))[1])
    half = np.asarray(size) * scale / 2
    local = ((sites - origin) * scale).astype(np.float32)
    subdivision = cv2.Subdiv2D((-4096, -4096, 8192, 8192))
    ids = [subdivision.insert(tuple(map(float, site))) for site in local]
    if len(set(ids)) != len(sites):
        raise ValidationError("Distinct sites coincide at OpenCV coordinate precision")
    facets, centers = subdivision.getVoronoiFacetList(ids)
    if len(facets) != len(sites) or not np.array_equal(centers, local):
        raise ValidationError("OpenCV did not return a facet for each site")
    polygons, areas = [], []
    for facet in facets:
        if not np.isfinite(facet).all():
            raise ValidationError("OpenCV returned an invalid Voronoi facet")
        polygon = _clip_facet(facet, half)
        area = cv2.contourArea(polygon.astype(np.float32)) if len(polygon) >= 3 else 0
        if area <= 0:
            raise ValidationError("OpenCV did not produce a positive-area cell for every site")
        polygons.append(polygon / scale + origin)
        areas.append(area / scale**2)
    if not math.isclose(math.fsum(areas), math.prod(size), rel_tol=1e-6):
        raise ValidationError("Voronoi cells do not cover the reference image")
    return polygons, areas


class VoronoiCells(Block):
    type_id, title = "inspect.voronoi_cells", "Voronoi cells"
    category, block_type, subtype = "Inspection", "Inspection", "Measurements"
    inputs = (Port("image"), *SITE_INPUTS)
    outputs = (
        Port("cells", "data"),
        Port("cell_count", "number"),
        Port("selected_area", "number"),
        Port("mask"),
    )
    defaults = {"cell_index": 0}
    parameter_ranges = {"cell_index": NumericRange(0, 9_999, whole=True)}
    parameter_help = {
        "cell_index": "Zero-based index in the unique site order, used for the mask and highlight."
    }
    process_inputs_readonly = True

    def process(self, inputs):
        image = inputs["image"]
        size = list(image.shape[:2][::-1])
        sites, first, _, payload = read_sites(self, inputs, minimum=1, image_size=size)
        if payload.get("spatial", {}).get("image_size", size) != size:
            raise ValidationError("Site dimensions must match the reference image")
        require_same_frame(
            (image, ValuePacket(sites, payload)),
            message="Sites and the reference image must use the same coordinate frame",
        )
        if np.any(sites < -0.5) or np.any(sites > np.asarray(size) - 0.5):
            raise ValidationError("Every site must be inside the reference image")
        selected = int(self.parameters["cell_index"])
        if selected >= len(sites):
            raise ValidationError(f"Cell index must be between 0 and {len(sites) - 1}")

        polygons, areas = _cells(sites, size)
        cells = [
            {"site": site, "source_index": index, "area": area, "vertices": polygon.tolist()}
            for site, index, area, polygon in zip(sites.tolist(), first.tolist(), areas, polygons)
        ]
        mask = np.zeros(image.shape[:2], np.uint8)
        shift = 8
        vertices = np.rint(polygons[selected] * (1 << shift)).astype(np.int32)
        cv2.fillConvexPoly(mask, vertices, 255, shift=shift)

        drawings = {}
        if self.generate_graphics:
            outlines = tuple(
                graphic_object("polyline", polygon, (255, 180, 0), closed=True, thickness=1)
                for polygon in polygons
            )
            highlight = graphic_object("polyline", polygons[selected], ACCEPTED, closed=True)
            drawings = {
                "Voronoi cells": GraphicDrawing(outlines),
                "Selected cell": GraphicDrawing((highlight,)),
            }
        return Result(
            {
                "cells": cells,
                "cell_count": len(cells),
                "selected_area": areas[selected],
                "mask": mask,
            },
            drawings=drawings,
            message=f"{len(cells)} cells, selected area {areas[selected]:.1f} square pixels",
        )
