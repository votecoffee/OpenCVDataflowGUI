"""Shared geometry for Hough peaks and their finite support spans."""

import math

import numpy as np

from .objects import line_item


def supported_lines(peaks, points, support_distance, image_size):
    # Native polar coordinates use float32. Leave a rounding margin at the bin edge.
    tolerance = support_distance + 4 * np.finfo(np.float32).eps * max(image_size)
    items, polar_lines = [], []
    for index, (votes, rho, theta) in enumerate(peaks):
        nx, ny = math.cos(theta), math.sin(theta)
        supported = points[np.abs(points[:, 0] * nx + points[:, 1] * ny - rho) <= tolerance]
        polar = {
            "votes": int(votes),
            "rho_pixels": float(rho),
            "normal_angle_degrees": math.degrees(theta),
            "support_count": len(supported),
        }
        polar_lines.append(polar)
        if not len(supported):
            continue
        tangent = (-ny, nx)
        positions = supported @ tangent
        endpoints = rho * np.array([nx, ny]) + np.outer([positions.min(), positions.max()], tangent)
        item = line_item(endpoints, index)
        item["metrics"]["strength"] = polar["votes"]
        item["geometry"]["polar"] = polar
        items.append(item)
    return items, polar_lines
