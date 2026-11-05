"""Drawable result objects and the image frames they belong to."""

from dataclasses import dataclass

import cv2
import numpy as np

RESULT_GRAPHICS = "result_graphics"


@dataclass(frozen=True)
class GraphicObject:
    kind: str
    points: tuple[tuple[float, float], ...]
    color: tuple[int, int, int]
    thickness: int = 2
    closed: bool = False
    text: str = ""


def graphic_object(kind, points, color, **options):
    coordinates = tuple(tuple(map(float, point)) for point in points)
    return GraphicObject(kind, coordinates, color, **options)


def graphic_box(x, y, width, height, color):
    right, bottom = x + width - 1, y + height - 1
    return graphic_object(
        "polyline", ((x, y), (right, y), (right, bottom), (x, bottom)), color, closed=True
    )


def graphic_circle(x, y, radius, color):
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    points = np.column_stack((x + radius * np.cos(angles), y + radius * np.sin(angles)))
    return graphic_object("polyline", points, color, closed=True, thickness=1)


@dataclass(frozen=True)
class GraphicDrawing:
    objects: tuple[GraphicObject, ...]
    source: str = "image"


@dataclass(frozen=True)
class GraphicLayer:
    name: str
    objects: tuple[GraphicObject, ...]
    payload: dict


@dataclass(frozen=True)
class GraphicsBundle:
    layers: tuple[GraphicLayer, ...]

    def describe(self):
        return {
            "model": "result_graphics_v1",
            "layers": [
                {"name": layer.name, "objects": len(layer.objects)} for layer in self.layers
            ],
        }


def graphics_bundle(block, result):
    layers = [
        GraphicLayer(name, drawing.objects, block.input_payloads.get(drawing.source, {}))
        for name, drawing in result.drawings.items()
    ]
    if block.type_id == "core.toolblock" and block.generate_graphics:
        for child in block.graph.blocks.values():
            if child.state in ("inactive", "skipped", "error") or child.last_result is None:
                continue
            for layer in graphics_bundle(child, child.last_result).layers:
                name = f"{child.name} [{child.id[:8]}] / {layer.name}"
                layers.append(GraphicLayer(name, layer.objects, layer.payload))
    return GraphicsBundle(tuple(layers))


def layer_transform(layer, context):
    source_frame, target_frame = layer.payload.get("spatial"), context.get("spatial")
    if not source_frame or not target_frame:
        return None
    root = source_frame.get("root_id")
    if root != target_frame.get("root_id"):
        return None
    if source_frame["root_size"] != target_frame["root_size"]:
        return None
    source_matrix, target_matrix = (np.asarray(f["to_root"]) for f in (source_frame, target_frame))
    if root is None and (
        source_frame["image_size"] != target_frame["image_size"]
        or not np.allclose(source_matrix, target_matrix)
    ):
        return None
    return np.linalg.solve(target_matrix, source_matrix)


def draw_object(image, graphic, transform):
    points = np.asarray(graphic.points, np.float64)
    if not len(points):
        return
    mapped = cv2.perspectiveTransform(points[:, None, :], transform).reshape(-1, 2)
    if not np.isfinite(mapped).all() or np.any(np.abs(mapped) > 1_000_000):
        return
    pixels = np.rint(mapped).astype(np.int32)
    color = graphic.color
    if image.dtype == np.uint16:
        color = tuple(value * 257 for value in color)
    elif np.issubdtype(image.dtype, np.floating):
        color = tuple(value / 255 for value in color)
    if graphic.kind == "polyline":
        cv2.polylines(image, [pixels], graphic.closed, color, graphic.thickness)
    elif graphic.kind == "marker":
        cv2.drawMarker(image, tuple(pixels[0]), color, cv2.MARKER_CROSS, 12, 1)
    elif graphic.kind == "text":
        cv2.putText(
            image, graphic.text, tuple(pixels[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
        )
