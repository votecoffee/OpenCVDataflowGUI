"""Bounded, thread-safe result thumbnails; full image histories are never retained."""

import copy
import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from .graphics_layers import (
    GraphicDrawing,
    GraphicLayer,
    GraphicObject,
    draw_object,
    layer_transform,
)
from .images import preview_rgb
from .tickets import ticket_key


def _encode_png(bgr):
    ok, encoded = cv2.imencode(".png", bgr)
    return encoded.tobytes() if ok else b""


def encode_thumbnail(image):
    rgb = preview_rgb(image)
    height, width = rgb.shape[:2]
    scale = min(160 / width, 100 / height, 1.0)
    rgb = cv2.resize(rgb, (max(1, round(width * scale)), max(1, round(height * scale))))
    return _encode_png(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def annotate_filmstrip_image(bgr, frame, layers):
    height, width = bgr.shape[:2]
    sx, sy = np.array([width, height]) / frame["image_size"]
    resize = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
    for layer in layers:
        try:
            transform = layer_transform(layer, {"spatial": frame})
            if transform is not None:
                transform = resize @ transform
                for graphic in layer.objects:
                    draw_object(bgr, graphic, transform)
        except (cv2.error, TypeError, ValueError, OverflowError, np.linalg.LinAlgError):
            continue
    return bgr


class FilmstripCapture:
    """Collect display graphics for each activation, including buffered tickets."""

    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        self.bytes = 0
        self.lock = threading.Lock()
        self.layers = {}

    def record(self, block, result):
        if not block.capture_graphics or not isinstance(result.drawings, dict):
            return
        for name, drawing in result.drawings.items():
            if not isinstance(name, str) or not isinstance(drawing, GraphicDrawing):
                continue
            if not isinstance(drawing.source, str) or not drawing.objects:
                continue
            payload = block.input_payloads.get(drawing.source, {})
            ticket, frame = payload.get("ticket"), payload.get("spatial")
            if not ticket or not frame:
                continue
            key = ticket_key(ticket)
            label = (getattr(block, "_event_path", block.id), name)
            # Charge compact geometry as well as PNG bytes to the history budget.
            cost = 256
            for obj in drawing.objects:
                if (
                    not isinstance(obj, GraphicObject)
                    or not isinstance(obj.points, tuple)
                    or not isinstance(obj.text, str)
                ):
                    cost = self.max_bytes + 1
                    break
                cost += 128 + 80 * len(obj.points) + 4 * len(obj.text)
                if cost > self.max_bytes:
                    break
            with self.lock:
                bucket = self.layers.get(key, {})
                previous = bucket.get(label, (None, 0))[1]
                if self.bytes - previous + cost <= self.max_bytes:
                    bucket[label] = (GraphicLayer(name, drawing.objects, {"spatial": frame}), cost)
                    self.layers[key] = bucket
                    self.bytes += cost - previous

    def record_snapshot(self, block):
        if block.state in ("pass", "fail", "waiting") and block.last_result is not None:
            self.record(block, block.last_result)
            if hasattr(block, "graph"):
                for child in block.graph.blocks.values():
                    self.record_snapshot(child)


class Filmstrip:
    def __init__(self, capacity=200, max_bytes=8 * 1024 * 1024):
        if capacity < 1 or max_bytes < 1:
            raise ValueError("Filmstrip limits must be positive")
        self.capacity, self.max_bytes = capacity, max_bytes
        self.lock = threading.RLock()
        self.rows = OrderedDict()
        self.revision = 0
        self.dropped = 0
        self.bytes = 0

    def clear(self):
        with self.lock:
            self.rows.clear()
            self.bytes = self.dropped = 0
            self.revision += 1

    def add(self, ticket, path, image=None, *, image_file=None, frame=None):
        key = ticket_key(ticket)
        with self.lock:
            if key in self.rows:
                return
        thumbnail = encode_thumbnail(image) if image is not None else b""
        with self.lock:
            if key in self.rows:
                return
            row = {
                "ticket": copy.deepcopy(ticket),
                "filename": Path(path).name,
                "path": str(path),
                "thumbnail": thumbnail,
                "source_thumbnail": thumbnail,
                "image_file": copy.deepcopy(image_file),
                "frame": copy.deepcopy(frame),
                "layers": {},
                "image_revision": 0,
                "state": "pending",
                "reasons": [],
            }
            self.rows[key] = row
            self.bytes += len(thumbnail)
            self._trim()
            self.revision += 1

    @staticmethod
    def _row_bytes(row):
        size = len(row["source_thumbnail"])
        if row["layers"]:
            size += len(row["thumbnail"]) + sum(cost for _, cost in row["layers"].values())
        return size

    def _trim(self):
        while len(self.rows) > self.capacity or self.bytes > self.max_bytes:
            _, removed = self.rows.popitem(last=False)
            self.bytes -= self._row_bytes(removed)
            self.dropped += 1

    def _apply_graphics(self, graphics):
        with self.lock:
            for key, updates in graphics.layers.items():
                row = self.rows.get(key)
                if row is None or not row["frame"] or not row["source_thumbnail"]:
                    continue
                layers = row["layers"].copy()
                for label, (layer, cost) in updates.items():
                    try:
                        if layer_transform(layer, {"spatial": row["frame"]}) is not None:
                            layers[label] = (layer, cost)
                    except np.linalg.LinAlgError:
                        continue
                if layers == row["layers"]:
                    continue
                encoded = np.frombuffer(row["source_thumbnail"], np.uint8)
                bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                objects = [layers[label][0] for label in sorted(layers)]
                annotated = _encode_png(annotate_filmstrip_image(bgr, row["frame"], objects))
                size = len(row["source_thumbnail"]) + len(annotated)
                size += sum(cost for _, cost in layers.values())
                if not annotated or size > self.max_bytes:
                    continue
                previous = self._row_bytes(row)
                row.update(layers=layers, thumbnail=annotated)
                row["image_revision"] += 1
                self.bytes += size - previous
                self._trim()
                self.revision += 1

    def update_results(self, consumer):
        with self.lock:
            for row in self.rows.values():
                verdict = consumer.verdict(row["ticket"])
                if verdict and (row["state"], row["reasons"]) != verdict:
                    row["state"], row["reasons"] = verdict
                    self.revision += 1

    def _capture_sources(self, graph):
        for block in graph.blocks.values():
            if getattr(block, "file_source", False):
                selected = block._sequence.get("selected")
                packets = block.current_emissions.get("image", [])
                for packet in packets:
                    ticket = packet.payload.get("ticket")
                    if not ticket:
                        continue
                    page = packet.payload.get("tiff_page")
                    path = selected or (page["path"] if page else "Source image")
                    image_file = None
                    if block.type_id == "input.file" and selected:
                        settings = block.parameters | block.last_recipe_parameters
                        image_file = {
                            "path": str(Path(selected).absolute()),
                            "metadata_mode": settings["metadata_mode"],
                            "color_format": settings["color_format"],
                            "pixel_depth": settings["pixel_depth"],
                        }
                    self.add(
                        ticket, path, packet.value, image_file=image_file,
                        frame=packet.payload.get("spatial"),
                    )
                if block.state == "error" and block._source_ticket:
                    self.add(block._source_ticket, selected or "Unreadable image")
            if hasattr(block, "graph"):
                self._capture_sources(block.graph)

    def capture(self, graph, graphics=None):
        self._capture_sources(graph)
        if graphics is not None:
            self._apply_graphics(graphics)
        self.update_results(graph.ticket_results)

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(list(self.rows.values()))
