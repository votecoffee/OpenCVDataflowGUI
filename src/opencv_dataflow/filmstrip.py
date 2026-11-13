"""Bounded, thread-safe result thumbnails; full image histories are never retained."""

import copy
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .graphics_layers import (
    GraphicDrawing,
    GraphicLayer,
    GraphicObject,
    draw_object,
    drawing_payload,
    layer_transform,
)
from .images import preview_rgb
from .tickets import ticket_key


def _encode_png(bgr):
    ok, encoded = cv2.imencode(".png", bgr)
    return encoded.tobytes() if ok else b""


def _resize_preview(pixels, max_width, max_height):
    height, width = pixels.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return pixels
    return cv2.resize(pixels, (max(1, round(width * scale)), max(1, round(height * scale))))


def encode_thumbnail(image):
    rgb = _resize_preview(preview_rgb(image), 160, 100)
    return _encode_png(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


@dataclass(frozen=True)
class _FilmstripOverride:
    ticket: dict
    rank: tuple[int, str]
    thumbnail: bytes
    detail: bytes

    @property
    def size(self):
        return len(self.thumbnail) + len(self.detail)


def _encode_override(ticket, rank, image, max_bytes):
    rgb = preview_rgb(image)
    small = _resize_preview(rgb, 160, 100)
    thumbnail = _encode_png(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
    if not thumbnail or len(thumbnail) > max_bytes:
        return None
    detail = b""
    if small is not rgb:
        # Keep one bounded zoom image, with a compressed fallback for busy scenes.
        detail_limit = min(512 * 1024, max_bytes - len(thumbnail), max_bytes // 8)
        if detail_limit > 0:
            larger = cv2.cvtColor(_resize_preview(rgb, 1280, 1280), cv2.COLOR_RGB2BGR)
            detail = _encode_png(larger)
            if not detail or len(detail) > detail_limit:
                ok, encoded = cv2.imencode(".jpg", larger, [cv2.IMWRITE_JPEG_QUALITY, 90])
                detail = encoded.tobytes() if ok and encoded.nbytes <= detail_limit else b""
    return _FilmstripOverride(ticket, rank, thumbnail, detail)


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
    """Collect display graphics and selected images, including buffered tickets."""

    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        self.bytes = 0
        self.lock = threading.Lock()
        self.layers = {}
        self.overrides = OrderedDict()
        self.override_bytes = 0

    def _record_override(self, block, packet):
        ticket = packet.payload.get("ticket")
        if ticket is None:
            return
        key = ticket_key(ticket)
        priority = packet.payload["outcome"]["data"]["filmstrip_priority"]
        rank = (priority, getattr(block, "_event_path", block.id))
        try:
            override = _encode_override(ticket, rank, packet.value, self.max_bytes)
        except (cv2.error, TypeError, ValueError, OverflowError):
            return
        if override is None:
            return
        with self.lock:
            previous = self.overrides.get(key)
            if previous is not None:
                if rank < previous.rank:
                    return
                self.override_bytes -= previous.size
            self.overrides[key] = override
            self.overrides.move_to_end(key)
            self.override_bytes += override.size
            while self.override_bytes > self.max_bytes:
                _, removed = self.overrides.popitem(last=False)
                self.override_bytes -= removed.size

    def record(self, block, result):
        if output := getattr(block, "filmstrip_output", None):
            for packet in result.emissions.get(output, []):
                self._record_override(block, packet)
        if not block.capture_graphics or not isinstance(result.drawings, dict):
            return
        for name, drawing in result.drawings.items():
            if not isinstance(name, str) or not isinstance(drawing, GraphicDrawing):
                continue
            if not isinstance(drawing.source, str) or not drawing.objects:
                continue
            payload = drawing_payload(block, result, drawing)
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
                if obj.text_style is not None:
                    cost += 128 + 4 * len(obj.text_style.font_file)
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
                "override": None,
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
        if row["override"] is not None:
            return row["override"].size
        size = len(row["source_thumbnail"])
        if row["layers"]:
            size += len(row["thumbnail"]) + sum(cost for _, cost in row["layers"].values())
        return size

    def _trim(self):
        while len(self.rows) > self.capacity or self.bytes > self.max_bytes:
            _, removed = self.rows.popitem(last=False)
            self.bytes -= self._row_bytes(removed)
            self.dropped += 1

    def _apply_overrides(self, graphics):
        with self.lock:
            for key, override in graphics.overrides.items():
                if key not in self.rows:
                    self.add(override.ticket, "Result image")
                row = self.rows[key]
                previous = row["override"]
                if previous is not None and (override.rank < previous.rank or override == previous):
                    continue
                self.bytes += override.size - self._row_bytes(row)
                row.update(
                    override=override, thumbnail=override.thumbnail, source_thumbnail=b"",
                    image_file=None, frame=None, layers={},
                )
                row["image_revision"] += 1
                self._trim()
                self.revision += 1

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
            self._apply_overrides(graphics)
            self._apply_graphics(graphics)
        self.update_results(graph.ticket_results)

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(list(self.rows.values()))
