"""Bounded, thread-safe result thumbnails; full image histories are never retained."""

import copy
import threading
from collections import OrderedDict
from pathlib import Path

import cv2

from .images import preview_rgb
from .tickets import ticket_key


def encode_thumbnail(image):
    rgb = preview_rgb(image)
    height, width = rgb.shape[:2]
    scale = min(160 / width, 100 / height, 1.0)
    rgb = cv2.resize(rgb, (max(1, round(width * scale)), max(1, round(height * scale))))
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return encoded.tobytes() if ok else b""


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

    def add(self, ticket, path, image=None, *, image_file=None):
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
                "image_file": copy.deepcopy(image_file),
                "state": "pending",
                "reasons": [],
            }
            self.rows[key] = row
            self.bytes += len(thumbnail)
            while len(self.rows) > self.capacity or self.bytes > self.max_bytes:
                _, removed = self.rows.popitem(last=False)
                self.bytes -= len(removed["thumbnail"])
                self.dropped += 1
            self.revision += 1

    def update_results(self, consumer):
        with self.lock:
            for row in self.rows.values():
                verdict = consumer.verdict(row["ticket"])
                if verdict and (row["state"], row["reasons"]) != verdict:
                    row["state"], row["reasons"] = verdict
                    self.revision += 1

    def capture(self, graph):
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
                    self.add(ticket, path, packet.value, image_file=image_file)
                if block.state == "error" and block._source_ticket:
                    self.add(block._source_ticket, selected or "Unreadable image")
            if hasattr(block, "graph"):
                self.capture(block.graph)
        self.update_results(graph.ticket_results)

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(list(self.rows.values()))
