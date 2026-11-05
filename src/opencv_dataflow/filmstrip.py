"""Bounded, thread-safe result thumbnails; full image histories are never retained."""

import copy
import threading
from collections import OrderedDict
from pathlib import Path

import cv2

from .images import preview_rgb
from .tickets import ticket_key


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

    def add(self, ticket, path, image=None):
        key = ticket_key(ticket)
        with self.lock:
            if key in self.rows:
                return
        thumbnail = b""
        if image is not None:
            rgb = preview_rgb(image)
            scale = min(160 / rgb.shape[1], 100 / rgb.shape[0], 1.0)
            rgb = cv2.resize(
                rgb, (max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale)))
            )
            ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if ok:
                thumbnail = encoded.tobytes()
        with self.lock:
            if key in self.rows:
                return
            row = {
                "ticket": copy.deepcopy(ticket),
                "filename": Path(path).name,
                "path": str(path),
                "thumbnail": thumbnail,
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
                packets = block.current_emissions.get("image", [])
                for packet in packets:
                    ticket = packet.payload.get("ticket")
                    if ticket:
                        self.add(ticket, block._sequence["selected"], packet.value)
                if block.state == "error" and block._source_ticket:
                    self.add(
                        block._source_ticket, block._sequence["selected"] or "Unreadable image"
                    )
            if hasattr(block, "graph"):
                self.capture(block.graph)
        self.update_results(graph.ticket_results)

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(list(self.rows.values()))
