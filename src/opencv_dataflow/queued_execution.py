"""Background acquisition on a private upstream graph, one consumer graph at a time."""

import copy
import threading

import numpy as np

from .core import Graph
from .image_queue import ImageQueue, QueuedCapture
from .outcomes import publish_result, rejected_inputs
from .payloads import NO_VALUE
from .tickets import ticket_key
from .toolblocks import ToolBlock

SNAPSHOT_FIELDS = (
    "last_inputs",
    "input_payloads",
    "last_result",
    "last_output_images",
    "last_graphics",
    "state",
    "error",
    "elapsed_ms",
    "current_emissions",
    "input_rejected",
    "last_rejection",
)


def background_queue(graph):
    for block in graph.blocks.values():
        if isinstance(block, ToolBlock) or getattr(block, "owns_workers", False):
            if background_queue(block.graph) is not None:
                raise ValueError(
                    "Background queues belong in the root graph; nested queues use buffer_only"
                )
    queues = [
        b
        for b in graph.blocks.values()
        if isinstance(b, ImageQueue) and b.parameters["mode"] == "background_acquisition"
    ]
    if len(queues) > 1:
        raise ValueError(
            "Use one background acquisition queue per root graph; other queues can use buffer_only"
        )
    return queues[0] if queues else None


def ancestors(graph, key):
    selected = set()
    changed = True
    while changed:
        changed = False
        for connection in graph.connections:
            if connection.target in selected | {key} and connection.source not in selected:
                selected.add(connection.source)
                changed = True
    return selected


def check_upstream(block):
    if (
        getattr(block, "queue_boundary", False)
        or getattr(block, "owns_workers", False)
        or (not isinstance(block, ToolBlock) and getattr(block, "accept_events", False))
    ):
        raise ValueError(
            "Place background acquisition before buffered/parallel inspections, or use buffer_only"
        )
    if isinstance(block, ToolBlock):
        for child in block.graph.blocks.values():
            check_upstream(child)


def same_ticket(a, b):
    if "ticket" not in a or "ticket" not in b:
        return False
    return (ticket_key(a["ticket"]), a["ticket"]["source"]) == (
        ticket_key(b["ticket"]),
        b["ticket"]["source"],
    )


def snapshots_for_frame(graph, frame, keys=None):
    snapshots = {}
    for key, block in graph.blocks.items():
        if keys is not None and key not in keys:
            continue
        snapshot = {name: getattr(block, name) for name in SNAPSHOT_FIELDS}
        snapshot["last_output_images"] = dict(block.last_output_images)
        if isinstance(block, ToolBlock):
            snapshot["_captured_graph"] = snapshots_for_frame(block.graph, frame)
        result = copy.copy(block.last_result)
        snapshot["last_result"] = result
        if result is not None:
            result.outputs, result.payloads = dict(result.outputs), dict(result.payloads)
            result.emissions = dict(block.current_emissions)
            for port in block.outputs:
                packets = result.emissions.get(port.name, [])
                if len(packets) > 1:
                    packets = [p for p in packets if same_ticket(p.payload, frame.payload)]
                    if len(packets) != 1:
                        raise ValueError(
                            "Cannot correlate upstream burst outputs with the queued frame"
                        )
                    snapshot["last_inputs"], snapshot["input_payloads"] = {}, {}
                    snapshot["last_graphics"], result.graphics = {}, {}
                result.emissions[port.name] = packets
                if packets:
                    result.outputs[port.name] = packets[0].value
                    result.payloads[port.name] = packets[0].payload
                    if isinstance(packets[0].value, np.ndarray):
                        snapshot["last_output_images"][port.name] = packets[0].value
                else:
                    result.outputs[port.name] = NO_VALUE
            snapshots[key] = copy.deepcopy(snapshot)
    return snapshots


class QueuedExecution:
    """Own the acquisition thread and join it before returning control to callers."""

    def __init__(self, graph, registry, cancelled=lambda: False):
        self.graph, self.registry, self.cancelled = graph, registry, cancelled
        graph.bind_ticket_results()
        self.queue = background_queue(graph)
        if self.queue is None:
            raise ValueError("No background acquisition queue configured")
        self.queue.limits()
        self.upstream = ancestors(graph, self.queue.id)
        self.edge = next(
            (c for c in graph.connections if c.target == self.queue.id and c.input == "image"), None
        )
        if self.edge is None:
            raise ValueError("Background queue requires a connected image source")
        for key in self.upstream:
            check_upstream(graph.blocks[key])
        self.producer = Graph.from_dict(graph.to_dict(), registry, project_dir=graph.project_dir)
        self.producer.blocks = {k: b for k, b in self.producer.blocks.items() if k in self.upstream}
        self.producer.connections = [
            c
            for c in self.producer.connections
            if c.source in self.upstream and c.target in self.upstream
        ]
        self.producer.bind_ticket_results(graph.ticket_results)
        self.stop = threading.Event()
        self.thread = None

    def stopped(self):
        return self.stop.is_set() or self.cancelled()

    def _capture(self):
        try:
            while not self.stopped():
                if not self.queue.wait_for_capture_space(self.stopped):
                    break
                self.producer.run(self.stopped)
                if self.stopped():
                    break
                errors = [
                    f"{b.name}: {b.error}"
                    for b in self.producer.blocks.values()
                    if b.state in ("error", "skipped")
                ]
                if errors:
                    raise RuntimeError("; ".join(errors))
                packets = self.producer.blocks[self.edge.source].current_emissions.get(
                    self.edge.output, []
                )
                for packet in packets:
                    frame = self.queue.normalize(packet)
                    if rejected_inputs(self.queue, {"image": frame}):
                        publish_result(
                            self.queue,
                            {"image": frame},
                            kind="input_rejected",
                            message="Rejected failed acquisition",
                        )
                        continue
                    capture = QueuedCapture(frame, snapshots_for_frame(self.producer, frame))
                    self.queue.enqueue(capture, self.stopped, background=True)
                delay = self.queue.parameters["capture_interval_ms"] / 1000
                if self.stop.wait(max(delay, 0.01 if not packets else 0)):
                    break
        except Exception as exc:
            self.queue.producer_failed(f"Acquisition failed: {type(exc).__name__}: {exc}")
        finally:
            self.producer.close()

    def start(self):
        if self.thread is not None:
            raise ValueError("Acquisition already started")
        with self.queue.condition:
            self.queue.runtime["producer_error"] = ""
        self.thread = threading.Thread(
            target=self._capture, name="opencv-image-acquisition", daemon=False
        )
        self.thread.start()

    def run_one(self):
        capture = self.queue.take(self.cancelled, wait_for_frame=True)
        if capture is None:
            return False
        self.queue.use_lease = True
        try:
            self.graph.run(self.cancelled, provided=capture.snapshots)
        except Exception:
            self.queue.complete(failed=True)
            self.queue.update_metrics()
            raise
        return True

    def close(self):
        self.stop.set()
        with self.queue.condition:
            self.queue.condition.notify_all()
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        else:
            self.producer.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.close()
