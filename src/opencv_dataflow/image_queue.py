"""FIFO acquisition boundary with bounded storage and one downstream frame in flight."""

import copy
import threading
from dataclasses import dataclass, field

import numpy as np

from .control_flow import FlowBlock, byte_size
from .core import Port, Result, validate_value
from .payloads import NO_VALUE, ValuePacket, prepare_inputs
from .processing import finite, integer


@dataclass
class QueuedCapture:
    frame: ValuePacket
    snapshots: dict = field(default_factory=dict)

    def size(self):
        # Conservatively count the frame again when upstream snapshots also reference it.
        def snapshots_size(snapshots):
            total = 0
            for snapshot in snapshots.values():
                total += byte_size(snapshot.get("last_recipe_parameters", {}))
                result = snapshot["last_result"]
                if result is not None:
                    total += byte_size(snapshot["last_inputs"]) + byte_size(result.outputs)
                    total += byte_size(result.payloads) + byte_size(snapshot["last_graphics"])
                total += snapshots_size(snapshot.get("_captured_graph", {}))
            return total

        return byte_size(self.frame) + snapshots_size(self.snapshots)


class ImageQueue(FlowBlock):
    type_id, title, subtype = "flow.image_queue", "Queue input images", "Acquisition queues"
    inputs = (Port("image", required=False),)
    outputs = (
        Port("image"),
        Port("queued", "number"),
        Port("busy", "number"),
        Port("dropped", "number"),
        Port("stats", "data"),
    )
    defaults = {
        "mode": "buffer_only",
        "capacity": 32,
        "max_bytes_mb": 128.0,
        "overflow": "block",
        "capture_interval_ms": 0.0,
    }
    parameter_choices = {
        "mode": ("buffer_only", "background_acquisition"),
        "overflow": ("block", "drop_oldest", "drop_newest", "error"),
    }
    parameter_help = {
        "mode": "Buffer bursts, or capture in the background during Continuous/CLI repeats.",
        "capacity": "Maximum waiting frames, excluding the one being inspected.",
        "overflow": "Block pauses background capture. Synchronous overflow is an error.",
        "capture_interval_ms": "Pause between acquisition iterations; zero uses source pacing.",
    }
    queue_boundary = True
    preserve_image_sets = False
    flow_primary = ("image",)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.condition = threading.Condition()
        self.lease = None
        self.pending_events = []
        self.reset_runtime()

    def reset_runtime(self):
        with self.condition:
            self.runtime = {
                "pending": [],
                "bytes": 0,
                "enqueued": 0,
                "released": 0,
                "completed": 0,
                "failed": 0,
                "dropped": 0,
                "rejected": 0,
                "high_water": 0,
                "producer_error": "",
            }
            self.lease = None
            self.use_lease = False
            self.pending_events = []
            self.preloaded_events = []
            self.condition.notify_all()

    def snapshot_runtime(self):
        with self.condition:
            return copy.deepcopy(self.runtime)

    def restore_runtime(self, snapshot):
        with self.condition:
            self.runtime = copy.deepcopy(snapshot)
            self.lease = None
            self.use_lease = False
            self.pending_events, self.preloaded_events = [], []
            self.condition.notify_all()

    def copy_runtime_from(self, other):
        self.restore_runtime(other.snapshot_runtime())

    def close(self):
        self.reset_runtime()

    def limits(self):
        capacity = integer(self.parameters["capacity"], "Queue capacity", 1, 4096)
        megabytes = finite(self.parameters["max_bytes_mb"], "Queue MB", 0)
        if not 0 < megabytes <= 4096:
            raise ValueError("Queue storage must be in (0,4096] MiB")
        finite(self.parameters["capture_interval_ms"], "Capture interval", 0)
        self.validate_parameters()
        return capacity, megabytes * 1024 * 1024

    def normalize(self, packet):
        values, payloads = prepare_inputs(self, {"image": packet})
        validate_value(Port("image"), values["image"])
        if not isinstance(values["image"], np.ndarray):
            raise ValueError("Select one image/pyramid level before queueing")
        return ValuePacket(values["image"], payloads["image"])

    def enqueue(self, capture, cancelled=lambda: False, background=False):
        if not isinstance(capture, QueuedCapture):
            capture = QueuedCapture(self.normalize(capture))
        else:
            capture.frame = self.normalize(capture.frame)
        capacity, maximum = self.limits()
        size = capture.size()
        if size > maximum:
            raise ValueError("One capture exceeds the queue byte limit")
        policy = self.parameters["overflow"]
        with self.condition:
            if self.lease is not None:
                self.report_event(
                    "queue_busy",
                    {"reason": "frame arrived during downstream work"},
                    {"image": capture.frame},
                )
            waited = False
            while (
                len(self.runtime["pending"]) >= capacity or self.runtime["bytes"] + size > maximum
            ):
                if cancelled():
                    return False
                if policy == "drop_newest":
                    self.runtime["dropped"] += 1
                    self.report_event("dropped", {"policy": policy}, {"image": capture.frame})
                    return False
                if policy == "drop_oldest":
                    old, old_size = self.runtime["pending"].pop(0)
                    self.runtime["bytes"] -= old_size
                    self.runtime["dropped"] += 1
                    self.report_event("dropped", {"policy": policy}, {"image": old.frame})
                    del old
                elif policy == "block" and background:
                    if not waited:
                        self.report_event(
                            "queue_wait", {"reason": "queue capacity"}, {"image": capture.frame}
                        )
                        waited = True
                    self.condition.wait(0.05)
                else:
                    self.runtime["rejected"] += 1
                    raise ValueError(
                        "Image queue full; captured frames retained. Drain it or change limits."
                    )
            if cancelled():
                return False
            if waited:
                self.report_event("queue_resume", values={"image": capture.frame})
            self.runtime["pending"].append((copy.deepcopy(capture), size))
            if not background:
                self.runtime["producer_error"] = ""
            self.runtime["bytes"] += size
            self.runtime["enqueued"] += 1
            self.runtime["high_water"] = max(
                self.runtime["high_water"], len(self.runtime["pending"])
            )
            self.condition.notify_all()
            return True

    def wait_for_capture_space(self, cancelled):
        capacity, maximum = self.limits()
        with self.condition:
            waited = False
            while self.parameters["overflow"] == "block" and (
                len(self.runtime["pending"]) >= capacity or self.runtime["bytes"] >= maximum
            ):
                if cancelled():
                    return False
                if not waited:
                    self.report_event("queue_wait", {"reason": "capacity gate before capture"})
                    waited = True
                self.condition.wait(0.05)
            if waited:
                self.report_event("queue_resume", {"reason": "capacity gate released"})
            return not cancelled()

    def take(self, cancelled=lambda: False, wait_for_frame=False):
        with self.condition:
            if self.lease is not None:
                raise ValueError("An input image is already in flight")
            while not self.runtime["pending"]:
                if cancelled():
                    return None
                if self.runtime["producer_error"]:
                    raise RuntimeError(self.runtime["producer_error"])
                if not wait_for_frame:
                    return None
                self.condition.wait(0.05)
            if cancelled():
                return None
            self.lease, size = self.runtime["pending"].pop(0)
            self.runtime["bytes"] -= size
            self.runtime["released"] += 1
            self.condition.notify_all()
            return self.lease

    def complete(self, failed=False):
        with self.condition:
            if self.lease is not None:
                self.runtime["completed"] += 1
                self.runtime["failed"] += int(failed)
            self.lease = None
            self.use_lease = False
            self.condition.notify_all()

    def producer_failed(self, message, exception=None):
        if exception is not None:
            self.logger.exception(message, exc_info=exception)
        else:
            self.logger.error(message)
        with self.condition:
            self.runtime["producer_error"] = message
            self.condition.notify_all()

    def stats(self):
        with self.condition:
            return {key: value for key, value in self.runtime.items() if key != "pending"} | {
                "queued": len(self.runtime["pending"]),
                "busy": int(self.lease is not None),
            }

    def update_metrics(self):
        stats = self.stats()
        if self.last_result is not None:
            self.last_result.outputs.update(
                queued=stats["queued"], busy=stats["busy"], dropped=stats["dropped"], stats=stats
            )
            self.last_result.message = (
                f"{stats['queued']} queued · {stats['busy']} busy · {stats['dropped']} dropped"
            )

    def process(self, inputs):
        capacity, maximum = self.limits()
        pending_events = [*self.preloaded_events, *self.pending_events]
        self.preloaded_events, self.pending_events = [], []
        if self.lease is None:
            if "image" in inputs:
                pending_events.insert(0, ValuePacket(inputs["image"], self.input_payloads["image"]))
            captures = [
                event if isinstance(event, QueuedCapture) else QueuedCapture(self.normalize(event))
                for event in pending_events
            ]
            with self.condition:
                planned = [*self.runtime["pending"], *((c, c.size()) for c in captures)]
                if any(size > maximum for _, size in planned):
                    self.runtime["rejected"] += len(captures)
                    raise ValueError("One capture exceeds the queue byte limit")
                if self.parameters["overflow"] in ("block", "error") and (
                    len(planned[1:]) > capacity or sum(size for _, size in planned[1:]) > maximum
                ):
                    self.runtime["rejected"] += len(captures)
                    raise ValueError(
                        "Image queue full; waiting frames retained and incoming burst rejected"
                    )
                capture = self.take(self._cancelled)
                for incoming in captures:
                    self.enqueue(incoming, self._cancelled)
                    if capture is None:
                        capture = self.take(self._cancelled)
        else:
            if not self.use_lease:
                raise ValueError(
                    "Previous image is still in flight; complete it before releasing another"
                )
            capture = self.lease
        if capture is not None:
            self.last_inputs["image"] = copy.deepcopy(capture.frame.value)
            self.input_payloads["image"] = capture.frame.payload
        stats = self.stats()
        return Result(
            {
                "image": capture.frame.value if capture else NO_VALUE,
                "queued": stats["queued"],
                "busy": stats["busy"],
                "dropped": stats["dropped"],
                "stats": stats,
            },
            payloads={"image": capture.frame.payload} if capture else {},
            message=f"{stats['queued']} queued · {stats['busy']} busy",
        )
