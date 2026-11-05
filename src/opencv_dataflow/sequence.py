"""Copyable temporal state, stream contexts, readiness and bounded memory."""

import copy
import hashlib
import json
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from .control_flow import byte_size
from .core import Block, Port, Result
from .images import image_format
from .payloads import NO_VALUE
from .processing import finite, image_value, integer
from .selection import SELECTION_INPUTS, selection_mask

SEQUENCE_DEFAULTS = {
    "warmup": 2,
    "max_streams": 4,
    "max_bytes_mb": 128.0,
    "overflow": "error",
    "context_change": "reset",
    "max_gap_ms": 0.0,
}
SEQUENCE_CHOICES = {"overflow": ("error", "evict_oldest"), "context_change": ("reset", "separate")}
SEQUENCE_HELP = {
    "warmup": "Minimum eligible activations before primary data outputs are emitted.",
    "context_change": "Reset a source on recipe/grid/settings change, or cache separate contexts.",
    "max_gap_ms": "Reset after this ingress gap; zero keeps state across pauses.",
    "max_bytes_mb": "Stored array/metadata budget; intermediates and GUI copies use extra memory.",
}


@dataclass
class SequenceStep:
    outputs: dict
    ready: bool = True
    learned: bool = True
    details: dict = field(default_factory=dict)
    graphics: dict = field(default_factory=dict)


class SequenceBlock(Block):
    """Plugins override step(). State must be deepcopy-able and fit the configured budget."""

    category, block_type, subtype = "Sequence", "Sequence", "Temporal processing"
    sequence_memory = True
    requires_serial_execution = True
    inputs = (
        *SELECTION_INPUTS,
        Port("reset", "data", required=False),
        Port("stream", "data", required=False),
    )
    defaults, parameter_choices, parameter_help = SEQUENCE_DEFAULTS, SEQUENCE_CHOICES, SEQUENCE_HELP
    data_outputs = ()
    status_outputs = (Port("ready", "data"), Port("activations", "number"), Port("status", "data"))

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.outputs = (*cls.data_outputs, *cls.status_outputs)
        cls.flow_primary = tuple(p.name for p in cls.data_outputs)

    def __init__(self, **kwargs):
        self.outputs = (*self.data_outputs, *self.status_outputs)
        self.flow_primary = tuple(p.name for p in self.data_outputs)
        self._memory_lock = threading.RLock()
        super().__init__(**kwargs)
        self.reset_runtime()

    def reset_runtime(self):
        with self._memory_lock:
            self.memory = {
                "streams": OrderedDict(),
                "epoch": uuid.uuid4().hex,
                "generation": 0,
                "evicted": 0,
                "errors": 0,
            }

    def snapshot_runtime(self):
        with self._memory_lock:
            return copy.deepcopy(self.memory)

    def copy_runtime_from(self, other):
        snapshot = other.snapshot_runtime()
        with self._memory_lock:
            self.memory = snapshot

    def close(self):
        self.reset_runtime()

    def now(self):
        return time.monotonic()

    def extra_context(self, inputs):
        return None

    def estimate_state_bytes(self, image):
        return image.nbytes

    def state_bytes(self, states):
        return byte_size(states)

    def step(self, inputs, state, mask):
        raise NotImplementedError

    def process(self, inputs):
        image = inputs["image"]
        image_value(image)
        mask = selection_mask(inputs, image)
        p = self.parameters
        required = integer(p["warmup"], "Warmup", 1, 10000)
        max_streams = integer(p["max_streams"], "Maximum streams", 1, 64)
        maximum = finite(p["max_bytes_mb"], "Sequence memory MB", 0) * 1024 * 1024
        if not 0 < maximum <= 4096 * 1024 * 1024:
            raise ValueError("Sequence memory must be in (0,4096] MiB")
        gap = finite(p["max_gap_ms"], "Maximum gap", 0) / 1000
        reset = inputs.get("reset", False)
        if type(reset) is not bool:
            raise ValueError("Sequence reset input must be boolean")
        stream = inputs.get(
            "stream", self.input_payloads["image"].get("ticket", {}).get("source", "default")
        )
        if type(stream) not in (str, int) or not str(stream) or len(str(stream)) > 256:
            raise ValueError("Sequence stream must be nonempty text or an integer")
        payload = self.input_payloads["image"]
        context = {
            k: copy.deepcopy(payload[k])
            for k in ("spatial", "pyramid", "recipe_ref")
            if k in payload
        }
        signature = {
            "stream": stream,
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "encoding": image_format(image),
            "context": context,
            "parameters": p,
            "extra": self.extra_context(inputs),
        }
        key = hashlib.sha256(
            json.dumps(signature, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        if max(image.nbytes, self.estimate_state_bytes(image)) > maximum:
            raise ValueError("Image/model would exceed the sequence memory byte limit")
        logical = json.dumps(stream)
        now = self.now()
        with self._memory_lock:
            states = self.memory["streams"]
            if reset:
                for old in [k for k, s in states.items() if s["logical"] == logical]:
                    del states[old]
            if p["context_change"] == "reset":
                for old in [k for k, s in states.items() if s["logical"] == logical and k != key]:
                    del states[old]
            state = states.get(key)
            if state is not None and gap and now - state["last_time"] > gap:
                del states[key]
                state = None
            if state is None:
                if image.nbytes > maximum:
                    raise ValueError("One image exceeds the sequence memory budget")
                if len(states) >= max_streams:
                    if p["overflow"] == "error":
                        raise ValueError("Sequence stream limit exceeded")
                    states.popitem(last=False)
                    self.memory["evicted"] += 1
                self.memory["generation"] += 1
                state = {
                    "logical": logical,
                    "generation": f"{self.memory['epoch']}:{self.memory['generation']}",
                    "activations": 0,
                    "learning": 0,
                    "last_time": now,
                    "last_ticket": None,
                }
                states[key] = state
            previous = copy.deepcopy(state["last_ticket"])
            try:
                step = self.step(inputs, state, mask)
                if not isinstance(step, SequenceStep) or set(step.outputs) != set(
                    self.flow_primary
                ):
                    raise ValueError("Sequence step must declare every primary output")
                state["activations"] += 1
                state["learning"] += int(step.learned)
                state["last_time"] = now
                state["last_ticket"] = copy.deepcopy(payload.get("ticket"))
                states.move_to_end(key)
                while self.state_bytes(states) > maximum:
                    if p["overflow"] == "error" or len(states) == 1:
                        raise ValueError("Sequence memory byte limit exceeded")
                    states.popitem(last=False)
                    self.memory["evicted"] += 1
                needed = 0 if state.get("seeded") else required
                ready = bool(step.ready and state["learning"] >= needed)
                status = {
                    "model": "sequence_status_v1",
                    "ready": ready,
                    "activations": state["activations"],
                    "learning_activations": state["learning"],
                    "required_activations": needed,
                    "stream": stream,
                    "context": key,
                    "generation": state["generation"],
                    "previous_ticket": previous,
                    "current_ticket": state["last_ticket"],
                    "streams": len(states),
                    "evicted": self.memory["evicted"],
                    "seeded": bool(state.get("seeded")),
                    **step.details,
                }
                output_payload = copy.deepcopy(payload)
                output_payload["sequence"] = status
                outputs = {
                    name: value if ready else NO_VALUE for name, value in step.outputs.items()
                }
                outputs.update(ready=ready, activations=state["activations"], status=status)
                for value in outputs.values():
                    if hasattr(value, "metadata") and not isinstance(value, np.ndarray):
                        value.metadata = copy.deepcopy(output_payload)
                phase = (
                    "Ready" if ready else "Warming" if state["learning"] < needed else "Not ready"
                )
                return Result(
                    outputs,
                    payloads={
                        name: copy.deepcopy(output_payload)
                        for name in outputs
                        if outputs[name] is not NO_VALUE
                    },
                    graphics=step.graphics,
                    event_data={"sequence": status},
                    message=f"{phase} · {state['learning']}/{needed} activations · {stream}",
                )
            except Exception:
                states.pop(key, None)
                self.memory["errors"] += 1
                raise
