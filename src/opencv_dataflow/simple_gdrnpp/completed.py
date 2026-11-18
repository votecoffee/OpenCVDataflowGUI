"""Worker-owned evidence from one applied execution; retrieval never performs analysis."""

import hashlib
import json
from dataclasses import asdict

import numpy as np

SCHEMA = "simple_gdrnpp_completed/v1"
BINDING = "gdrnpp_completed_review"


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def report_digest(report):
    # The bridge appends its own transfer timing after the worker publishes.
    return hashlib.sha256(
        json_bytes({k: v for k, v in report.items() if k != "request_timing"})
    ).hexdigest()


def receipt(token, session, recipe, report):
    raw = recipe._source_bytes
    if raw is None:
        raise ValueError("Completed review requires the exact loaded recipe archive")
    return dict(
        schema=SCHEMA,
        token=token,
        session=session,
        archive_sha256=hashlib.sha256(raw).hexdigest(),
        content_sha256=recipe.content_sha256,
        report_sha256=report_digest(report),
        candidates=len(report["candidates"]),
    )


def transport_arrays(arrays):
    """Convert native integer indices only during explicit transfer, not execution."""
    for value in arrays.values():
        if (
            value.dtype.kind in "iu"
            and value.dtype.itemsize > 4
            and value.size
            and (value.min() < -(2**31) or value.max() >= 2**31)
        ):
            raise ValueError("Stored source indices exceed the lossless wire representation")
    return {
        key: np.asarray(value, np.int32)
        if value.dtype.kind in "iu" and value.dtype.itemsize > 4
        else value
        for key, value in arrays.items()
    }


class RetainedExecution:
    def __init__(self, identity, recipe, capture, report, evidence, scene, graphics):
        self.identity = identity
        self.recipe, self.capture, self.report = recipe, capture, report
        self.evidence, self.scene, self.graphics = evidence, scene, graphics

    def retrieve(self, operation, data):
        if data.get("identity") != self.identity:
            raise ValueError("Completed execution expired or belongs to another worker session")
        meta = dict(identity=self.identity)
        if operation == "capture_completed":
            from .setup import describe

            frame = self.capture.frame
            reference = self.recipe.reference.frame
            raw = self.recipe._source_bytes
            # A two-dimensional byte grid respects the wire's maximum dimension.
            width = min(len(raw), 1_000_000)
            packed = np.zeros(((len(raw) + width - 1) // width, width), np.uint8)
            packed.flat[: len(raw)] = np.frombuffer(raw, np.uint8)
            meta.update(
                report=self.report,
                setup=describe(self.recipe),
                archive_bytes=len(raw),
                capture=dict(
                    capture_id=self.capture.capture_id,
                    camera_frame=frame.frame,
                    model_frame=self.capture.model_frame,
                    detections=[asdict(d) for d in self.capture.detections],
                ),
                scene_available=bool(self.scene),
                graphics=list(self.graphics),
                scope="Exact consumed RGB-D and stored fit samples; no inference, render or refit",
            )
            arrays = dict(
                rgb=frame.rgb,
                depth=frame.depth,
                camera=frame.camera,
                mask=frame.mask,
                reference_rgb=reference.rgb,
                reference_depth=reference.depth,
                reference_camera=reference.camera,
                reference_mask=reference.mask,
                recipe_bytes=packed,
                **self.scene,
            )
        elif operation == "capture_completed_candidate":
            index = data.get("index")
            if type(index) is not int or not 0 <= index < len(self.report["candidates"]):
                raise ValueError("Unknown completed candidate")
            candidate = self.report["candidates"][index]
            arrays = self.evidence.get(candidate["id"], {})
            meta.update(index=index, candidate_id=candidate["id"], available=bool(arrays))
        elif operation == "capture_completed_graphics":
            arrays = self.graphics
        else:
            raise ValueError("Unknown completed retrieval")
        return meta, transport_arrays(arrays)
