"""Isolated per-pose inspection jobs with bounded concurrency and aggregate outcomes."""

import copy
import hashlib
import json
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import cv2
import numpy as np

from .core import Block, Graph, Port, Result
from .features import overlay_image
from .objects import validate_collection
from .payloads import NO_VALUE, FlowGroup, ValuePacket, describe_value, metadata
from .poses import PoseFromFeature, pose_for_image
from .processing import finite, integer
from .tickets import DIRECT_CLOCK, ticket_key
from .toolblocks import Binding, ToolBlock


class RepeatedPartsDemo(Block):
    type_id, title, category = "input.parts_demo", "Repeated parts demo", "Input"
    block_type, subtype = "Input", "Synthetic"
    outputs = (Port("image"), Port("reference"), Port("info", "data"))
    defaults = {"count": 25, "columns": 5, "fail_every": 0}

    def process(self, inputs):
        count = integer(self.parameters["count"], "Part count", 1, 64)
        columns = integer(self.parameters["columns"], "Columns", 1, 16)
        fail_every = integer(self.parameters["fail_every"], "Fail every", 0, 64)
        reference = np.zeros((32, 32), np.uint8)
        cv2.line(reference, (5, 5), (5, 26), 255, 2)
        cv2.line(reference, (5, 26), (25, 26), 255, 2)
        cv2.rectangle(reference, (20, 5), (25, 10), 255, 1)
        image = np.zeros(
            (((count + columns - 1) // columns) * 56 + 40, columns * 56 + 40), np.uint8
        )
        failures = 0
        for index in range(count):
            x, y = 20 + (index % columns) * 56, 20 + (index // columns) * 56
            image[y : y + 32, x : x + 32] = reference
            if fail_every and (index + 1) % fail_every == 0:
                failures += 1
            else:
                cv2.circle(image, (x + 16, y + 16), 3, 255, -1)
        return Result(
            {
                "image": image,
                "reference": reference,
                "info": {
                    "count": count,
                    "expected_pass": count - failures,
                    "expected_fail": failures,
                },
            }
        )


def default_inspection(registry):
    graph = Graph()
    image = graph.add(
        registry.create("flow.split", block_id="part_image", parameters={"workers": 1})
    )
    region = graph.add(
        registry.create(
            "region.pose", block_id="part_region", parameters={"shape": "circle", "radius": 5}
        )
    )
    inspect = graph.add(
        registry.create(
            "inspect.blobs",
            block_id="part_blobs",
            parameters={
                "min_area": 3,
                "check_count": True,
                "expected_count": 1,
                "region_mode": "clip",
            },
        )
    )
    image.name, region.name, inspect.name = (
        "Part image",
        "Local inspection region",
        "One local blob",
    )
    image.position, region.position, inspect.position = [0, 0], [310, 0], [620, 0]
    graph.connect(image.id, "branch_1", region.id, "image")
    graph.connect(image.id, "branch_2", inspect.id, "image")
    graph.connect(region.id, "region", inspect.id, "region")
    tool = ToolBlock(
        graph,
        {"image": Binding(image.id, "value"), "pose": Binding(region.id, "pose")},
        {
            "data": Binding(inspect.id, "data"),
            "count": Binding(inspect.id, "count"),
            "overlay": Binding(inspect.id, "overlay"),
        },
    )
    tool.name = "Per-part inspection"
    return tool


def walk(graph):
    for block in graph.blocks.values():
        yield block
        if isinstance(block, ToolBlock):
            yield from walk(block.graph)


class ParallelInspect(Block):
    type_id, title, category = "flow.inspect_each", "Inspect each pose in parallel", "Control flow"
    block_type, subtype = "Control flow", "Parallel inspection"
    owns_workers = True
    inputs = (Port("image"), Port("features", "data"), Port("context", "data", required=False))
    outputs = (
        Port("results", "data"),
        Port("group", "any"),
        Port("images", "any"),
        Port("pass_count", "number"),
        Port("fail_count", "number"),
        Port("error_count", "number"),
        Port("complete", "data"),
        Port("overlay"),
    )
    defaults = {
        "workers": 4,
        "max_items": 64,
        "max_result_mb": 128.0,
        "max_input_mb": 256.0,
        "stop_after_passes": 0,
        "stop_after_failures": 0,
        "result_policy": "all_pass",
        "required_passes": 1,
        "allowed_failures": 0,
        "empty_policy": "fail",
        "image_terminal": "image",
        "pose_terminal": "pose",
        "context_terminal": "context",
        "image_output": "overlay",
        "preview_index": 0,
        "feature_angle": "fixed",
    }
    parameter_choices = {
        "result_policy": ("all_pass", "any_pass", "min_passes", "max_failures"),
        "empty_policy": ("fail", "pass"),
        "feature_angle": ("fixed", "feature"),
    }
    parameter_help = {
        "workers": "1..32 concurrent part inspections; each owns a fresh ToolBlock graph.",
        "max_items": "Reject larger collections before starting work (1..64).",
        "max_result_mb": "Retained per-part data/images budget; stop with an error if exceeded.",
        "max_input_mb": "Image bytes times workers ceiling; intermediates use extra memory.",
        "stop_after_passes": "Zero disables. Stop submitting after N completed passing parts.",
        "stop_after_failures": "Zero disables. Failures and errors count toward this limit.",
        "result_policy": "All-pass/max-failures need coverage; other policies allow partial work.",
        "image_terminal": "Public image input name in the inspection ToolBlock.",
        "pose_terminal": "Public pose input name in the inspection ToolBlock.",
        "context_terminal": "Optional shared data input in the inspection ToolBlock.",
        "image_output": "Public image output collected into the images group; empty disables it.",
        "preview_index": "Retain this part's internal steps; fall back to a completed part.",
    }

    def __init__(self, *, inspection=None, registry=None, **kwargs):
        if registry is None:
            from .blocks import builtin_registry

            registry = builtin_registry()
        self.registry = registry
        self.inspection = inspection if inspection is not None else default_inspection(registry)
        self.last_preview_index = None
        super().__init__(**kwargs)

    @property
    def graph(self):
        return self.inspection.graph

    def bind_registry(self, registry):
        self.registry = registry

    def validate_inspection(self, inspection=None):
        tool = inspection or self.inspection
        tool.validate_interface()
        p = self.parameters
        names = [p["image_terminal"], p["pose_terminal"], p["context_terminal"]]
        if len(set(names)) != 3 or any(not isinstance(n, str) or not n.strip() for n in names):
            raise ValueError(
                "Inspection image/pose/context terminal names must be distinct and nonempty"
            )
        for name, kinds in ((names[0], ("image", "any")), (names[1], ("data", "any"))):
            port = next((port for port in tool.inputs if port.name == name), None)
            if port is None or port.kind not in kinds:
                raise ValueError(f"Inspection needs a compatible public input named {name}")
        if any(port.required and port.name not in names for port in tool.inputs):
            raise ValueError("Map required inspection inputs to image, pose or context terminals")
        if any(b.type_id == self.type_id for b in walk(tool.graph)):
            raise ValueError(
                "Nested parallel per-pose blocks are not supported inside an inspection"
            )
        if any(getattr(b, "sequence_memory", False) for b in walk(tool.graph)):
            raise ValueError(
                "Sequence memory needs a stable stream outside fresh per-part inspection jobs"
            )

    def serialize_extra(self):
        return {"inspection": self.inspection.definition(include_recipes=False)}

    @classmethod
    def from_definition(cls, data, registry, depth=0, **kwargs):
        return cls(
            inspection=ToolBlock.from_definition(data, registry, depth), registry=registry, **kwargs
        )

    def clear_result(self, preserve_history=False):
        super().clear_result(preserve_history=preserve_history)
        if preserve_history and "reject_failed" in self.input_failure_policies.values():
            return
        self.inspection.graph.clear_results(preserve_history=preserve_history)
        self.last_preview_index = None

    def reset_runtime(self):
        self.inspection.reset_runtime()

    def copy_runtime_from(self, other):
        # Every part invocation starts a fresh graph; buffers never cross part boundaries.
        self.inspection.timing = copy.deepcopy(other.inspection.timing)
        self.graph.copy_timing_from(other.graph)

    def copy_results_from(self, other):
        super().copy_results_from(other)
        self.inspection.copy_results_from(other.inspection)
        self.last_preview_index = other.last_preview_index

    def close(self):
        self.inspection.close()

    def _poses(self, image, features, maximum):
        if isinstance(features, list):
            items = features
            poses = features
        elif isinstance(features, dict) and features.get("model") == "image_pose_v1":
            items, poses = [features], [features]
        elif isinstance(features, dict) and features.get("model") in (
            "pattern_matches_v1",
            "vision_collection_v1",
        ):
            items = features.get("items")
            poses = None
        else:
            raise ValueError(
                "Features must be pattern matches, a measured collection, or a pose list"
            )
        if not isinstance(items, list) or len(items) > maximum:
            raise ValueError(f"Inspection collection must contain at most {maximum} parts")
        if isinstance(features, dict) and features.get("model") == "vision_collection_v1":
            validate_collection(features)
        if isinstance(features, dict) and features.get("model") == "pattern_matches_v1":
            if features.get("count") != len(items):
                raise ValueError("Pattern match count differs from its items")
        converted = []
        for index, item in enumerate(items):
            if poses is not None:
                pose = pose_for_image(poses[index], image)
            else:
                converter = PoseFromFeature(
                    parameters={"index": index, "angle_mode": self.parameters["feature_angle"]}
                )
                converter.input_payloads = {"features": self.input_payloads["features"]}
                pose = converter.process({"image": image, "features": features}).outputs["pose"]
            converted.append((item.get("id", index) if isinstance(item, dict) else index, pose))
        return converted

    def process(self, inputs):
        self.validate_inspection()
        p, image = self.parameters, inputs["image"]
        image_ticket = self.input_payloads["image"].get("ticket")
        feature_ticket = self.input_payloads["features"].get("ticket")
        if (
            image_ticket
            and feature_ticket
            and ticket_key(image_ticket) != ticket_key(feature_ticket)
        ):
            raise ValueError("Image and feature tickets differ; synchronize them before inspection")
        workers = integer(p["workers"], "Workers", 1, 32)
        maximum = integer(p["max_items"], "Maximum parts", 1, 64)
        result_limit = finite(p["max_result_mb"], "Result budget", 0)
        input_limit = finite(p["max_input_mb"], "Input budget", 0)
        if not 0 < result_limit <= 4096 or not 0 < input_limit <= 4096:
            raise ValueError("Input/result budgets must be in (0,4096] MiB")
        pass_limit = integer(p["stop_after_passes"], "Pass stop threshold", 0, 64)
        fail_limit = integer(p["stop_after_failures"], "Failure stop threshold", 0, 64)
        required = integer(p["required_passes"], "Required passes", 1, 64)
        allowed = integer(p["allowed_failures"], "Allowed failures", 0, 64)
        preview = integer(p["preview_index"], "Preview index", 0, 63)
        poses = self._poses(image, inputs["features"], maximum)
        total = len(poses)
        if image.nbytes * min(workers, total) > input_limit * 1024 * 1024:
            raise ValueError(
                "Input image times workers exceeds budget; reduce image size or workers"
            )
        template = self.inspection.definition(include_recipes=False)
        context_name = p["context_terminal"]
        if (
            any(port.name == context_name and port.required for port in self.inspection.inputs)
            and "context" not in inputs
        ):
            raise ValueError("Inspection requires the context input")
        parent = self.input_payloads["image"].get("ticket") or getattr(
            self, "_clock", DIRECT_CLOCK
        ).issue(self.id)
        session = hashlib.sha256(json.dumps([parent, self.id], sort_keys=True).encode()).hexdigest()
        records, packets, pictures = [], {}, {}
        for index, (feature_id, pose) in enumerate(poses):
            records.append(
                {
                    "index": index,
                    "feature_id": feature_id,
                    "pose": pose,
                    "status": "not_started",
                    "passed": None,
                    "score": None,
                    "elapsed_ms": 0.0,
                    "outputs": {},
                    "error": "",
                }
            )
        stop = threading.Event()

        def cancelled():
            return stop.is_set() or self._cancelled()

        def job(index):
            started = time.perf_counter()
            record = copy.deepcopy(records[index])
            if cancelled():
                record["status"] = "cancelled"
                return index, record, None, None, None, time.perf_counter()
            tool = None
            picture = None
            ticket = {
                "model": "flow_ticket_v1",
                "session": session,
                "kind": "number",
                "value": index + 1,
                "sequence": index + 1,
                "source": parent["source"],
            }
            payload = {
                **metadata(image),
                "ticket": ticket,
                "inspection": {
                    "index": index,
                    "feature_id": record["feature_id"],
                    "parent_ticket": parent,
                },
            }
            try:
                tool = ToolBlock.from_definition(template, self.registry)
                tool._ticket_results = self._ticket_results
                tool._recipes = self._recipes
                tool._allow_recipe_activation = False
                tool.graph.bind_recipes(self._recipes, allow_activation=False)
                tool._event_path = f"{getattr(self, '_event_path', self.id)}/part-{index}"
                tool._activity = self._activity
                tool.run_profile = self.run_profile
                tool._job_graphics_mode = self._job_graphics_mode
                tool.graph._inherit_graphics_from(tool)
                tool.graph.set_run_profile(self.run_profile)
                tool._production_excluded = True
                tool._activity_path = tool._event_path
                logical = getattr(self, "_activity_logical_path", self.id)
                tool._activity_logical_path = f"{logical}/[inspection]"
                tool.graph.bind_activity(self._activity, tool._event_path, logical)
                tool.graph.bind_ticket_results(self._ticket_results, tool._event_path)
                tool.graph.bind_project_dir(self.project_dir)
                # Per-item graphs run serially; the outer pool owns part concurrency.
                tool.graph.set_branch_workers(1)
                for block in walk(tool.graph):
                    block._production_excluded = True
                    if isinstance(block, ToolBlock):
                        block.graph.set_branch_workers(1)
                values = {
                    p["image_terminal"]: ValuePacket(image, payload),
                    p["pose_terminal"]: ValuePacket(record["pose"], payload),
                }
                if "context" in inputs and any(port.name == context_name for port in tool.inputs):
                    values[context_name] = ValuePacket(inputs["context"], payload)
                result = tool.execute(values, cancelled)
                children = list(walk(tool.graph))
                if any(b.state == "skipped" for b in children) and cancelled():
                    record["status"] = "cancelled"
                elif tool.state == "error":
                    record.update(status="error", passed=False, error=tool.error)
                elif any(b.state == "waiting" for b in children) or tool.state == "waiting":
                    record.update(
                        status="error",
                        passed=False,
                        error="Inspection has unfinished buffered work",
                    )
                else:
                    record.update(
                        status=tool.state,
                        passed=tool.state == "pass",
                        score=result.score,
                        outputs={
                            key: describe_value(value) for key, value in result.outputs.items()
                        },
                    )
                    value = result.outputs.get(p["image_output"])
                    if isinstance(value, np.ndarray):
                        picture = ValuePacket(
                            value, result.payloads.get(p["image_output"], payload)
                        )
            except Exception as exc:
                (tool or self).logger.exception(
                    "Part inspection failed",
                    exc_info=exc,
                    values={"image": ValuePacket(image, payload)},
                )
                record.update(status="error", passed=False, error=f"{type(exc).__name__}: {exc}")
            finally:
                record["elapsed_ms"] = (time.perf_counter() - started) * 1000
                if tool is not None:
                    try:
                        tool.close()
                    except Exception as exc:
                        tool.logger.exception(
                            "Part inspection cleanup failed",
                            exc_info=exc,
                            values={"image": ValuePacket(image, payload)},
                        )
                        record.update(
                            status="error",
                            passed=False,
                            error=f"Inspection cleanup failed: {type(exc).__name__}: {exc}",
                        )
            return index, record, ValuePacket(record, payload), picture, tool, time.perf_counter()

        active, next_index = {}, 0
        reason = "completed" if total else "no_items"
        counts = {"pass": 0, "fail": 0, "error": 0}
        result_bytes = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            while active or next_index < total:
                if self._cancelled() and not stop.is_set():
                    reason = "cancelled"
                    stop.set()
                while not stop.is_set() and next_index < total and len(active) < workers:
                    future = executor.submit(job, next_index)
                    active[future] = next_index
                    next_index += 1
                if not active:
                    break
                ready, _ = wait(active, timeout=0.05, return_when=FIRST_COMPLETED)
                completed = []
                for future in ready:
                    index = active.pop(future)
                    if not future.cancelled():
                        completed.append(future.result())
                for index, record, packet, picture, tool, _ in sorted(
                    completed, key=lambda row: row[-1]
                ):
                    from .control_flow import byte_size

                    size = byte_size(record) + (byte_size(picture) if picture is not None else 0)
                    if (
                        result_bytes + size > result_limit * 1024 * 1024
                        and record["status"] in counts
                    ):
                        record.update(
                            status="error",
                            passed=False,
                            outputs={},
                            error="Retained result budget exceeded",
                        )
                        picture = None
                        if not stop.is_set():
                            reason = "result_limit"
                            stop.set()
                    else:
                        result_bytes += size
                    records[index] = record
                    timing_history = [
                        (b, copy.deepcopy(b.timing), b.elapsed_ms)
                        for b in (self.inspection, *walk(self.graph))
                    ]
                    if record["status"] in counts:
                        counts[record["status"]] += 1
                        packets[index] = packet
                        if picture is not None:
                            pictures[index] = picture
                        if tool is not None and (
                            self.last_preview_index is None or index == preview
                        ):
                            if self.run_profile.output_previews:
                                self.inspection.copy_results_from(tool)
                            self.last_preview_index = index
                    for b, timing, elapsed in timing_history:
                        b.timing, b.elapsed_ms = timing, elapsed
                    if tool is not None:
                        for b, completed_block in zip(
                            (self.inspection, *walk(self.graph)),
                            (tool, *walk(tool.graph)),
                        ):
                            b.timing.absorb(completed_block.timing)
                            b.elapsed_ms = b.timing.last_ms or 0.0
                    if not stop.is_set():
                        if pass_limit and counts["pass"] >= pass_limit:
                            reason = "pass_limit"
                            stop.set()
                        elif fail_limit and counts["fail"] + counts["error"] >= fail_limit:
                            reason = "failure_limit"
                            stop.set()
                if stop.is_set():
                    for future in active:
                        future.cancel()
        completed_count = sum(counts.values())
        complete = completed_count == total
        policy = p["result_policy"]
        passed = (
            complete and counts["pass"] == total
            if policy == "all_pass"
            else counts["pass"] > 0
            if policy == "any_pass"
            else counts["pass"] >= required
            if policy == "min_passes"
            else complete and counts["fail"] <= allowed
        )
        passed = passed and counts["error"] == 0
        if not total:
            passed = p["empty_policy"] == "pass"
        if reason == "cancelled":
            passed = False
        summary = {
            "model": "parallel_inspection_v1",
            "items": records,
            "total": total,
            "completed": completed_count,
            "pass_count": counts["pass"],
            "fail_count": counts["fail"],
            "error_count": counts["error"],
            "cancelled": sum(r["status"] == "cancelled" for r in records),
            "not_started": sum(r["status"] == "not_started" for r in records),
            "complete": complete,
            "stop_reason": reason,
            "passed": passed,
            "preview_index": self.last_preview_index,
            "parent_ticket": parent,
        }
        overlay = overlay_image(image)
        colors = {
            "pass": (60, 220, 70),
            "fail": (30, 70, 240),
            "error": (30, 30, 210),
            "cancelled": (160, 160, 160),
            "not_started": (90, 90, 90),
        }
        for record in records:
            x, y = np.rint(record["pose"]["origin"]).astype(int)
            color = colors[record["status"]]
            cv2.circle(overlay, (x, y), 8, color, 2)
            cv2.putText(
                overlay,
                str(record["index"] + 1),
                (x + 8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
            )
        return Result(
            {
                "results": summary,
                "group": FlowGroup(
                    tuple(packets[i] for i in sorted(packets)),
                    tuple(str(i) for i in sorted(packets)),
                )
                if packets
                else NO_VALUE,
                "images": FlowGroup(
                    tuple(pictures[i] for i in sorted(pictures)),
                    tuple(str(i) for i in sorted(pictures)),
                )
                if pictures
                else NO_VALUE,
                "pass_count": counts["pass"],
                "fail_count": counts["fail"],
                "error_count": counts["error"],
                "complete": complete,
                "overlay": overlay,
            },
            passed=passed,
            score=counts["pass"] / total if total else None,
            graphics={"per-part outcomes": overlay},
            message=f"{completed_count}/{total} inspected · {counts['pass']} pass · "
            f"{counts['fail']} fail · {counts['error']} errors · {reason}",
        )
