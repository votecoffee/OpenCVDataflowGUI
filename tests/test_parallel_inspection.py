import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Block, Graph, Port, Result
from opencv_dataflow.examples import parallel_parts_graph
from opencv_dataflow.parallel_inspection import ParallelInspect
from opencv_dataflow.payloads import NO_VALUE, FlowGroup
from opencv_dataflow.poses import local_offset, make_pose
from opencv_dataflow.runner import describe
from opencv_dataflow.toolblocks import Binding, ToolBlock, encapsulate


def pose_list(count=25):
    return [
        make_pose(local_offset(20 + (i % 5) * 30, 20 + (i // 5) * 30, 0), [180, 180], {})
        for i in range(count)
    ]


def custom_inspector(workers=4, **parameters):
    r = builtin_registry()

    class InspectPart(Block):
        type_id, title = "custom.inspect_part", "Test part inspection"
        inputs = (Port("image"), Port("pose", "data"), Port("context", "data", required=False))
        outputs = (Port("data", "data"),)
        started, finished, threads, lock = [], [], set(), threading.Lock()
        barrier = None
        fail_every = 0
        raise_every = 0
        delay = 0.0

        def process(self, inputs):
            index = self.input_payloads["image"]["inspection"]["index"]
            with self.lock:
                self.started.append(index)
                self.threads.add(threading.get_ident())
            try:
                if self.barrier:
                    self.barrier.wait(timeout=3)
                if self.delay:
                    time.sleep(self.delay)
                if self.raise_every and (index + 1) % self.raise_every == 0:
                    raise ValueError("bad part")
                original = int(inputs["image"][0, 0])
                inputs["image"][:] = index + 1
                return Result(
                    {
                        "data": {
                            "index": index,
                            "original_pixel": original,
                            "context": inputs.get("context"),
                        }
                    },
                    passed=not (self.fail_every and (index + 1) % self.fail_every == 0),
                )
            finally:
                with self.lock:
                    self.finished.append(index)

    r.register(InspectPart)
    graph = Graph()
    part = graph.add(r.create(InspectPart.type_id))
    template = ToolBlock(
        graph,
        {
            "image": Binding(part.id, "image"),
            "pose": Binding(part.id, "pose"),
            "context": Binding(part.id, "context"),
        },
        {"data": Binding(part.id, "data")},
    )
    block = ParallelInspect(
        inspection=template, registry=r, parameters={"workers": workers, **parameters}
    )
    return r, block, InspectPart


def run(block, features=None):
    image = np.zeros((180, 180), np.uint8)
    block.execute(
        {
            "image": image,
            "features": pose_list() if features is None else features,
            "context": {"recipe": "test"},
        }
    )
    assert block.last_result is not None, block.error
    assert not image.any()
    return block.last_result.outputs["results"]


def test_25_inspections_use_distinct_workers_isolate_images_and_aggregate_original_order():
    _, block, tool = custom_inspector()
    tool.barrier = threading.Barrier(4)
    # 24 barrier participants followed by an unpaired 25th would stall; use 24 here.
    results = run(block, pose_list(24))
    assert block.state == "pass" and results["complete"] and results["pass_count"] == 24
    assert len(tool.threads) == 4 and sorted(tool.finished) == list(range(24))
    assert [r["index"] for r in results["items"]] == list(range(24))
    assert all(r["outputs"]["data"]["original_pixel"] == 0 for r in results["items"])
    assert all(r["outputs"]["data"]["context"] == {"recipe": "test"} for r in results["items"])


@pytest.mark.parametrize(
    "policy,parameters,passed",
    [
        ("all_pass", {}, False),
        ("any_pass", {}, True),
        ("min_passes", {"required_passes": 20}, True),
        ("min_passes", {"required_passes": 21}, False),
        ("max_failures", {"allowed_failures": 5}, True),
    ],
)
def test_25_parts_aggregate_policy_and_failures(policy, parameters, passed):
    _, block, tool = custom_inspector(result_policy=policy, **parameters)
    tool.fail_every = 5
    result = run(block)
    assert result["complete"] and result["completed"] == 25
    assert result["pass_count"] == 20 and result["fail_count"] == 5 and result["error_count"] == 0
    assert result["passed"] is passed
    assert block.state == ("pass" if passed else "fail")
    group = block.last_result.outputs["group"]
    assert isinstance(group, FlowGroup) and len(group.items) == 25
    tickets = [p.payload["ticket"] for p in group.items]
    assert [t["value"] for t in tickets] == list(range(1, 26))
    assert len({t["session"] for t in tickets}) == 1


@pytest.mark.parametrize(
    "limit,fail_every,reason,passes,failures",
    [
        ({"stop_after_passes": 3}, 0, "pass_limit", 3, 0),
        ({"stop_after_failures": 2}, 1, "failure_limit", 0, 2),
    ],
)
def test_exact_early_stop_with_one_worker(limit, fail_every, reason, passes, failures):
    _, block, tool = custom_inspector(workers=1, **limit)
    tool.fail_every = fail_every
    result = run(block)
    assert not result["complete"] and result["stop_reason"] == reason
    assert result["pass_count"] == passes and result["fail_count"] == failures
    assert len(tool.started) == passes + failures
    assert result["not_started"] == 25 - passes - failures
    assert block.state == "fail"  # all_pass cannot accept uninspected parts.


def test_parallel_early_stop_drains_inflight_work_and_returns_no_late_tasks():
    _, block, tool = custom_inspector(
        stop_after_passes=1, result_policy="min_passes", required_passes=1
    )
    tool.barrier = threading.Barrier(4)
    tool.delay = 0.02
    result = run(block)
    assert (
        block.state == "pass" and not result["complete"] and result["stop_reason"] == "pass_limit"
    )
    assert 1 <= result["pass_count"] <= 4 and len(tool.started) == 4
    assert sorted(tool.started) == sorted(tool.finished)
    assert result["not_started"] == 21


def test_errors_count_towards_failure_stop_and_do_not_disappear():
    _, block, tool = custom_inspector(workers=1, stop_after_failures=1)
    tool.raise_every = 1
    result = run(block)
    assert block.state == "fail" and result["error_count"] == 1 and result["fail_count"] == 0
    assert "bad part" in result["items"][0]["error"] and result["not_started"] == 24


def test_cancel_before_submission_has_explicit_unknown_parts():
    _, block, tool = custom_inspector()
    block.execute(
        {"image": np.zeros((180, 180), np.uint8), "features": pose_list()}, cancelled=lambda: True
    )
    result = block.last_result.outputs["results"]
    assert result["stop_reason"] == "cancelled" and result["completed"] == 0
    assert result["not_started"] == 25 and not result["complete"] and not tool.started


@pytest.mark.parametrize("policy,passed", [("fail", False), ("pass", True)])
def test_empty_collection_policy(policy, passed):
    _, block, tool = custom_inspector(empty_policy=policy)
    result = run(block, [])
    assert result["complete"] and result["passed"] is passed and not tool.started
    assert block.last_result.outputs["group"] is NO_VALUE


def test_invalid_pose_limit_and_template_fail_before_any_jobs():
    _, block, tool = custom_inspector(max_items=24)
    block.execute({"image": np.zeros((180, 180), np.uint8), "features": pose_list()})
    assert block.state == "error" and not tool.started
    block.parameters["max_items"] = 64
    bad = pose_list()
    bad[12] = {}
    block.execute({"image": np.zeros((180, 180), np.uint8), "features": bad})
    assert block.state == "error" and not tool.started
    block.parameters["image_terminal"] = "missing"
    block.execute({"image": np.zeros((180, 180), np.uint8), "features": pose_list()})
    assert block.state == "error" and not tool.started


def test_blob_collection_produces_25_local_inspections():
    registry = builtin_registry()
    image = np.zeros((180, 180), np.uint8)
    for pose in pose_list():
        cv2.circle(image, tuple(np.rint(pose["origin"]).astype(int)), 3, 255, -1)
    finder = registry.create("inspect.blobs", parameters={"min_area": 3})
    finder.execute({"image": image})
    inspect = registry.create("flow.inspect_each")
    inspect.execute({"image": image, "features": finder.last_result.outputs["collection"]})
    assert inspect.state == "pass", inspect.error
    assert inspect.last_result.outputs["pass_count"] == 25


def test_25_pattern_example_and_nested_saved_inspection(tmp_path):
    r = builtin_registry()
    graph = Graph.load(Path(__file__).parents[1] / "examples/parallel_parts/parallel_parts.json", r)
    assert graph.to_dict() == parallel_parts_graph(r).to_dict()
    for _ in range(3):
        graph.run()
        assert all(b.state == "pass" for b in graph.blocks.values()), [
            (b.state, b.error) for b in graph.blocks.values()
        ]
        assert graph.blocks["parts_match"].last_result.outputs["count"] == 25
        inspect = graph.blocks["parts_inspect"]
        assert inspect.last_result.outputs["pass_count"] == 25
        assert len(inspect.last_result.outputs["images"].items) == 25
        assert "parallel_inspection_v1" in json.dumps(describe(graph))
    nested = encapsulate(graph, ["parts_inspect"], "Repeated inspection", r)
    path = tmp_path / "nested.json"
    graph.save(path)
    loaded = Graph.load(path, r)
    loaded.run()
    assert loaded.blocks[nested.id].state == "pass", loaded.blocks[nested.id].error


def test_result_budget_stops_and_reports_error_without_retaining_large_images():
    graph = parallel_parts_graph(builtin_registry())
    block = graph.blocks["parts_inspect"]
    block.parameters.update(workers=1, max_result_mb=0.001)
    graph.run()
    result = block.last_result.outputs["results"]
    assert block.state == "fail" and result["stop_reason"] == "result_limit"
    assert result["error_count"] == 1 and result["not_started"] == 24
    assert block.last_result.outputs["images"] is NO_VALUE


def test_real_pattern_parts_missing_spots_produce_20_pass_and_5_fail():
    graph = parallel_parts_graph(builtin_registry())
    graph.blocks["parts_source"].parameters["fail_every"] = 5
    graph.run()
    assert graph.blocks["parts_match"].last_result.outputs["count"] == 25
    result = graph.blocks["parts_inspect"].last_result.outputs["results"]
    assert result["complete"] and result["pass_count"] == 20 and result["fail_count"] == 5


def test_input_memory_guard_and_ticket_mismatch_reject_before_submission():
    from opencv_dataflow.payloads import ValuePacket
    from opencv_dataflow.tickets import TicketClock

    _, block, tool = custom_inspector(max_input_mb=0.001)
    block.execute({"image": np.zeros((180, 180), np.uint8), "features": pose_list()})
    assert block.state == "error" and not tool.started and "budget" in block.error
    block.parameters["max_input_mb"] = 256.0
    clock = TicketClock()
    block.execute(
        {
            "image": ValuePacket(np.zeros((180, 180), np.uint8), {"ticket": clock.issue("camera")}),
            "features": ValuePacket(pose_list(), {"ticket": clock.issue("other")}),
        }
    )
    assert block.state == "error" and "tickets differ" in block.error and not tool.started


def test_editor_nested_inspection_region_save_reload_and_template_files(app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QDialog, QFileDialog
    from test_pattern_editor import run as run_window

    from opencv_dataflow.app import MainWindow
    from opencv_dataflow.region_editor import RegionEditor

    window = MainWindow()
    window.load_parallel_parts_example()
    run_window(window, app)
    block = window.graph.blocks["parts_inspect"]
    assert block.state == "pass", block.error
    assert block.last_result.outputs["results"]["completed"] == 25
    window.inspector.select(block)
    assert "parallel_inspection_v1" in window.inspector.details.toPlainText()
    window.enter_toolblock(block)
    assert window.graph is block.inspection.graph
    region = window.graph.blocks["part_region"]
    window.inspector.select(region)

    def edit(dialog):
        assert dialog.canvas.pose is not None
        dialog.canvas.parameters["radius"] = 4
        dialog.accept()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(RegionEditor, "exec", edit)
    window.inspector.edit_region()
    assert region.parameters["radius"] == 4
    window.leave_toolblock()
    assert window.inspector.block is block
    path = tmp_path / "parallel.json"
    window.graph.save(path)
    restored = Graph.load(path, window.registry)
    assert (
        restored.blocks[block.id].inspection.graph.blocks["part_region"].parameters["radius"] == 4
    )
    run_window(window, app)
    assert block.state == "pass", block.error
    template_path = tmp_path / "inspection.toolblock.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(template_path), ""))
    window.export_toolblock()
    assert template_path.exists()
    data = json.loads(template_path.read_text(encoding="utf-8"))
    next(b for b in data["graph"]["blocks"] if b["id"] == "part_region")["parameters"]["radius"] = 6
    template_path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(template_path), ""))
    window.inspector.load_inspection()
    assert block.inspection.graph.blocks["part_region"].parameters["radius"] == 6
    original = block.inspection.definition()
    warnings = []
    monkeypatch.setattr(window, "warn", warnings.append)
    del data["inputs"]["pose"]
    template_path.write_text(json.dumps(data), encoding="utf-8")
    window.inspector.load_inspection()
    assert warnings and block.inspection.definition() == original
    window.dirty = False
    window.close()
