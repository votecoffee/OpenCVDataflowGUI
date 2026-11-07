"""Run the shipped industrial graphs with synthetic data and fake transports."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from opencv_dataflow import fanuc, logix
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.tickets import ticket_key

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "industrial integration"
pytestmark = pytest.mark.local_examples


def load(name):
    return Graph.load(EXAMPLES / name / f"{name}.json", builtin_registry())


def assert_safe_idle(graph):
    graph.run()
    assert all(block.state in ("pass", "waiting", "inactive") for block in graph.blocks.values())


def test_fanuc_example_pairs_raw_cartesian_samples_by_ticket():
    graph = load("fanuc_hand_eye_calibration")
    try:
        graph.run()
        block = graph.blocks["fit"]
        assert block.state == "pass", block.error
        result = block.last_result.outputs
        np.testing.assert_allclose(
            result["camera_to_gripper"]["translation"], [40, -25, 90], atol=1e-8,
        )
        assert [row["ticket"] for row in result["report"]["samples"]] == [
            "capture-1", "capture-2", "capture-3", "capture-4",
        ]
        target_group = graph.blocks["target_group"].last_result.outputs["group"]
        assert [item.payload["ticket"]["value"] for item in target_group.items] == [
            "capture-1", "capture-2", "capture-3", "capture-4",
        ]
    finally:
        graph.close()


def test_fanuc_template_composes_indexed_offsets_and_leaves_writes_disabled(monkeypatch, tmp_path):
    graph = load("fanuc_registers_and_frames")
    calls = []
    profile = tmp_path / "placeholder.json"
    profile.write_bytes(b"{}\n")

    def values(x, y, z):
        return dict(zip("XYZWPR", (x, y, z, 0, 0, 0), strict=True))

    def workbench(path, operation, args, timeout):
        assert path == profile and args["robot_id"] == "test-robot"
        calls.append(operation)
        if operation == "core.controller.status":
            return {"timestamp": 1.0, "position": {
                "initialized": True, "representation": "cartesian",
                "user_frame": 3, "user_tool": 2, "values": values(100, 20, 30),
            }}
        if operation == "core.controller.sample":
            expression, = args["expressions"]
            if expression == "UFRAME[1,3]":
                kind, index, xyz = "UFRAME", 3, (10, 20, 30)
            else:
                assert expression == "UTOOL[1,2]"
                kind, index, xyz = "UTOOL", 2, (5, 1, 2)
            return {"robot_id": "test-robot", "provider": "frrobot", "timestamp": 1.0,
                    "errors": {}, "values": {expression: {
                        "kind": kind, "index": index, "group": 1, "initialized": True,
                        "representation": "cartesian", "values": values(*xyz),
                    }}}
        assert operation == "core.registers.read"
        target = args["target"]
        assert target in ("R[10]", "PR[1,10]")
        register = {"value": 12} if target == "R[10]" else {"values": values(1, 2, 3)}
        return {"target": target, "robot_id": "test-robot", "value": register}

    monkeypatch.setattr(fanuc, "workbench_call", workbench)
    try:
        assert_safe_idle(graph)
        assert not calls
        for key in ("position", "uframe", "utool", "r_read", "pr_read"):
            graph.blocks[key].parameters.update(enabled=True, profile=str(profile),
                                                robot_id="test-robot")
        graph.run()
        result = graph.blocks["flange_in_base"].last_result
        assert result is not None, [(b.id, b.error) for b in graph.blocks.values() if b.error]
        assert result.outputs["transform"]["translation"] == [105, 39, 58]
        assert result.outputs["transform"]["source_frame"] == "gripper"
        assert result.outputs["transform"]["target_frame"] == "base"
        assert sorted(calls) == sorted(["core.controller.status", "core.controller.sample",
                                       "core.controller.sample", "core.registers.read",
                                       "core.registers.read"])
    finally:
        graph.close()


def test_logix_template_handles_one_rising_request_and_verifies_writeback(monkeypatch):
    graph = load("controllogix_trigger_and_writeback")
    request = "Program:Vision.InspectRequest"
    part = "Program:Vision.PartSequence"
    result = "Program:Vision.ResultCount"
    polls = iter([(False, 70), (True, 71), (True, 71)])
    writes = []

    def read(*tags):
        if tags == (result,):
            return SimpleNamespace(tag=result, value=2 if writes else 0, type="DINT", error=None)
        assert set(tags) == {request, part}
        signal, sequence = next(polls)
        values = {request: (signal, "BOOL"), part: (sequence, "DINT")}
        return [
            SimpleNamespace(tag=tag, value=values[tag][0], type=values[tag][1], error=None)
            for tag in tags
        ]

    def write(item):
        writes.append(item)
        return SimpleNamespace(tag=item[0], value=item[1], type="DINT", error=None)

    client = SimpleNamespace(read=read, write=write, close=lambda: None)

    def unexpected_open(*args):
        pytest.fail("A saved template attempted to open a PLC connection")

    monkeypatch.setattr(logix, "open_logix", unexpected_open)
    try:
        assert_safe_idle(graph)
        monkeypatch.setattr(logix, "open_logix", lambda *args: client)
        graph.blocks["read"].parameters.update(route="192.0.2.1/0", enabled=True)
        graph.blocks["write"].parameters.update(route="192.0.2.1/0", write_enabled=True)
        graph.run()
        assert graph.blocks["capture"].state == "inactive"
        assert not writes
        graph.run()
        verdict = graph.blocks["judge"].last_result
        writer = graph.blocks["write"].last_result
        assert verdict.outputs["judgment"]["passed"]
        assert writer.outputs["readback"] == 2 and writes == [(result, 2)]
        ticket = verdict.payloads["value"]["ticket"]
        assert ticket["value"] == 71
        assert ticket_key(writer.payloads["readback"]["ticket"]) == ticket_key(ticket)
        graph.run()
        assert len(writes) == 1
    finally:
        graph.close()
