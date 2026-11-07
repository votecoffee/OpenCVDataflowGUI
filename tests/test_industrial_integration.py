"""Industrial blocks through test transports and real graph ticket scheduling."""

import struct
from types import SimpleNamespace

import numpy as np

from opencv_dataflow import fanuc, logix
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE, ValuePacket
from opencv_dataflow.tickets import TicketClock, ticket_key

ROUTE = "192.0.2.1/0"
POSITION = {
    "initialized": True, "representation": "cartesian",
    "user_frame": 0, "user_tool": 0,
    "values": {"X": 123, "Y": -12, "Z": 38, "W": 90, "P": 0, "R": 90},
}


def reply(tag, value, kind=None):
    return SimpleNamespace(tag=tag, value=value, type=kind, error=None)


def test_logix_trigger_gates_acquisition_and_preserves_the_plc_ticket(monkeypatch):
    polls = iter([(True, 10), (False, 10), (True, 11), (True, 11)])
    addresses = []

    def read(*tags):
        addresses.append(tags)
        trigger, part = next(polls)
        return [reply("Start", trigger), reply("PartId", part)]

    client = SimpleNamespace(read=read, close=lambda: None)
    monkeypatch.setattr(logix, "open_logix", lambda *args: client)
    registry = builtin_registry()
    graph = Graph()
    source = graph.add(registry.create("input.logix_tags", parameters={
        "route": ROUTE, "enabled": True, "trigger_tag": "Start",
        "ticket_tag": "PartId", "value_tag": "PartId",
    }))
    camera = graph.add(registry.create("input.demo"))
    camera.ticket_settings = {"mode": "external", "value": 1}
    graph.connect(source.id, "ticket", camera.id, "ticket")
    for _ in range(2):
        graph.run()
        assert source.state == "waiting"
        assert camera.state == "inactive"
        assert source.current_emissions["ticket"] == []
    graph.run()
    assert source.last_result.outputs["value"] == 11
    acquired = camera.current_emissions["image"][0].payload["ticket"]
    assert acquired["value"] == 11
    assert ticket_key(acquired) == ticket_key(source.last_result.outputs["ticket"])
    graph.run()
    assert camera.state == "inactive"
    assert addresses == [("Start", "PartId")] * 4


def test_logix_write_gates_verifies_real_rounding_and_holds_uncertain_writes(monkeypatch):
    value = 0.1
    rounded = struct.unpack("<f", struct.pack("<f", value))[0]
    reads = iter([0.0, rounded, rounded, 99.0])
    writes = []

    def write(item):
        writes.append(item)
        return reply(item[0], item[1], "REAL")

    def read(tag):
        return reply(tag, next(reads), "REAL")

    client = SimpleNamespace(read=read, write=write, close=lambda: None)
    monkeypatch.setattr(logix, "open_logix", lambda *args: client)
    block = builtin_registry().create("industrial.logix_write", parameters={
        "route": ROUTE, "tag": "Result", "write_enabled": True,
    })
    assert block.execute({"enable": False, "value": value}).outputs["written"] is NO_VALUE
    assert writes == []
    assert block.execute({"enable": True, "value": value}).outputs["readback"] == rounded
    block.execute({"enable": True, "value": value})
    assert "readback differs" in block.error
    block.execute({"enable": True, "value": value})
    assert "Inspect" in block.error and len(writes) == 2


def test_fanuc_register_read_and_ticketed_cartesian_pose(monkeypatch, tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text("{}", encoding="utf-8")
    common = {"profile": str(profile), "robot_id": "robot", "enabled": True}

    def call(path, operation, arguments, timeout):
        assert path == profile and arguments["robot_id"] == "robot"
        if operation == "core.controller.status":
            assert arguments == {"robot_id": "robot", "group": 1, "alarm_limit": 0}
            return {"position": POSITION, "timestamp": 1.0}
        assert operation == "core.registers.read"
        target = arguments["target"]
        value = {"kind": "R", "value": 42} if target == "R[17]" else POSITION
        return {"robot_id": "robot", "target": target, "kind": "register", "value": value}

    monkeypatch.setattr(fanuc, "workbench_call", call)
    registry = builtin_registry()
    for target, expected in [("R[17]", 42), ("PR[1,12]", POSITION)]:
        read = registry.create("industrial.fanuc_register", parameters={**common, "target": target})
        assert read.execute({}).outputs["value"] == expected

    current = registry.create("industrial.fanuc_position", parameters=common)
    converter = registry.create("industrial.fanuc_pose")
    ticket = TicketClock().issue("capture")
    captured = current.execute({"trigger": ValuePacket(True, {"ticket": ticket})})
    assert captured.outputs["user_frame"] == captured.outputs["user_tool"] == 0
    position = ValuePacket(captured.outputs["position"], captured.payloads["position"])
    converted = converter.execute({"position": position})
    assert converted.payloads["pose"]["ticket"] == ticket
    pose = converted.outputs["pose"]
    assert (pose["source_frame"], pose["target_frame"], pose["units"]) == ("gripper", "base", "mm")
    np.testing.assert_allclose(
        pose["source_to_target"],
        [[0, 0, 1, 123], [1, 0, 0, -12], [0, 1, 0, 38], [0, 0, 0, 1]],
        atol=1e-12,
    )
    converter.execute({"position": {**POSITION, "user_frame": 2}})
    assert "Read UF 2, UT 0. Expected UF 0, UT 0." in converter.error
    assert converter.execute({"position": {**POSITION, "user_tool": False}}) is None
    assert "integer user frame and tool" in converter.error


def test_fanuc_indexed_offset_reads_preserve_ticket_and_check_exact_receipt(monkeypatch, tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text("{}", encoding="utf-8")
    registry = builtin_registry()
    common = {"profile": str(profile), "robot_id": "robot", "group": 2, "index": 3}
    ticket = TicketClock().issue("capture")
    trigger = ValuePacket(True, {"ticket": ticket})
    pending = []

    def call(path, operation, arguments, timeout):
        assert path == profile and operation == "core.controller.sample"
        expected, reply = pending.pop(0)
        assert arguments == {"robot_id": "robot", "expressions": [expected]}
        return reply

    monkeypatch.setattr(fanuc, "workbench_call", call)
    for suffix, kind, label, target in [
        ("uframe", "UFRAME", "user_frame_4", "base"),
        ("utool", "UTOOL", "user_tool_4", "gripper"),
    ]:
        block = registry.create(f"industrial.fanuc_{suffix}", parameters=common)
        assert block.execute({}).outputs["offset"] is NO_VALUE
        block.parameters["enabled"] = True
        expression = f"{kind}[2,4]"
        offset = {"kind": kind, "group": 2, "index": 4, "initialized": True,
                  "representation": "cartesian", "values": POSITION["values"]}
        receipt = {"robot_id": "robot", "provider": "frrobot", "timestamp": 1,
                   "values": {expression: offset}, "errors": {}}
        pending.append((expression, receipt))
        result = block.execute({"trigger": trigger, "index": 4})
        assert result is not None, block.error
        assert result.payloads["offset"]["ticket"] == result.payloads["pose"]["ticket"] == ticket
        assert result.outputs["offset"] == {**offset, "robot_id": "robot", "timestamp": 1.0}
        pose = result.outputs["pose"]
        assert (pose["source_frame"], pose["target_frame"], pose["units"]) == (label, target, "mm")
        np.testing.assert_allclose(
            pose["source_to_target"],
            [[0, 0, 1, 123], [1, 0, 0, -12], [0, 1, 0, 38], [0, 0, 0, 1]],
            atol=1e-12,
        )
        pending.append((expression, {**receipt, "values": {expression: {**offset, "index": 3}}}))
        assert block.execute({"trigger": trigger, "index": 4}) is None
        assert "requested Cartesian" in block.error
        pending.append((expression, {
            **receipt, "values": {}, "errors": {expression: "ValueError"},
        }))
        assert block.execute({"trigger": trigger, "index": 4}) is None
        assert "could not read" in block.error
        assert block.execute({"trigger": trigger, "index": 0}) is None
        assert "Frame index" in block.error
    assert not pending


def test_fanuc_prepare_and_commit_bind_exact_review_and_never_replay(monkeypatch, tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text("{}", encoding="utf-8")
    common = {"profile": str(profile), "robot_id": "robot", "target": "PR[1,12]"}
    approval = {"plan_id": "1" * 32, "review_sha256": "a" * 64, "approved_sha256": "b" * 64}
    calls = []
    proposal = {"kind": "register", "robot_id": "robot", "target": "PR[1,12]"}
    review = {
        "status": "prepared_only", "controller_written": False,
        "proposed_sha256": approval["approved_sha256"], "proposal": proposal,
    }
    allow_reply = False

    def call(path, operation, arguments, timeout):
        assert path == profile
        calls.append(operation)
        if operation == "core.registers.update":
            assert arguments["expected_sha256"] == "c" * 64
            assert arguments["change"] == {"values": {"X": 30}}
            return review
        if operation == "change.status":
            return {
                "status": "prepared", "current_policy": True,
                "review_sha256": approval["review_sha256"], "review": review,
            }
        assert operation == "change.commit" and arguments == approval
        if not allow_reply:
            raise RuntimeError("transport interrupted")
        return {
            "status": "verified", "plan_id": approval["plan_id"],
            "robot_id": "robot", "target": "PR[1,12]",
        }

    monkeypatch.setattr(fanuc, "workbench_call", call)
    registry = builtin_registry()
    prepare = registry.create(
        "industrial.fanuc_prepare_register", parameters={**common, "enabled": True},
    )
    result = prepare.execute({
        "enable": True, "change": {"values": {"X": 30}},
        "reference": {**proposal, "sha256": "c" * 64, "local_path": str(tmp_path / "capture.json")},
    })
    assert result.outputs["review"]["controller_written"] is False

    def writer():
        return registry.create(
            "industrial.fanuc_commit_register", parameters={**common, "write_enabled": True},
        )

    commit = writer()
    assert commit.execute({"enable": False, "approval": approval}).outputs["receipt"] is NO_VALUE
    proposal["target"] = "PR[1,13]"
    commit.execute({"enable": True, "approval": approval})
    assert "does not match" in commit.error and "change.commit" not in calls
    proposal["target"] = common["target"]
    commit.execute({"enable": True, "approval": approval})
    assert "transport interrupted" in commit.error
    commit.execute({"enable": True, "approval": approval})
    assert "Inspect the retained" in commit.error and calls.count("change.commit") == 1

    allow_reply = True
    result = writer().execute({"enable": True, "approval": approval})
    assert result.outputs["receipt"]["status"] == "verified"
