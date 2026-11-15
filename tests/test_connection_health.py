"""Protocol/resource distinction, scheduling, lifecycle, logs and offline status UI."""

import copy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.connection_health import (
    CONNECTION_HEALTH_BLOCKS,
    HealthProbe,
    MQTTHealth,
    ROSHealth,
    providers,
)
from opencv_dataflow.core import Graph


class Probe(HealthProbe):
    def __init__(self, states=("ok",)):
        self.states = iter(states)
        self.calls = self.closed = 0

    def check(self, parameters, project_dir, cancelled):
        self.calls += 1
        state = next(self.states)
        if state == "transport":
            raise TimeoutError("Controlled protocol timeout")
        return {
            "connected": True,
            "message": "Endpoint answered",
            "resources": [{"name": "test", "state": state, "message": "Resource status"}],
        }

    def close(self):
        self.closed += 1


def execute(block):
    result = block.execute({})
    assert result is not None, block.error
    return result


@pytest.mark.parametrize("kind", CONNECTION_HEALTH_BLOCKS)
def test_disabled_monitors_do_not_create_probes_or_tickets(kind, monkeypatch):
    def forbidden():
        raise AssertionError("Disabled monitor attempted network I/O")

    block = kind()
    monkeypatch.setattr(block, "probe_factory", forbidden)
    result = execute(block)
    assert result.outputs["state"] == "disabled"
    assert not any(result.outputs[k] for k in ("ok", "warning", "alarm", "connected", "updated"))
    assert not block.ticket_source and "ticket" not in result.outputs
    assert not block.exposed_outputs
    assert block.status_indicator()["state"] == "disabled"


@pytest.mark.parametrize("state", ["ok", "warning", "alarm", "transport"])
def test_severity_optional_fail_and_protocol_vs_resource_state(state, monkeypatch):
    block = MQTTHealth(parameters={"enabled": True})
    probe = Probe([state])
    monkeypatch.setattr(block, "probe_factory", lambda: probe)
    result = execute(block)
    expected = "alarm" if state == "transport" else state
    assert result.outputs["state"] == expected
    assert result.outputs["connected"] == (state != "transport")
    assert result.outputs["alarm"] == (expected == "alarm")
    assert result.outputs["warning"] == (expected == "warning")
    assert result.passed and not result.fail_reasons
    assert block.status_indicator()["state"] == expected
    block.parameters["fail_on_alarm"] = True
    monkeypatch.setattr(block, "probe_factory", lambda: Probe([state]))
    result = execute(block)
    assert result.passed == (expected != "alarm")
    assert bool(result.fail_reasons) == (expected == "alarm")
    block.close()


def test_poll_interval_cached_outputs_recovery_changes_and_cleanup(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("opencv_dataflow.connection_health.blocks.time.monotonic", lambda: clock[0])
    block = MQTTHealth(parameters={"enabled": True})
    probes = [Probe(["ok", "warning"]), Probe(["ok"])]
    monkeypatch.setattr(block, "probe_factory", lambda: probes.pop(0))
    first = execute(block)
    own = block._probe
    clock[0] += 1
    cached = execute(block)
    assert own.calls == 1 and not cached.outputs["updated"]
    assert cached.outputs["report"]["checked_utc"] == first.outputs["report"]["checked_utc"]
    assert cached.outputs["report"]["age_seconds"] == 1
    clock[0] += 2
    assert execute(block).outputs["warning"] and own.calls == 2
    block.parameters["port"] = 1884
    assert execute(block).outputs["ok"] and own.closed == 1
    newer = block._probe
    block.reset_runtime()
    assert newer.closed == 1 and block._report is None


def test_disconnected_root_runs_with_missing_input_branch_and_no_ticket_contamination(monkeypatch):
    graph, registry = Graph(), builtin_registry()
    monitor = graph.add(registry.create("industrial.mqtt_health", parameters={"enabled": True}))
    graph.add(registry.create("cv.canny"))  # Missing image must not gate the monitor.
    probe = Probe(["warning"])
    monkeypatch.setattr(monitor, "probe_factory", lambda: probe)
    try:
        graph.run()
        assert probe.calls == 1 and monitor.last_result.outputs["warning"]
        graph.run()
        assert probe.calls == 1 and not monitor.last_result.outputs["updated"]
        assert not graph.ticket_results.snapshot()["tickets"]
    finally:
        graph.close()
    assert probe.closed == 1


def test_endpoint_log_aggregation_filters_peers_and_keeps_tracebacks(monkeypatch):
    graph, registry = Graph(), builtin_registry()
    monitor = graph.add(
        registry.create(
            "industrial.mqtt_health", block_id="health", parameters={"include_logs": True}
        )
    )
    peer = graph.add(registry.create("input.mqtt", block_id="reader"))
    other = graph.add(
        registry.create("input.mqtt", block_id="elsewhere", parameters={"host": "other.invalid"})
    )
    peer.logger.warning("Subscription unavailable")
    try:
        raise ValueError("Controlled peer exception")
    except ValueError:
        peer.logger.exception("Decode failed")
    other.logger.error("Other connection")
    result = execute(monitor)
    assert {"health", "reader"} == set(result.outputs["report"]["log_sources"])
    events = result.outputs["logs"]
    assert "Other connection" not in [e["message"] for e in events]
    assert any(
        e.get("exception") and "Controlled peer exception" in e["exception"]["traceback"]
        for e in events
    )
    peer.parameters["host"] = "other.invalid"
    assert monitor.related_log_sources() == {"health"}
    graph.close()


@pytest.mark.parametrize(
    "parameters",
    [
        {"enabled": 1},
        {"poll_interval_seconds": 0},
        {"poll_interval_seconds": 20, "stale_after_seconds": 10},
        {"subscriptions": ["bad/#/filter"]},
        {"subscriptions": ["x", "x"]},
        {"require_message": "yes"},
        {"password_env": "SECRET", "username": ""},
    ],
)
def test_parameter_validation_does_not_probe(parameters, monkeypatch):
    block = MQTTHealth(parameters=parameters)
    monkeypatch.setattr(block, "probe_factory", lambda: pytest.fail("Validation attempted I/O"))
    assert block.execute({}) is None and block.error_kind == "invalid"


@pytest.mark.parametrize(
    "parameters",
    [
        {"required_nodes": ["relative"]},
        {"required_services": [{"name": "/s", "type": "bad"}]},
        {"topics": [{"name": "/t", "type": "std_msgs/msg/String", "max_age_seconds": -1}]},
        {"topics": [{"name": "/t", "type": "std_msgs/msg/String", "minimum_publishers": 0}]},
        {"topics": [{"name": "/t", "type": "std_msgs/msg/String", "unexpected": 1}]},
    ],
)
def test_ros_resource_validation(parameters):
    block = ROSHealth(parameters=parameters)
    assert block.execute({}) is None and block.error_kind == "invalid"


@pytest.mark.parametrize(
    "service",
    [
        {"name": "/robot/trigger", "type": "std_srvs/srv/Trigger", "check_response": True},
        {
            "name": "/driver/get_parameters",
            "type": "rcl_interfaces/srv/GetParameters",
            "check_response": 1,
        },
    ],
)
def test_ros_response_checks_reject_arbitrary_or_nonboolean_service_calls(service):
    block = ROSHealth(parameters={"required_services": [service]})
    assert block.execute({}) is None and block.error_kind == "invalid"


def test_logix_rechecks_protocol_after_a_tag_failure(monkeypatch):
    identities = []
    monkeypatch.setattr(providers, "cip_identity", lambda *a: identities.append(a) or {"serial": 1})
    client = SimpleNamespace(
        read=lambda *a: [
            SimpleNamespace(tag="Good", value=42, error=None),
            SimpleNamespace(tag="Bad", value=None, error="Path unknown"),
        ],
        close=lambda: None,
    )
    monkeypatch.setattr(providers.logix, "open_logix", lambda *a: client)
    report = providers.LogixProbe().check(
        {"route": "example/0", "timeout_seconds": 1, "tags": ["Good", "Bad"]}, None, lambda: False
    )
    assert len(identities) == 2
    assert [r["state"] for r in report["resources"]] == ["ok", "warning"]
    count = [0]

    def disappears(*args):
        count[0] += 1
        if count[0] == 2:
            raise TimeoutError("No CIP response")
        return {}

    monkeypatch.setattr(providers, "cip_identity", disappears)
    with pytest.raises(TimeoutError):
        providers.LogixProbe().check(
            {"route": "example/0", "timeout_seconds": 1, "tags": ["Good", "Bad"]},
            None,
            lambda: False,
        )


def test_fanuc_separates_register_fault_from_controller_fault(monkeypatch, tmp_path):
    calls = []

    def call(profile, operation, arguments, timeout):
        calls.append(operation)
        if operation == "core.controller.status":
            return {"timestamp": 1}
        raise RuntimeError("Register unavailable")

    monkeypatch.setattr(providers.fanuc, "workbench_call", call)
    p = {
        "profile": "p.json",
        "robot_id": "robot",
        "timeout_seconds": 1,
        "group": 1,
        "registers": ["R[1]"],
    }
    report = providers.FanucProbe().check(p, tmp_path, lambda: False)
    assert report["connected"] and report["resources"][0]["state"] == "warning"
    assert calls == ["core.controller.status", "core.registers.read", "core.controller.status"]


def test_healthy_logix_tag_channel_is_reused_and_closed(monkeypatch):
    opened, closed = [], []
    monkeypatch.setattr(providers, "cip_identity", lambda *a: {"serial": 1})

    def open_client(*args):
        opened.append(args)
        return SimpleNamespace(
            read=lambda *a: SimpleNamespace(tag="Good", value=1, error=None),
            close=lambda: closed.append(True),
        )

    monkeypatch.setattr(providers.logix, "open_logix", open_client)
    probe = providers.LogixProbe()
    p = {"route": "example/0", "timeout_seconds": 1, "tags": ["Good"]}
    for _ in range(2):
        assert probe.check(p, None, lambda: False)["resources"][0]["state"] == "ok"
    assert len(opened) == 1 and not closed
    probe.close()
    assert closed == [True]


def test_removing_a_peer_updates_diagnostic_membership_immediately():
    registry, graph = builtin_registry(), Graph()
    monitor = graph.add(registry.create("industrial.mqtt_health", block_id="monitor"))
    peer = graph.add(registry.create("input.mqtt", block_id="peer"))
    assert monitor.related_log_sources() == {"monitor", "peer"}
    graph.remove(peer.id)
    assert monitor.related_log_sources() == {"monitor"}
    graph.close()


def test_saved_monitors_optional_ports_xml_and_graphics_off(tmp_path):
    registry, graph = builtin_registry(), Graph()
    for kind in CONNECTION_HEALTH_BLOCKS:
        block = graph.add(registry.create(kind.type_id))
        block.exposed_outputs = ["ok", "warning", "alarm"]
    try:
        for suffix in ("json", "xml"):
            path = tmp_path / f"health.{suffix}"
            graph.save(path)
            restored = Graph.load(path, registry)
            try:
                restored.set_run_profile("lean")
                restored.run()
                for block in restored.blocks.values():
                    assert block.last_result.outputs["state"] == "disabled"
                    assert block.exposed_outputs == ["ok", "warning", "alarm"]
                    assert block.status_indicator()["state"] == "disabled"
            finally:
                restored.close()
    finally:
        graph.close()


def test_status_thumbnail_staleness_and_aggregate_diagnostic_popup(app, tmp_path):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage, QPainter

    from opencv_dataflow.app import MainWindow
    from opencv_dataflow.diagnostic_viewer import DiagnosticViewer

    graph, registry = Graph(), builtin_registry()
    monitor = graph.add(registry.create("industrial.mqtt_health"))
    peer = graph.add(registry.create("input.mqtt"))
    peer.logger.error("Peer diagnostic")
    graph.run()
    window = MainWindow()
    try:
        window.replace_graph(graph)
        app.processEvents()
        node = window.scene.nodes[monitor.id]
        image = QImage(int(node.width), int(node.height), QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        node.paint(painter, None)
        painter.end()
        point = node.thumbnail_rect.topLeft()
        assert image.pixelColor(int(point.x() + 8), int(point.y() + 8)).name() == "#74859b"
        dialog = monitor.create_plugin_dialog(window, copy.deepcopy(monitor.parameters), None)
        viewer = dialog.findChild(DiagnosticViewer)
        assert viewer.scope.text() == "Connection endpoint"
        assert any(
            viewer.table.topLevelItem(i).text(4) == "Peer diagnostic"
            for i in range(viewer.table.topLevelItemCount())
        )
        dialog.close()
        monitor.last_result.outputs["report"].update(
            state="ok", checked_utc=(datetime.now(timezone.utc) - timedelta(seconds=20)).isoformat()
        )
        assert monitor.status_indicator()["title"] == "STALE"
    finally:
        window.dirty = False
        window.close()
        app.processEvents()
