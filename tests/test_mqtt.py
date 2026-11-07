"""MQTT block contracts independent of broker and optional driver availability."""

import json
from collections import deque
from types import SimpleNamespace

from opencv_dataflow import mqtt
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.runner import describe
from opencv_dataflow.tickets import ticket_key
from opencv_dataflow.toolbox import block_entry


class Connection:
    def __init__(self):
        self.pending = deque()
        self.published = []
        self.closed = []
        self.acknowledge = True
        self.client = self

    def open(self, settings, cancelled, *, subscribe):
        pass

    def wait(self, ready, cancelled):
        return ready()

    def close(self, *, abort=False):
        self.closed.append(abort)

    def publish(self, *args):
        self.published.append(args)
        return SimpleNamespace(rc=0, is_published=lambda: self.acknowledge)


def test_mqtt_reader_only_triggers_acquisition_for_a_message(monkeypatch):
    connection = Connection()
    monkeypatch.setattr(mqtt, "MQTTConnection", lambda *args, **kwargs: connection)
    registry = builtin_registry()
    graph = Graph()
    source = graph.add(
        registry.create(
            "input.mqtt",
            parameters={
                "enabled": True,
                "topic": "cell/+/trigger",
                "ticket_field": "part",
            },
        )
    )
    camera = graph.add(registry.create("input.demo"))
    camera.ticket_settings = {"mode": "external", "value": 1}
    graph.connect(source.id, "ticket", camera.id, "ticket")
    try:
        assert block_entry(source).path == ("Industrial integration", "MQTT")
        graph.run()
        assert source.state == "waiting" and not source.current_emissions["ticket"]
        assert camera.state == "inactive"

        topic = "cell/a/trigger"
        connection.pending.append(
            SimpleNamespace(payload=b'{"part":"A-14"}', topic=topic, qos=1, retain=False)
        )
        graph.run()
        result = source.last_result
        assert result.outputs["payload"] == {"part": "A-14"}
        assert result.outputs["message"] == {"topic": topic, "qos": 1, "retained": False}
        ticket = result.outputs["ticket"]
        assert ticket["value"] == "A-14"
        assert ticket_key(camera.last_result.outputs["ticket"]) == ticket_key(ticket)

        connection.pending.append(
            SimpleNamespace(payload=b'{"part":NaN}', topic=topic, qos=1, retain=False)
        )
        graph.run()
        assert source.state == "error" and not source.current_emissions
        assert camera.last_result is None
    finally:
        graph.close()
    assert connection.closed == [False]


def test_mqtt_writer_requires_enable_and_holds_after_an_unconfirmed_publish(monkeypatch):
    connection = Connection()
    opened = []

    def transport(*args, **kwargs):
        opened.append(True)
        return connection

    monkeypatch.setattr(mqtt, "MQTTConnection", transport)
    writer = builtin_registry().create(
        "industrial.mqtt_publish",
        parameters={
            "topic": "cell/result",
            "payload_format": "text",
            "retain": True,
        },
    )
    data = {"payload": "pièce", "enable": True}
    assert writer.execute(data).outputs["published"] is NO_VALUE
    writer.parameters["write_enabled"] = True
    assert writer.execute({**data, "enable": False}).outputs["published"] is NO_VALUE
    assert writer.execute({**data, "payload": 27}) is None
    assert not opened

    connection.acknowledge = False
    assert writer.execute(data) is None
    assert "outcome is unknown" in writer.error
    assert writer.execute(data) is None
    assert "unconfirmed" in writer.error and len(opened) == 1
    assert connection.published == [("cell/result", "pièce".encode(), 1, True)]
    assert connection.closed == [True]

    writer.reset_runtime()
    connection.acknowledge = True
    assert writer.execute(data).outputs["delivery"]["acknowledged"]
    writer.close()


def test_mqtt_binary_outputs_are_reportable(monkeypatch):
    connection = Connection()
    monkeypatch.setattr(mqtt, "MQTTConnection", lambda *args, **kwargs: connection)
    graph = Graph()
    graph.add(
        builtin_registry().create(
            "input.mqtt",
            parameters={
                "enabled": True,
                "topic": "cell/binary",
                "payload_format": "bytes",
            },
        )
    )
    connection.pending.append(
        SimpleNamespace(payload=b"\xff\x00", topic="cell/binary", qos=0, retain=True)
    )
    try:
        graph.run()
        (record,) = json.loads(json.dumps(describe(graph)))
        assert record["outputs"]["payload"] == {"byte_count": 2, "hex_preview": "ff00"}
    finally:
        graph.close()
