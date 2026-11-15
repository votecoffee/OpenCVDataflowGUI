"""Independent local MQTT packet server and explicit EtherNet/IP request contract."""

import socket
import threading
from types import SimpleNamespace

import pytest

from opencv_dataflow.connection_health import MQTTHealth
from opencv_dataflow.connection_health.providers import cip_identity


class Broker:
    def __init__(self, *, acknowledge=True, subscribe=True, message=False, retained=False):
        self.server = socket.socket()
        self.server.bind(("127.0.0.1", 0))
        self.port = self.server.getsockname()[1]
        self.server.listen()
        self.server.settimeout(0.1)
        self.done = threading.Event()
        self.error = None
        self.acknowledge, self.subscribe = acknowledge, subscribe
        self.message, self.retained = message, retained
        self.operations = []
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    @staticmethod
    def exact(sock, size):
        data = b""
        while len(data) < size:
            chunk = sock.recv(size - len(data))
            if not chunk:
                raise EOFError
            data += chunk
        return data

    def run(self):
        try:
            while not self.done.is_set():
                try:
                    client, _ = self.server.accept()
                    break
                except socket.timeout:
                    continue
            else:
                return
            with client:
                client.settimeout(1)
                while not self.done.is_set():
                    header = self.exact(client, 1)[0]
                    size, shift = 0, 0
                    while True:
                        byte = self.exact(client, 1)[0]
                        size += (byte & 127) << shift
                        shift += 7
                        if not byte & 128:
                            break
                    data = self.exact(client, size)
                    self.operations.append(header >> 4)
                    if header >> 4 == 1 and self.acknowledge:
                        client.sendall(b"\x20\x02\x00\x00")
                    elif header >> 4 == 8:
                        client.sendall(
                            b"\x90\x03" + data[:2] + bytes([0 if self.subscribe else 128])
                        )
                        if self.message:
                            topic = data[4 : 4 + int.from_bytes(data[2:4], "big")]
                            packet = len(topic).to_bytes(2, "big") + topic + b"{}"
                            client.sendall(
                                bytes([0x31 if self.retained else 0x30, len(packet)]) + packet
                            )
                    elif header >> 4 == 14:
                        break
        except EOFError, ConnectionError, socket.timeout:
            pass
        except Exception as exc:
            self.error = exc

    def close(self):
        self.done.set()
        self.thread.join(timeout=2)
        self.server.close()
        assert not self.thread.is_alive()
        assert self.error is None


@pytest.mark.parametrize(
    "case,expected",
    [
        ("broker", "ok"),
        ("subscription", "ok"),
        ("rejected", "warning"),
        ("no_message", "warning"),
        ("message", "ok"),
        ("retained_excluded", "warning"),
        ("no_connack", "alarm"),
    ],
)
def test_real_paho_handshake_subscription_and_receipt(case, expected):
    pytest.importorskip("paho.mqtt.client")
    server = Broker(
        acknowledge=case != "no_connack",
        subscribe=case != "rejected",
        message=case in ("message", "retained_excluded"),
        retained=case == "retained_excluded",
    )
    block = MQTTHealth(
        parameters={
            "enabled": True,
            "host": "127.0.0.1",
            "port": server.port,
            "timeout_seconds": 0.2,
            "subscriptions": [] if case in ("broker", "no_connack") else ["cell/health"],
            "require_message": case in ("no_message", "message", "retained_excluded"),
            "include_retained": case != "retained_excluded",
        }
    )
    try:
        result = block.execute({})
        assert result is not None, block.error
        assert result.outputs["state"] == expected
        assert result.outputs["connected"] == (expected != "alarm")
        # The monitor never sends a PUBLISH packet to the broker.
        assert 1 in server.operations and 3 not in server.operations
    finally:
        block.close()
        server.close()


def test_cip_uses_read_only_identity_service_and_always_closes(monkeypatch):
    pycomm3 = pytest.importorskip("pycomm3")
    requests, closed = [], []

    class Client:
        def __init__(self, route):
            assert route == "example/0"

        def open(self):
            return True

        def generic_message(self, **request):
            assert self.socket_timeout == 0.25
            requests.append(request)
            return SimpleNamespace(value={"serial": 1, "status": b"\0\0"})

        def close(self):
            closed.append(True)

    monkeypatch.setattr(pycomm3, "CIPDriver", Client)
    assert cip_identity("example/0", 0.25) == {"serial": 1}
    assert closed == [True]
    request = requests[0]
    assert request["service"] == pycomm3.Services.get_attributes_all
    assert request["class_code"] == pycomm3.ClassCode.identity_object
    assert request["instance"] == 1 and request["connected"] is False
    assert request["unconnected_send"] is True  # Routed backplane identity.


def test_direct_cip_device_does_not_require_an_unconnected_send_router(monkeypatch):
    pycomm3 = pytest.importorskip("pycomm3")
    requests = []

    class Client:
        def __init__(self, route):
            assert route == "example"

        def open(self):
            return True

        def generic_message(self, **request):
            requests.append(request)
            return SimpleNamespace(value={"serial": 2})

        def close(self):
            pass

    monkeypatch.setattr(pycomm3, "CIPDriver", Client)
    assert cip_identity("example", 1) == {"serial": 2}
    assert requests[0]["unconnected_send"] is False
