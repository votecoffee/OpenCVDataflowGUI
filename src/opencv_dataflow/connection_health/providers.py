"""Read-only protocol probes; no Qt, acquisition tickets or equipment writes."""

from abc import ABC, abstractmethod
from pathlib import Path

from .. import fanuc, logix, mqtt


def resource(name, ok, message, *, critical=False, **details):
    return {
        "name": name,
        "state": "ok" if ok else "alarm" if critical else "warning",
        "message": message,
        **details,
    }


class HealthProbe(ABC):
    """An owned, serially used observer; implementations must bound I/O and close."""

    @abstractmethod
    def check(self, parameters, project_dir, cancelled):
        """Return connected/message/resources. Transport failures may raise."""

    def close(self):
        pass


class ROSProbe(HealthProbe):
    def __init__(self):
        self.client = None

    def check(self, p, project_dir, cancelled):
        from ..ros2_client import ROSClient

        if self.client is None:
            config = {
                k: p[k]
                for k in (
                    "domain_id",
                    "namespace",
                    "discovery_range",
                    "required_nodes",
                    "required_services",
                    "topics",
                    "reliability",
                    "durability",
                )
            }
            config.update(kind="health", role="health", publication_mode="asynchronous")
            self.client = ROSClient(config, p["startup_timeout_seconds"], cancelled)
        return self.client.request(
            "health",
            p["timeout_seconds"] + 1,
            cancelled,
            timeout=p["timeout_seconds"],
        )

    def close(self):
        client, self.client = self.client, None
        if client is not None:
            client.close()


class MQTTProbe(HealthProbe):
    """Fresh CONNACK on every check, then independent SUBACK/receipt checks."""

    def check(self, p, project_dir, cancelled):
        settings = {**mqtt.MQTTBlock.defaults, **p, "topic": "health"}
        connection = mqtt.MQTTConnection(settings, project_dir, subscribe=False)
        rows = []
        try:
            connection.open(settings, cancelled, subscribe=False)
            if p["subscriptions"]:
                acknowledgments, received = {}, set()

                def subscribed(_client, _userdata, mid, reasons, _properties):
                    acknowledgments[mid] = bool(reasons) and all(not r.is_failure for r in reasons)

                def message(_client, _userdata, value):
                    for topic in p["subscriptions"]:
                        if connection.topic_matches(topic, value.topic):
                            if p["include_retained"] or not value.retain:
                                received.add(topic)

                connection.client.on_subscribe = subscribed
                connection.client.on_message = message
                mids = {}
                for topic in p["subscriptions"]:
                    rc, mid = connection.client.subscribe(topic, p["qos"])
                    if rc:
                        raise ConnectionError(f"MQTT subscription transport failed ({rc})")
                    mids[topic] = mid
                connection.wait(
                    lambda: (
                        all(mid in acknowledgments for mid in mids.values())
                        and (not p["require_message"] or all(t in received for t in mids))
                    ),
                    cancelled,
                )
                for topic, mid in mids.items():
                    acknowledged = acknowledgments.get(mid)
                    ok = acknowledged is True and (not p["require_message"] or topic in received)
                    rows.append(
                        resource(
                            topic,
                            ok,
                            "Subscription acknowledged"
                            if ok
                            else "Subscription rejected"
                            if acknowledged is False
                            else "No subscription acknowledgment"
                            if acknowledged is None
                            else "No qualifying message within the probe window",
                            acknowledged=acknowledged,
                            message_received=topic in received,
                        )
                    )
            return {"connected": True, "message": "MQTT broker answered CONNACK", "resources": rows}
        finally:
            connection.close(abort=True)


def cip_identity(route, timeout):
    from pycomm3 import CIPDriver, ClassCode, Services
    from pycomm3.custom_types import ModuleIdentityObject

    client = CIPDriver(route)
    client.socket_timeout = timeout
    try:
        if not client.open():
            raise ConnectionError("EtherNet/IP session did not open")
        reply = client.generic_message(
            service=Services.get_attributes_all,
            class_code=ClassCode.identity_object,
            instance=1,
            connected=False,
            unconnected_send="/" in route.strip("/"),
            route_path=True,
            data_type=ModuleIdentityObject,
            name="health_identity",
        )
        if not reply or not isinstance(reply.value, dict):
            raise ConnectionError("No valid CIP Identity response")
        # Keep only identification fields, not driver objects or binary status.
        return {
            k: reply.value[k]
            for k in ("vendor", "product_type", "product_code", "serial", "product_name")
            if k in reply.value
        }
    finally:
        client.close()


class EtherNetIPProbe(HealthProbe):
    def check(self, p, project_dir, cancelled):
        if cancelled():
            raise InterruptedError("Health check stopped")
        identity = cip_identity(p["route"], p["timeout_seconds"])
        return {
            "connected": True,
            "message": "CIP Identity answered",
            "resources": [],
            "identity": identity,
        }


class LogixProbe(EtherNetIPProbe):
    def __init__(self):
        self.client = None

    def close(self):
        client, self.client = self.client, None
        if client is not None:
            client.close()

    def check(self, p, project_dir, cancelled):
        result = super().check(p, project_dir, cancelled)
        if not p["tags"]:
            return result
        try:
            if self.client is None:
                self.client = logix.open_logix(
                    p["route"],
                    p["timeout_seconds"],
                    any(t.casefold().startswith("program:") for t in p["tags"]),
                )
            values = self.client.read(*p["tags"])
            replies = values if isinstance(values, list) else [values]
            if len(replies) != len(p["tags"]):
                raise ConnectionError("Incomplete Logix tag response")
            rows = []
            for name, reply in zip(p["tags"], replies, strict=True):
                try:
                    logix.check_reply(reply, name)
                    rows.append(resource(name, True, "Tag read answered"))
                except Exception as exc:
                    rows.append(
                        resource(
                            name,
                            False,
                            f"Tag read failed: {str(exc)[:256]}",
                            error_type=type(exc).__name__,
                        )
                    )
            result["resources"] = rows
        except Exception as exc:
            result["resources"] = [
                resource(
                    t,
                    False,
                    f"Tag access unavailable: {str(exc)[:256]}",
                    error_type=type(exc).__name__,
                )
                for t in p["tags"]
            ]
        if any(r["state"] != "ok" for r in result["resources"]):
            self.close()  # Reconnect the tag channel on the next scheduled check.
            # Do not label a dead connection as merely a bad tag.
            cip_identity(p["route"], p["timeout_seconds"])
        return result


class FanucProbe(HealthProbe):
    def check(self, p, project_dir, cancelled):
        profile = (Path(project_dir or Path.cwd()) / p["profile"]).resolve()

        def call(operation, **arguments):
            if cancelled():
                raise InterruptedError("Health check stopped")
            return fanuc.workbench_call(profile, operation, arguments, p["timeout_seconds"])

        def status():
            value = call(
                "core.controller.status", robot_id=p["robot_id"], group=p["group"], alarm_limit=0
            )
            if (
                not isinstance(value, dict)
                or not value.get("timestamp")
                or value.get("connected") is False
            ):
                raise ConnectionError("FANUC status did not return a timestamped response")
            return value

        status()
        rows = []
        for target in p["registers"]:
            try:
                value = call("core.registers.read", robot_id=p["robot_id"], target=target)
                ok = value.get("robot_id") == p["robot_id"] and value.get("target") == target
                rows.append(
                    resource(
                        target,
                        ok,
                        "Register read answered"
                        if ok
                        else "Register receipt did not match the requested target",
                    )
                )
            except Exception as exc:
                rows.append(
                    resource(
                        target,
                        False,
                        f"Register access unavailable: {str(exc)[:256]}",
                        error_type=type(exc).__name__,
                    )
                )
        if any(r["state"] != "ok" for r in rows):
            status()  # Confirm the controller still responds before choosing yellow.
        return {"connected": True, "message": "FANUC controller status answered", "resources": rows}


def endpoint_identity(block):
    """Safe endpoint grouping for logs; custom plugins may supply their own identity."""
    custom = getattr(block, "connection_identity", None)
    if callable(custom):
        return custom()
    p, kind = block.parameters, block.type_id
    if "mqtt" in kind and "host" in p:
        return (
            "mqtt",
            p["host"].strip().casefold(),
            p["port"],
            p["tls"],
            p["username"],
            p["ca_file"],
        )
    if "domain_id" in p and ("ros" in kind or "ros2" in block.__module__):
        return ("ros2", p["domain_id"], p["discovery_range"])
    if "route" in p and ("logix" in kind or "ethernetip" in kind):
        return ("ethernetip", p["route"].strip().casefold())
    if "fanuc" in kind and "profile" in p:
        path = (Path(block.project_dir or Path.cwd()) / p["profile"]).resolve()
        return ("fanuc", str(path).casefold(), p["robot_id"].strip())
    return None
