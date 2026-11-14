"""Lyrical rclpy endpoint used only in the ROS interpreter, never the GUI process."""

import array
import collections
import math
import os
import re
import sys
import threading
import time
import uuid
from importlib.metadata import version

try:
    from .cloud_wire import ROSPointCloud
    from .wire import MAX_DATA_BYTES, ROSImage, header_value, json_bytes
except ImportError:  # Standalone worker, under the ROS interpreter.
    from cloud_wire import ROSPointCloud
    from wire import MAX_DATA_BYTES, ROSImage, header_value, json_bytes


def plain(value, depth=0):
    if depth > 32:
        raise ValueError("ROS data nesting exceeds 32 levels")
    if isinstance(value, dict):
        return {k: plain(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, array.array)):
        return [plain(v, depth + 1) for v in value]
    if callable(getattr(value, "tolist", None)):
        return plain(value.tolist(), depth + 1)
    if callable(getattr(value, "item", None)):
        return plain(value.item(), depth + 1)
    return value


def preserved(requested, actual):
    """Reject silent int/string/bool coercions; permit ordinary float32 rounding."""
    if isinstance(requested, dict):
        return isinstance(actual, dict) and all(
            k in actual and preserved(v, actual[k]) for k, v in requested.items()
        )
    if isinstance(requested, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(requested)
            and all(preserved(a, b) for a, b in zip(requested, actual))
        )
    if type(requested) is bool or type(actual) is bool:
        return type(requested) is type(actual) and requested == actual
    if type(actual) is int:
        return type(requested) in (int, float) and requested == actual
    if type(actual) is float:
        if type(requested) is int:
            return requested == actual
        return (
            type(requested) is float
            and math.isfinite(actual)
            and math.isclose(requested, actual, rel_tol=1e-7, abs_tol=1e-45)
        )
    return type(requested) is type(actual) and requested == actual


class PublishUncertain(RuntimeError):
    """Publication was attempted but its outcome cannot be confirmed."""


class Endpoint:
    def __init__(self, config):
        self.context = self.node = self.executor = self.thread = self.entity = None
        self.lock = threading.Condition()
        self.pending = collections.deque()
        self.pending_bytes = 0
        self.received = self.dropped = 0
        self.error = None
        self.config = config
        self._closed = False
        if os.environ.get("ROS_DISTRO") != "lyrical":
            raise RuntimeError("A sourced ROS 2 Lyrical environment is required for this worker")
        if os.environ.get("RMW_IMPLEMENTATION") != "rmw_fastrtps_cpp":
            raise RuntimeError("This integration requires Fast DDS (rmw_fastrtps_cpp)")
        if config["kind"] not in ("image", "cloud", "data") or config["role"] not in (
            "read",
            "write",
        ):
            raise ValueError("Invalid ROS endpoint role/kind")
        if not re.fullmatch(r"[a-z][a-z0-9_]*/msg/[A-Za-z][A-Za-z0-9_]*", config["message_type"]):
            raise ValueError("Expected an installed package/msg/Type interface")
        import rclpy
        from rclpy.context import Context
        from rclpy.duration import Duration
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.parameter import Parameter
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from rosidl_runtime_py.convert import message_to_ordereddict
        from rosidl_runtime_py.set_message import set_message_fields
        from rosidl_runtime_py.utilities import get_message

        self.rclpy, self.Duration = rclpy, Duration
        self.to_dict = lambda message: plain(message_to_ordereddict(message))
        self.set_fields = set_message_fields
        self.message_class = get_message(config["message_type"])
        try:
            self.context = Context()
            rclpy.init(args=[], context=self.context, domain_id=int(config["domain_id"]))
            identifier = rclpy.get_rmw_implementation_identifier()
            if identifier != "rmw_fastrtps_cpp":
                raise RuntimeError(f"Unexpected RMW implementation: {identifier}")
            self.node = rclpy.create_node(
                "opencv_dataflow_" + uuid.uuid4().hex[:12],
                context=self.context,
                namespace=config["namespace"],
                use_global_arguments=False,
                parameter_overrides=[Parameter("use_sim_time", value=config["use_sim_time"])],
            )
            qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=int(config["queue_depth"]),
                reliability=(
                    ReliabilityPolicy.RELIABLE
                    if config["reliability"] == "reliable"
                    else ReliabilityPolicy.BEST_EFFORT
                ),
                durability=(
                    DurabilityPolicy.TRANSIENT_LOCAL
                    if config["durability"] == "transient_local"
                    else DurabilityPolicy.VOLATILE
                ),
            )
            if config["deadline_seconds"] > 0:
                qos.deadline = Duration(seconds=config["deadline_seconds"])
            if config["lifespan_seconds"] > 0:
                qos.lifespan = Duration(seconds=config["lifespan_seconds"])
            if config["role"] == "read":
                self.entity = self.node.create_subscription(
                    self.message_class, config["topic"], self._receive, qos
                )
            else:
                self.entity = self.node.create_publisher(self.message_class, config["topic"], qos)
            self.info = {
                "ros_distro": os.environ["ROS_DISTRO"],
                "rclpy": version("rclpy"),
                "python": sys.version.split()[0],
                "rmw": identifier,
                "domain_id": config["domain_id"],
                "topic": self.entity.topic_name,
                "message_type": config["message_type"],
                "node": self.node.get_fully_qualified_name(),
                "reliability": config["reliability"],
                "durability": config["durability"],
                "queue_depth": config["queue_depth"],
                "node_use_sim_time": config["use_sim_time"],
            }
            self.executor = SingleThreadedExecutor(context=self.context)
            self.executor.add_node(self.node)
            self.thread = threading.Thread(target=self._spin, name="ros2-endpoint", daemon=True)
            self.thread.start()
        except BaseException:
            self.close()
            raise

    def _spin(self):
        try:
            self.executor.spin()
        except Exception as exc:
            if not self._closed:
                with self.lock:
                    self.error = f"ROS executor stopped: {exc}"
                    self.lock.notify_all()

    def _receive(self, message):
        try:
            if self.config["kind"] in ("image", "cloud"):
                data = message.data
                data = data.to_bytes() if callable(getattr(data, "to_bytes", None)) else bytes(data)
            if self.config["kind"] == "image":
                sample = ROSImage(
                    message.height,
                    message.width,
                    message.encoding,
                    message.step,
                    message.is_bigendian,
                    data,
                    self.to_dict(message.header),
                ).to_wire()
            elif self.config["kind"] == "cloud":
                sample = ROSPointCloud(
                    message.height,
                    message.width,
                    [self.to_dict(f) for f in message.fields],
                    message.is_bigendian,
                    message.point_step,
                    message.row_step,
                    data,
                    message.is_dense,
                    self.to_dict(message.header),
                ).to_wire()
            else:
                sample = self.to_dict(message)
                json_bytes(sample, MAX_DATA_BYTES)
            size = len(json_bytes(sample))
            with self.lock:
                self.received += 1
                if size > self.config["queue_bytes"]:
                    self.dropped += 1
                    return
                while self.pending and (
                    len(self.pending) >= self.config["queue_depth"]
                    or self.pending_bytes + size > self.config["queue_bytes"]
                ):
                    _, _, removed, _ = self.pending.popleft()
                    self.pending_bytes -= removed
                    self.dropped += 1
                self.pending.append((sample, time.monotonic(), size, self.received))
                self.pending_bytes += size
                self.lock.notify_all()
        except Exception as exc:
            with self.lock:
                self.error = f"Rejected ROS message: {exc}"
                self.lock.notify_all()

    def poll(self, timeout):
        if self.config["role"] != "read":
            raise ValueError("Not a ROS subscription")
        end = time.monotonic() + timeout
        with self.lock:
            while not self.pending and self.error is None and not self._closed:
                remaining = end - time.monotonic()
                if remaining <= 0:
                    break
                self.lock.wait(min(remaining, 0.05))
            if self.error is not None:
                raise RuntimeError(self.error)
            age_limit = self.config["max_age_seconds"]
            while (
                self.pending and age_limit > 0 and time.monotonic() - self.pending[0][1] > age_limit
            ):
                _, _, size, _ = self.pending.popleft()
                self.pending_bytes -= size
                self.dropped += 1
            if self.config["consume"] == "latest":
                while len(self.pending) > 1:
                    _, _, size, _ = self.pending.popleft()
                    self.pending_bytes -= size
                    self.dropped += 1
            sample, sequence, age = None, None, None
            if self.pending:
                sample, arrival, size, sequence = self.pending.popleft()
                self.pending_bytes -= size
                age = time.monotonic() - arrival
            return {
                "sample": sample,
                "info": {
                    **self.info,
                    "receive_sequence": sequence,
                    "received": self.received,
                    "dropped": self.dropped,
                    "queued_bytes": self.pending_bytes,
                    "local_age_seconds": age,
                    "matched_publishers": self.entity.get_publisher_count(),
                },
            }

    def _make_message(self, value, stamp_mode):
        message = self.message_class()
        if self.config["kind"] in ("image", "cloud"):
            packet = (ROSImage if self.config["kind"] == "image" else ROSPointCloud).from_wire(
                value
            )
            message.height, message.width = packet.height, packet.width
            message.is_bigendian = packet.is_bigendian
            if self.config["kind"] == "image":
                message.encoding, message.step = packet.encoding, packet.step
            else:
                self.set_fields(message, {"fields": packet.fields})
                message.point_step, message.row_step = packet.point_step, packet.row_step
                message.is_dense = packet.is_dense
            message.data = array.array("B", packet.data)
            header = header_value(packet.header)
            self.set_fields(message.header, header)
        else:
            if not isinstance(value, dict):
                raise ValueError("ROS message fields must be an object")
            json_bytes(value, MAX_DATA_BYTES)
            self.set_fields(message, value)
            if not preserved(value, self.to_dict(message)):
                raise ValueError("ROS message conversion changed a field's type/range/value")
        if stamp_mode == "node_clock":
            if not hasattr(message, "header"):
                raise ValueError("Node clock stamping requires a message header")
            message.header.stamp = self.node.get_clock().now().to_msg()
        return message

    def publish(self, value, timeout, require_subscriber, wait_for_ack, stamp_mode):
        if self.config["role"] != "write":
            raise ValueError("Not a ROS publisher")
        message = self._make_message(value, stamp_mode)  # Validate before waiting or writing.
        header = self.to_dict(message.header) if hasattr(message, "header") else None
        end = time.monotonic() + timeout
        count = self.entity.get_subscription_count()
        while require_subscriber and not count and time.monotonic() < end:
            time.sleep(0.01)
            count = self.entity.get_subscription_count()
        if require_subscriber and not count:
            return {"published": False, "attempted": False, "matched_subscribers": 0}
        try:
            self.entity.publish(message)
            acknowledged = False
            if wait_for_ack and self.config["reliability"] == "reliable" and count:
                acknowledged = bool(self.entity.wait_for_all_acked(self.Duration(seconds=timeout)))
                if not acknowledged:
                    raise RuntimeError("DDS acknowledgment timed out after publication")
        except Exception as exc:
            raise PublishUncertain(
                f"Publish outcome unknown; inspect the receiver before reset: {exc}"
            ) from exc
        return {
            **self.info,
            "published": True,
            "attempted": True,
            "matched_subscribers": count,
            "dds_acknowledged": acknowledged,
            "application_acknowledged": False,
            "header": header,
        }

    def close(self):
        if self._closed:
            return
        self._closed = True
        with self.lock:
            self.lock.notify_all()
        if self.executor is not None:
            self.executor.shutdown(timeout_sec=1.0)
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.node is not None:
            self.node.destroy_node()
        if self.context is not None:
            self.context.try_shutdown()
