"""Read-only graph discovery and receipt observers in the trusted ROS runtime."""

import os
import threading
import time
import uuid


class HealthEndpoint:
    def __init__(self, config):
        self.context = self.node = self.executor = self.thread = None
        self.closed, self.error = False, None
        self.lock = threading.Lock()
        self.config, self.topics = config, {}
        if (
            os.environ.get("ROS_DISTRO") != "lyrical"
            or os.environ.get("RMW_IMPLEMENTATION") != "rmw_fastrtps_cpp"
        ):
            raise RuntimeError("Health worker requires Lyrical and rmw_fastrtps_cpp")
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from rosidl_runtime_py.utilities import get_message

        try:
            self.context = Context()
            rclpy.init(args=[], context=self.context, domain_id=int(config["domain_id"]))
            if rclpy.get_rmw_implementation_identifier() != "rmw_fastrtps_cpp":
                raise RuntimeError("Unexpected health worker middleware")
            self.node = rclpy.create_node(
                "opencv_dataflow_health_" + uuid.uuid4().hex[:12],
                context=self.context,
                namespace=config["namespace"],
                use_global_arguments=False,
            )
            qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE
                if config["reliability"] == "reliable"
                else ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.TRANSIENT_LOCAL
                if config["durability"] == "transient_local"
                else DurabilityPolicy.VOLATILE,
            )
            for row in config["topics"]:
                state = {"last_received": None, "received": 0, "error": None, "entity": None}
                self.topics[row["name"]] = state

                def receive(_message, state=state):
                    with self.lock:
                        state["last_received"] = time.monotonic()
                        state["received"] += 1

                try:
                    state["entity"] = self.node.create_subscription(
                        get_message(row["type"]), row["name"], receive, qos
                    )
                except Exception as exc:
                    state["error"] = type(exc).__name__
            self.executor = SingleThreadedExecutor(context=self.context)
            self.executor.add_node(self.node)
            self.thread = threading.Thread(target=self.spin, name="ros2-health", daemon=True)
            self.thread.start()
            self.info = {
                "node": self.node.get_fully_qualified_name(),
                "domain_id": config["domain_id"],
                "rmw": "rmw_fastrtps_cpp",
            }
        except BaseException:
            self.close()
            raise

    def spin(self):
        try:
            self.executor.spin()
        except Exception as exc:
            if not self.closed:
                self.error = str(exc)

    def snapshot(self):
        if self.error or not self.context.ok() or not self.thread.is_alive():
            raise ConnectionError("ROS observer executor is not running")
        nodes = {
            namespace.rstrip("/") + "/" + name
            for name, namespace in self.node.get_node_names_and_namespaces()
        }
        services = dict(self.node.get_service_names_and_types())
        rows = []

        def add(name, ok, message, critical=False, **details):
            rows.append(
                {
                    "name": name,
                    "state": "ok" if ok else "alarm" if critical else "warning",
                    "message": message,
                    **details,
                }
            )

        for name in self.config["required_nodes"]:
            found = name in nodes
            add(name, found, "Node discovered" if found else "Required node not discovered", True)
        for row in self.config["required_services"]:
            found = row["type"] in services.get(row["name"], [])
            add(
                row["name"],
                found,
                "Service discovered (not called)"
                if found
                else "Required service/type not discovered",
                True,
                service_called=False,
            )
        for row in self.config["topics"]:
            state = self.topics[row["name"]]
            with self.lock:
                last, received = state["last_received"], state["received"]
            age = None if last is None else max(0, time.monotonic() - last)
            count = state["entity"].get_publisher_count() if state["entity"] else 0
            minimum, limit = row.get("minimum_publishers", 1), row.get("max_age_seconds", 0)
            ok = (
                not state["error"]
                and count >= minimum
                and (limit == 0 or (age is not None and age <= limit))
            )
            message = (
                "Observer interface unavailable"
                if state["error"]
                else "Too few type/QoS-compatible publishers"
                if count < minimum
                else "No recent observer receipt"
                if not ok
                else "Topic observer ready"
            )
            add(
                row["name"],
                ok,
                message,
                matched_publishers=count,
                received=received,
                receipt_age_seconds=age,
                max_age_seconds=limit,
                error_type=state["error"],
            )
        return {
            "connected": True,
            "message": "ROS participant and observer executor ready",
            "resources": rows,
            **self.info,
        }

    def health(self, timeout):
        end = time.monotonic() + min(30, max(0, float(timeout)))
        while True:
            value = self.snapshot()
            if all(r["state"] == "ok" for r in value["resources"]) or time.monotonic() >= end:
                return value
            time.sleep(min(0.02, max(0, end - time.monotonic())))

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.executor is not None:
            self.executor.shutdown(timeout_sec=1)
        if self.thread is not None:
            self.thread.join(timeout=2)
        if self.node is not None:
            self.node.destroy_node()
        if self.context is not None:
            self.context.try_shutdown()
