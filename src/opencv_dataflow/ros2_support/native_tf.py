"""Persistent TF2 in the trusted ROS Python, with explicit query and clock epochs."""

import os
import threading
import time
import uuid
from importlib.metadata import version

try:
    from .native import plain
    from .wire import header_value
except ImportError:
    from native import plain
    from wire import header_value


class TFClockChanged(RuntimeError):
    pass


class TFUnavailable(RuntimeError):
    pass


class TFEndpoint:
    def __init__(self, config):
        self.context = self.node = self.executor = self.thread = self.listener = self.jump = None
        self.closed, self.error, self.epoch = False, None, 0
        self.lock = threading.RLock()
        self.config = config
        if (
            os.environ.get("ROS_DISTRO") != "lyrical"
            or os.environ.get("RMW_IMPLEMENTATION") != "rmw_fastrtps_cpp"
        ):
            raise RuntimeError("TF worker requires Lyrical and rmw_fastrtps_cpp")
        if config["kind"] != "tf" or config["role"] != "lookup":
            raise ValueError("Invalid TF worker configuration")
        import rclpy
        from rclpy.clock import JumpThreshold
        from rclpy.clock_type import ClockType
        from rclpy.context import Context
        from rclpy.duration import Duration
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.parameter import Parameter
        from rclpy.time import Time
        from rosidl_runtime_py.convert import message_to_ordereddict
        from tf2_ros import Buffer, TransformListener

        self.Time, self.Duration, self.ClockType = Time, Duration, ClockType
        self.to_dict = lambda msg: plain(message_to_ordereddict(msg))
        try:
            self.context = Context()
            rclpy.init(args=[], context=self.context, domain_id=int(config["domain_id"]))
            identifier = rclpy.get_rmw_implementation_identifier()
            if identifier != "rmw_fastrtps_cpp":
                raise RuntimeError("Unexpected TF worker middleware")
            self.node = rclpy.create_node(
                "opencv_dataflow_tf_" + uuid.uuid4().hex[:12],
                context=self.context,
                namespace=config["namespace"],
                use_global_arguments=False,
                cli_args=["--ros-args", "-r", "/clock:=" + config["clock_topic"]],
                parameter_overrides=[Parameter("use_sim_time", value=config["use_sim_time"])],
            )
            # No auxiliary FrameGraph service. The node clock below owns reset detection.
            self.buffer = Buffer(cache_time=Duration(seconds=config["cache_seconds"]))
            self.listener = TransformListener(
                self.buffer,
                self.node,
                spin_thread=False,
                tf_topic=config["tf_topic"],
                tf_static_topic=config["tf_static_topic"],
            )
            self.jump = self.node.get_clock().create_jump_callback(
                JumpThreshold(
                    min_forward=None, min_backward=Duration(nanoseconds=-1), on_clock_change=True
                ),
                post_callback=self._clock_jump,
            )
            self.info = {
                "ros_distro": "lyrical",
                "rclpy": version("rclpy"),
                "rmw": identifier,
                "domain_id": config["domain_id"],
                "node": self.node.get_fully_qualified_name(),
                "worker_id": uuid.uuid4().hex,
                "tf_epoch": self.epoch,
                "tf_topic": config["tf_topic"],
                "tf_static_topic": config["tf_static_topic"],
                "node_use_sim_time": config["use_sim_time"],
            }
            self.executor = SingleThreadedExecutor(context=self.context)
            self.executor.add_node(self.node)
            self.thread = threading.Thread(target=self._spin, name="ros2-tf-listener", daemon=True)
            self.thread.start()
        except BaseException:
            self.close()
            raise

    def _spin(self):
        try:
            self.executor.spin()
        except Exception as exc:
            if not self.closed:
                self.error = str(exc)

    def _clock_jump(self, _jump):
        with self.lock:
            self.epoch += 1
            self.buffer.clear()

    def _check(self, expected_epoch, query_ns, max_age, future_tolerance):
        if self.error:
            raise RuntimeError("TF executor stopped: " + self.error)
        if expected_epoch != self.epoch:
            raise TFClockChanged("TF clock changed; reset flow state or supply a new run context")
        if max_age == 0:
            return None
        now = self.node.get_clock().now().nanoseconds
        if now <= 0:
            raise ValueError("TF age checking requires a nonzero node clock")
        age = (now - query_ns) / 1e9
        if age > max_age:
            raise ValueError("Acquisition is too old for the configured TF freshness limit")
        if age < -future_tolerance:
            raise ValueError("Acquisition stamp is ahead of the configured TF clock")
        return age

    def lookup(self, target, source, stamp, timeout, expected_epoch, max_age, future_tolerance):
        stamp = header_value({"stamp": stamp, "frame_id": source})["stamp"]
        query_ns = stamp["sec"] * 1_000_000_000 + stamp["nanosec"]
        if query_ns <= 0:
            raise ValueError(
                "TF lookup requires a positive acquisition stamp; zero/latest is forbidden"
            )
        if (
            not 0 < timeout <= 30
            or not 0 <= max_age <= 3600
            or not 0 <= future_tolerance <= 30
            or type(expected_epoch) is not int
        ):
            raise ValueError("Invalid bounded TF query settings")
        requested = self.Time(nanoseconds=query_ns, clock_type=self.ClockType.ROS_TIME)
        end = time.monotonic() + timeout
        reason = "Transform not available"
        while True:
            with self.lock:
                age = self._check(expected_epoch, query_ns, max_age, future_tolerance)
                available, reason = self.buffer.can_transform(
                    target, source, requested, timeout=self.Duration(), return_debug_tuple=True
                )
                if available:
                    message = self.buffer.lookup_transform(target, source, requested)
                    original = self.to_dict(message.header)
                    actual = original["stamp"]["sec"] * 1_000_000_000 + original["stamp"]["nanosec"]
                    if actual not in (0, query_ns):
                        raise RuntimeError(
                            "TF returned an unexpected stamp; no latest fallback is permitted"
                        )
                    self._check(expected_epoch, query_ns, max_age, future_tolerance)
                    # This is the transform sampled at query time, including timeless static chains.
                    message.header.stamp = requested.to_msg()
                    return {
                        "message": self.to_dict(message),
                        "info": {
                            **self.info,
                            "tf_epoch": self.epoch,
                            "requested_stamp": stamp,
                            "tf2_header": original,
                            "stamp_semantics": "query_time_not_source_measurement",
                            "source_sample_times_available": False,
                            "query_mode": "exact",
                            "age_seconds": age,
                            "age_checked": max_age > 0,
                            "target_frame": target,
                            "source_frame": source,
                        },
                    }
            remaining = end - time.monotonic()
            if remaining <= 0:
                raise TFUnavailable(str(reason)[:1000])
            time.sleep(min(0.01, remaining))

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.jump is not None:
            self.jump.unregister()
        if self.executor is not None:
            self.executor.shutdown(timeout_sec=1)
        if self.thread is not None:
            self.thread.join(timeout=2)
        if self.listener is not None:
            self.listener.unregister()
        if self.node is not None:
            self.node.destroy_node()
        if self.context is not None:
            self.context.try_shutdown()
