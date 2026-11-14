"""ROS plan-only bridge. No execution action/client or robot command publisher exists."""

import time
import uuid
from collections import deque

from .contracts import (
    decode,
    encoded,
    identifier,
    moveit_request,
    number,
    request,
    scene_digest,
    trajectory_preview,
)


def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)) or type(value).__module__ == "array":
        return [plain(v) for v in value]
    if callable(getattr(value, "tolist", None)):
        return plain(value.tolist())
    return value


def make_node():
    # ROS extensions are imported only by the ROS entry point, never by the app/core.
    from moveit_msgs.msg import DisplayTrajectory
    from moveit_msgs.srv import GetMotionPlan, GetPlanningScene
    from rclpy.node import Node
    from rosidl_runtime_py.convert import message_to_ordereddict
    from rosidl_runtime_py.set_message import set_message_fields
    from std_msgs.msg import String

    class PlanBridge(Node):
        def __init__(self):
            super().__init__("opencv_dataflow_planning")
            defaults = {
                "request_topic": "/simple_tools/planning/request",
                "result_topic": "/simple_tools/planning/result",
                "preview_topic": "/simple_tools/planning/display_trajectory",
                "scene_service": "/get_planning_scene",
                "plan_service": "/plan_kinematic_path",
                "robot_revision": "unconfigured",
                "max_wait_seconds": 30.0,
            }
            parameters = {
                key: self.declare_parameter(key, value).value for key, value in defaults.items()
            }
            self.robot_revision = identifier(parameters["robot_revision"])
            self.max_wait = number(parameters["max_wait_seconds"], 0.1, 120)
            self.server_id = str(uuid.uuid4())
            self.run_id = None
            self.job = None
            self.seen = deque(maxlen=128)
            self.results = self.create_publisher(String, parameters["result_topic"], 10)
            self.previews = self.create_publisher(DisplayTrajectory, parameters["preview_topic"], 1)
            self.scene_client = self.create_client(GetPlanningScene, parameters["scene_service"])
            self.plan_client = self.create_client(GetMotionPlan, parameters["plan_service"])
            self.subscription = self.create_subscription(
                String, parameters["request_topic"], self.receive, 1
            )
            self.timer = self.create_timer(0.02, self.advance)

        def emit(self, item, status, **details):
            correlation = {}
            for name in ("request_id", "run_id"):
                value = item.get(name)
                try:
                    correlation[name] = identifier(value)
                except ValueError:
                    correlation[name] = None
            result = {
                "schema": "opencv_dataflow_plan_result_v1",
                "server_id": self.server_id,
                **correlation,
                "status": status,
                "robot_revision": self.robot_revision,
                "robot_revision_source": "operator_configuration_assertion",
                "execution_authorized": False,
                **details,
            }
            self.results.publish(String(data=encoded(result).decode()))

        def receive(self, message):
            item = {}
            try:
                item = request(decode(message.data))
                op = item["operation"]
                if op != "snapshot" and item["server_id"] != self.server_id:
                    self.emit(item, "server_changed")
                    return
                if op == "cancel":
                    if (
                        self.job
                        and self.job["item"]["request_id"] == item["request_id"]
                        and self.job["item"]["run_id"] == item["run_id"]
                    ):
                        self.discard("cancelled")
                    else:
                        self.emit(item, "not_active")
                    return
                if self.job:
                    self.emit(item, "busy", draining=bool(self.job.get("discarded")))
                    return
                if op == "reset":
                    self.run_id = item["run_id"]
                    self.emit(item, "reset")
                    return
                if self.run_id is not None and self.run_id != item["run_id"]:
                    self.emit(item, "run_changed")
                    return
                if (item["run_id"], item["request_id"]) in self.seen:
                    self.emit(item, "duplicate_request")
                    return
                if op == "plan" and (
                    self.robot_revision == "unconfigured"
                    or item["robot_revision"] != self.robot_revision
                ):
                    self.emit(item, "robot_changed")
                    return
                if op == "plan" and item["plan"]["allowed_planning_time"] >= self.max_wait:
                    raise ValueError(
                        "Planner budget must be smaller than the overall bridge deadline"
                    )
                self.run_id = item["run_id"]
                self.seen.append((item["run_id"], item["request_id"]))
                self.job = {
                    "item": item,
                    "stage": "before",
                    "future": None,
                    "started": time.monotonic(),
                }
            except Exception as error:
                self.emit(
                    item if isinstance(item, dict) else {},
                    "invalid_request",
                    reason=str(error)[:1024],
                )

        def discard(self, status):
            job = self.job
            if job.get("discarded"):
                return
            self.emit(
                job["item"],
                status,
                stage=job["stage"],
                wall_s=time.monotonic() - job["started"],
                remote_service_cancellation=False,
            )
            # A ROS service has no action cancellation. Keep at most one unresolved
            # request and discard its eventual response rather than retry/overlap it.
            job["discarded"] = status
            if job["future"] is None:
                self.job = None

        def advance(self):
            job = self.job
            if not job:
                return
            if not job.get("discarded") and time.monotonic() - job["started"] >= self.max_wait:
                self.discard("timed_out")
                return
            if job.get("discarded"):
                if job["future"] is None or job["future"].done():
                    self.job = None
                return
            try:
                if job["future"] is None:
                    client = self.plan_client if job["stage"] == "plan" else self.scene_client
                    if not client.service_is_ready():
                        return
                    if job["stage"] == "plan":
                        value = GetMotionPlan.Request()
                        set_message_fields(value, moveit_request(job["item"]["plan"], job["scene"]))
                    else:
                        value = GetPlanningScene.Request()
                        value.components.components = 1023
                    job["future"] = client.call_async(value)
                    return
                if not job["future"].done():
                    return
                message = job["future"].result()
                if message is None:
                    raise RuntimeError("ROS service returned no response")
                value = plain(message_to_ordereddict(message))
                job["future"] = None
                item = job["item"]
                if job["stage"] == "before":
                    scene = value["scene"]
                    digest = scene_digest(scene)
                    if item["operation"] == "snapshot":
                        self.emit(
                            item,
                            "snapshot",
                            scene_revision=digest,
                            robot_model_name=scene["robot_model_name"],
                            robot_state=scene["robot_state"],
                            world_object_count=len(scene["world"]["collision_objects"]),
                            state_age_verified=False,
                        )
                        self.job = None
                    elif digest != item["scene_revision"]:
                        self.emit(item, "scene_changed", scene_revision=digest)
                        self.job = None
                    else:
                        moveit_request(item["plan"], scene)
                        job.update(scene=scene, digest=digest, stage="plan")
                elif job["stage"] == "plan":
                    metrics, response = trajectory_preview(item["plan"], value)
                    if response is None:
                        self.emit(item, **metrics, wall_s=time.monotonic() - job["started"])
                        self.job = None
                    else:
                        job.update(metrics=metrics, response=response, stage="after")
                else:
                    if scene_digest(value["scene"]) != job["digest"]:
                        self.emit(item, "scene_changed", reason="Scene differs after planning")
                    else:
                        response = job["response"]
                        preview = DisplayTrajectory()
                        set_message_fields(
                            preview,
                            {
                                "model_id": job["scene"]["robot_model_name"],
                                "trajectory_start": response["trajectory_start"],
                                "trajectory": [response["trajectory"]],
                            },
                        )
                        self.previews.publish(preview)
                        self.emit(
                            item,
                            **job["metrics"],
                            scene_revision=job["digest"],
                            wall_s=time.monotonic() - job["started"],
                            scene_consistency="equal_before_and_after_only",
                            continuous_scene_lock=False,
                        )
                    self.job = None
            except Exception as error:
                self.emit(
                    job["item"], "invalid_result", reason=str(error)[:1024], stage=job["stage"]
                )
                self.job = None

        def destroy_node(self):
            if self.job and self.job["future"] is not None:
                self.job["future"].cancel()
            self.job = None
            return super().destroy_node()

    return PlanBridge()


def main(args=None):
    import rclpy

    rclpy.init(args=args)
    node = None
    try:
        node = make_node()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
