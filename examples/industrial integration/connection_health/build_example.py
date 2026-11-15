"""Disabled real-protocol wiring and explicitly offline health status demonstrations."""

from pathlib import Path

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.plugins import load_plugin

HERE = Path(__file__).resolve().parent


def build():
    registry = builtin_registry()
    load_plugin(HERE / "plugin.py", registry)
    cases = {
        "ros2_disabled": (
            "industrial.ros2_health",
            {
                "required_nodes": ["/camera/driver"],
                "required_services": [
                    {
                        "name": "/camera/driver/get_parameters",
                        "type": "rcl_interfaces/srv/GetParameters",
                        "check_response": True,
                    }
                ],
                "topics": [
                    {"name": "/camera/image", "type": "sensor_msgs/msg/Image", "max_age_seconds": 2}
                ],
            },
        ),
        "mqtt_disabled": (
            "industrial.mqtt_health",
            {
                "host": "192.0.2.1",
                "subscriptions": ["cell/camera/status"],
                "require_message": True,
            },
        ),
        "ethernetip_disabled": ("industrial.ethernetip_health", {"route": "192.0.2.1"}),
        "logix_disabled": (
            "industrial.logix_health",
            {
                "route": "192.0.2.1/0",
                "tags": ["CameraReady", "AcquisitionCount"],
            },
        ),
        "fanuc_disabled": ("industrial.fanuc_health", {"robot_id": "robot", "registers": ["R[1]"]}),
        **{
            f"offline_{state}": (
                "example.recorded_connection_health",
                {
                    "recorded_state": state,
                    "fail_on_alarm": True,
                },
            )
            for state in ("ok", "warning", "alarm")
        },
    }
    for name, (kind, p) in cases.items():
        graph = Graph()
        monitor = graph.add(registry.create(kind, block_id="health", parameters=p))
        monitor.exposed_outputs = ["ok", "warning", "alarm", "report"]
        monitor.position = [420, 0]
        monitor.node_size = "large"
        note = graph.add(registry.create("layout.comment", block_id="scope"))
        note.position = [0, -100]
        note.node_size = "large"
        note.name = "Source and monitoring scope"
        note.description = (
            "Original data-only example by OpenCVDataflowGUI contributors (MIT). "
            "No customer data or images. Status is not a physical acceptance result. "
            + (
                "OFFLINE REPLAY: explicitly load this folder's trusted plugin.py first. "
                "It returns authored observations and performs no network I/O. "
                "Green=OK, yellow=resource warning with endpoint responsive, red=protocol alarm. "
                "This disconnected block runs without image inputs."
                if name.startswith("offline_")
                else "REAL-PROTOCOL TEMPLATE: monitoring disabled. Configure the intended endpoint "
                "and enable only when ready to make read-only probes. These documentation IPs "
                "are placeholders. ROS checks discovery, observer receipts and optional read-only "
                "empty GetParameters responses. FANUC needs a local Workbench profile. "
                "No outputs, parameters or motion are written."
            )
        )
        graph.save(HERE / f"{name}.json")
        graph.close()


if __name__ == "__main__":
    build()
