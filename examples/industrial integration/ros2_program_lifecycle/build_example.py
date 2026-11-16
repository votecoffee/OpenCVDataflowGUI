"""Build self-starting ROS image/data programs from an attributed real PCB photo."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

HERE = Path(__file__).resolve().parent
PHOTO = HERE.parent.parent / "assets/showcase/visa-pcb2-0000.png"


def build():
    image = cv2.imdecode(np.frombuffer(PHOTO.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ppm = f"P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n".encode() + rgb.tobytes()
    (HERE / "pcb.ppm").write_bytes(ppm)
    registry = builtin_registry()
    for case in ("image", "data"):
        graph = Graph()
        graph.set_program(
            {
                "name": "Self-starting ROS " + case + " demo",
                "description": (
                    "Starts a private local ROS publisher on first run; closes it with the program."
                ),
                "enabled": True,
                "variables": {"cycles": 0},
                "scripts": {
                    "startup": "print('Starting the local ROS demo source')",
                    "before_run": "state['cycles'] += 1",
                    "after_run": "state['last_status'] = context['status']",
                    "shutdown": "print('Closing the local ROS demo source')",
                },
                "services": [
                    {
                        "id": "ros_demo",
                        "runtime": "ros2",
                        "argv": [
                            "{python}",
                            "-u",
                            "{project_dir}/demo_publisher.py",
                            "--image",
                            "{project_dir}/pcb.ppm",
                        ],
                        "environment": {
                            "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
                            "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST",
                            "ROS_STATIC_PEERS": "",
                        },
                        "ready_text": "ROS_DEMO_READY",
                        "shutdown_timeout_seconds": 5,
                    }
                ],
                "assets": ["demo_publisher.py", "pcb.ppm", "SOURCES.json"],
            }
        )
        source = graph.add(
            registry.create(
                "input.ros2_" + case,
                block_id="source",
                parameters={
                    "topic": "/opencv_dataflow/program_demo/" + case,
                    "enabled": True,
                    "timeout_seconds": 3.0,
                    **(
                        {
                            "data_mode": "json_string",
                            "ticket_mode": "field",
                            "ticket_field": "sequence",
                        }
                        if case == "data"
                        else {"ticket_mode": "header_stamp"}
                    ),
                },
            )
        )
        source.position = [0, 0]
        if case == "image":
            gray = graph.add(registry.create("cv.gray", block_id="gray"))
            canny = graph.add(registry.create("cv.canny", block_id="edges"))
            gray.position, canny.position = [320, 0], [640, 0]
            graph.connect("source", "image", "gray", "image")
            graph.connect("gray", "image", "edges", "image")
        note = graph.add(registry.create("layout.comment", block_id="instructions"))
        note.name = "Run once starts ROS; closing this program stops it"
        note.position, note.node_size = [0, 400], "large"
        note.description = (
            "Configure the local Lyrical runtime once, then Run once or Continuous. This program "
            "starts its own localhost Fast DDS image/data publisher in domain 71. Stop pauses "
            "graph cycles; the publisher remains until New/Open, Stop program services, "
            "or application close. "
            "Program > Settings and scripts shows hooks, service state and diagnostics. No robot, "
            "simulator or hardware connection. Image: Yang Zou, Jongheon Jeong, Latha Pemula, "
            "Dongqing Zhang and Onkar Dabeer, VisA / SPot-the-Difference, ECCV 2022, CC BY 4.0. "
            "https://registry.opendata.aws/visa/ . Reuses PCB2 0000, resized/letterboxed by the "
            "showcase fixture, converted losslessly to RGB PPM here. Code and graph: MIT."
        )
        graph.save(HERE / f"managed_{case}.json")
        graph.close()
    (HERE / "SOURCES.json").write_bytes(
        (
            json.dumps(
                {
                    "creator": (
                        "Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang and Onkar Dabeer"
                    ),
                    "dataset": "VisA / SPot-the-Difference, ECCV 2022; PCB2 0000",
                    "license": "CC BY 4.0",
                    "license_url": "https://creativecommons.org/licenses/by/4.0/",
                    "source": "https://registry.opendata.aws/visa/",
                    "prior_record": "../../assets/showcase/sources.json",
                    "input_sha256": hashlib.sha256(PHOTO.read_bytes()).hexdigest(),
                    "output_sha256": hashlib.sha256(ppm).hexdigest(),
                    "changes": (
                        "Existing resized/letterboxed showcase PNG converted to lossless RGB8 PPM; "
                        "no further pixel edits."
                    ),
                },
                indent=2,
            )
            + "\n"
        ).encode()
    )


if __name__ == "__main__":
    build()
