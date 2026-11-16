"""Local image/data demo source, owned by the saved Dataflow program. Code: MIT."""

import argparse
import array
import json
import os
import sys
import threading
import uuid
from pathlib import Path

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--domain", type=int, default=71)
    args = parser.parse_args()
    # Original photo was converted to this bounded RGB PPM without pixel edits.
    magic, dimensions, maximum, pixels = args.image.read_bytes().split(b"\n", 3)
    width, height = map(int, dimensions.split())
    if magic != b"P6" or maximum != b"255" or len(pixels) != width * height * 3:
        raise ValueError("Expected an RGB8 PPM fixture")
    stopped = threading.Event()

    def watch_owner():
        # The program closes standard input during graceful service shutdown.
        while os.read(sys.stdin.fileno(), 1) not in (b"", b"\n"):
            pass
        stopped.set()

    threading.Thread(target=watch_owner, daemon=True).start()
    context = Context()
    node = executor = None
    try:
        rclpy.init(args=[], context=context, domain_id=args.domain)
        if rclpy.get_rmw_implementation_identifier() != "rmw_fastrtps_cpp":
            raise RuntimeError("This demo requires Fast DDS")
        node = rclpy.create_node("dataflow_program_demo_" + uuid.uuid4().hex[:8], context=context)
        images = node.create_publisher(
            Image,
            "/opencv_dataflow/program_demo/image",
            QoSProfile(depth=4, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        data = node.create_publisher(String, "/opencv_dataflow/program_demo/data", 4)
        sequence = 0

        def publish():
            nonlocal sequence
            sequence += 1
            image = Image()
            image.header.stamp = node.get_clock().now().to_msg()
            image.header.frame_id = "demo_camera_optical"
            image.width, image.height = width, height
            image.encoding, image.step = "rgb8", width * 3
            image.data = array.array("B", pixels)
            images.publish(image)
            message = String()
            message.data = json.dumps(
                {"sequence": sequence, "part": "VisA PCB2 0000", "source": "program-owned demo"}
            )
            data.publish(message)

        node.create_timer(0.1, publish)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        print(f"ROS_DEMO_READY pid={os.getpid()} rmw=rmw_fastrtps_cpp", flush=True)
        while not stopped.is_set() and context.ok():
            executor.spin_once(timeout_sec=0.05)
    finally:
        if executor is not None:
            executor.shutdown(timeout_sec=1)
        if node is not None:
            node.destroy_node()
        if context.ok():
            context.shutdown()
        print("ROS_DEMO_STOPPED", flush=True)


if __name__ == "__main__":
    main()
