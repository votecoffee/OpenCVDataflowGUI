"""Metric board generation and private native launch contracts."""

import json
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from opencv_hand_eye_demo.prepare import prepare
from opencv_hand_eye_demo.run import environment
from opencv_hand_eye_demo.scene import add_camera, world_xml

EXAMPLE = (
    Path(__file__).resolve().parents[5]
    / "examples/industrial integration/ros2_hand_eye_calibration"
)


class SceneTests(unittest.TestCase):
    def test_board_rectangles_reproduce_identified_target(self):
        import cv2

        scene = json.loads((EXAMPLE / "assets_diverse/scene.json").read_text())
        actual = np.full((1900, 2500), 255, np.uint8)
        for x, y, w, h in scene["black_rectangles_m"]:
            left, top, width, height = np.rint(
                np.array([x + 0.02, y + 0.02, w, h]) / 0.0001
            ).astype(int)
            actual[top : top + height, left : left + width] = 0
        np.testing.assert_array_equal(
            actual, cv2.imread(str(EXAMPLE / "assets_diverse/target.png"), cv2.IMREAD_GRAYSCALE)
        )
        board = cv2.aruco.CharucoBoard(
            (7, 5), 0.03, 0.022, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        )
        _, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(actual)
        self.assertEqual(len(ids), 24)
        root = ET.fromstring(world_xml(scene))
        self.assertEqual(len(root.findall("world/model")), 2)

    def test_robot_camera_optical_axes_and_mass(self):
        scene = json.loads((EXAMPLE / "assets_diverse/scene.json").read_text())
        robot = ET.fromstring(add_camera('<robot name="test"><link name="tool0"/></robot>', scene))
        self.assertEqual(
            robot.find("link[@name='hand_eye_camera_link']/inertial/mass").get("value"), "0.02"
        )
        self.assertEqual(
            robot.find("joint[@name='hand_eye_camera_mount']/parent").get("link"), "tool0"
        )
        self.assertEqual(
            robot.find("gazebo/sensor/gz_frame_id").text, "hand_eye_camera_optical_frame"
        )

    def test_preparation_preserves_runtime_and_detects_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            (runtime / "launch").mkdir(parents=True)
            (runtime / "config").mkdir()
            for name in [
                "ur_sim_control_windows.launch.py",
                "ur_moveit_windows.launch.py",
                "gripper_description.py",
            ]:
                (runtime / "launch" / name).write_text(
                    "value = " + repr(runtime.as_posix() + "/generated") + "\n"
                )
            record = prepare(runtime, EXAMPLE / "assets_diverse/scene.json", root / "overlay")
            self.assertIn("package_source_sha256", record)
            for path in (runtime / "launch").glob("*.py"):
                self.assertIn(runtime.as_posix(), path.read_text())
            with self.assertRaises(ValueError):
                prepare(runtime, EXAMPLE / "assets_diverse/scene.json", root / "overlay")
            with self.assertRaises(ValueError):
                prepare(runtime, EXAMPLE / "assets_diverse/scene.json", runtime / "unsafe")
            _, values = environment(root / "overlay", "opencv_hand_eye_test")
            self.assertEqual(values["ROS_DOMAIN_ID"], "44")
            (root / "overlay/world.sdf").write_text("changed")
            with self.assertRaisesRegex(ValueError, "Prepared input changed"):
                environment(root / "overlay", "opencv_hand_eye_test")


if __name__ == "__main__":
    unittest.main()
