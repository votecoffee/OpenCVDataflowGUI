import hashlib
import math
from pathlib import Path
import tempfile
import unittest
from xml.etree import ElementTree as ET

from opencv_bin_picking_fixture.scene import default_scene, world_xml, add_assistance, bridge_config, camera_transform, validate
from opencv_bin_picking_fixture.state import FixtureState, Observation, axis_z
from opencv_bin_picking_fixture.prepare import prepare
from opencv_bin_picking_fixture.run import environment


class SceneTests(unittest.TestCase):
    def test_geometry_has_six_nonintersecting_dynamic_parts_and_real_collisions(self):
        scene = default_scene(); world = ET.fromstring(world_xml(scene)).find("world")
        parts = [model for model in world.findall("model") if model.findtext("static") == "false"]
        self.assertEqual(len(parts), 6)
        for model in parts:
            self.assertIsNotNone(model.find("link/collision/geometry/box/size"))
            self.assertGreater(float(model.findtext("link/inertial/mass")), 0)
        for a in scene["parts"]:
            for b in scene["parts"]:
                if a is b: continue
                self.assertTrue(any(abs(a["pose"][i] - b["pose"][i]) >= (a["size"][i] + b["size"][i])/2 for i in range(3)))
        self.assertAlmostEqual(scene["static_boxes"][1]["pose"][2] + scene["static_boxes"][1]["size"][2]/2, 0.2)
        self.assertGreater(min(p["pose"][2] - p["size"][2]/2 for p in scene["parts"]), 0.2)

    def test_source_camera_and_evaluation_are_separate(self):
        scene = default_scene(); root = ET.fromstring(world_xml(scene)); rows = bridge_config(scene)
        self.assertEqual(root.find("world/model[@name='fixture_camera']/link/sensor").attrib["type"], "rgbd_camera")
        sources = [row for row in rows if row["ros_topic_name"].startswith("/opencv_fixture/source/")]
        self.assertEqual(len(sources), 3)
        self.assertTrue(all(row["direction"] == "GZ_TO_ROS" for row in sources))
        self.assertEqual(sum("/evaluation/" in row["ros_topic_name"] for row in rows), 6)
        self.assertLess(axis_z(camera_transform(scene)["quaternion_xyzw"])[2], -0.999)

    def test_generated_robot_starts_unattached_and_preserves_parent_link(self):
        original = '<robot name="ur"><link name="gripper_base_link"/></robot>'
        root = ET.fromstring(add_assistance(original, default_scene()))
        plugins = root.findall("gazebo/plugin")
        self.assertEqual(len(plugins), 6)
        self.assertTrue(all(p.findtext("initial_attach") == "false" and p.findtext("child_link") == "body" for p in plugins))
        self.assertEqual(len({p.findtext("attach_topic") for p in plugins}), 6)
        self.assertEqual(root.findtext("gazebo[@reference='gripper_mount_base']/preserveFixedJoint"), "true")
        self.assertNotIn("plugin", original)

    def test_invalid_and_nonfinite_scene_inputs_are_rejected(self):
        for edit in [lambda s: s["parts"][0]["size"].__setitem__(0, -1),
                     lambda s: s["parts"][0]["pose"].__setitem__(0, float("nan")),
                     lambda s: s["parts"][1].__setitem__("id", s["parts"][0]["id"]),
                     lambda s: s.__setitem__("world", "../unsafe")]:
            scene = default_scene(); edit(scene)
            with self.assertRaises(ValueError): validate(scene)


class GateTests(unittest.TestCase):
    def fixture(self):
        clock = [100.0]
        state = FixtureState(default_scene(), now=lambda: clock[0])
        part = state.scene["parts"][0]
        state.poses[part["id"]] = Observation(tuple(part["pose"][:3]), (0, 0, 0, 1), 100, 1000)
        state.tcp = Observation((0.45, -0.12, part["pose"][2] + .015), (1, 0, 0, 0), 100, 1000)
        state.joints = {"gripper_left_finger_joint": .020, "gripper_right_finger_joint": .020}
        state.joints_received = 100; state.arm_stationary = True; state.camera_ready = True
        return state, clock

    def test_request_is_not_observed_attachment_and_only_one_part_can_be_held(self):
        state, _ = self.fixture(); operation = state.request("attach", "part_01")
        self.assertEqual(operation["outcome"], "requested")
        self.assertNotEqual(state.state["part_01"], "attached")
        with self.assertRaises(ValueError): state.request("attach", "part_02")
        state.acknowledge("part_01", "attached")
        self.assertIsNone(state.pending)
        with self.assertRaises(ValueError): state.request("attach", "part_02")
        state.request("detach", "part_01"); state.acknowledge("part_01", "detached")
        self.assertEqual(state.last_operation["outcome"], "observed")

    def test_stale_nan_moving_and_open_jaw_inputs_do_not_attach(self):
        for edit in [lambda s: setattr(s, "arm_stationary", False),
                     lambda s: setattr(s, "camera_ready", False),
                     lambda s: setattr(s, "joints_received", 90),
                     lambda s: s.joints.__setitem__("gripper_left_finger_joint", .028),
                     lambda s: setattr(s, "tcp", Observation((float("nan"), -.12, .23), (1, 0, 0, 0), 100, 1000)),
                     lambda s: setattr(s, "tcp", Observation((.45, -.12, .23), (1, 0, 0, 0), 90, 1000))]:
            state, _ = self.fixture(); edit(state)
            with self.assertRaises(ValueError): state.request("attach", "part_01")
            self.assertIsNone(state.pending)

    def test_unknown_command_outcome_never_becomes_false_or_automatically_retried(self):
        state, clock = self.fixture(); state.request("attach", "part_01")
        clock[0] = 103; state.expire()
        self.assertEqual(state.state["part_01"], "outcome_unknown")
        with self.assertRaises(ValueError): state.request("attach", "part_01")
        state.acknowledge("part_01", "attached")
        self.assertIsNone(state.pending)
        self.assertEqual(state.state["part_01"], "attached")


class PreparationTests(unittest.TestCase):
    def test_overlay_preserves_original_bytes_and_refuses_existing_or_shared_destinations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); runtime = root / "native"; (runtime / "launch").mkdir(parents=True)
            (runtime / "config").mkdir()
            for name in ("ur_sim_control_windows.launch.py", "ur_moveit_windows.launch.py", "gripper_description.py"):
                (runtime / "launch" / name).write_text(f"project = {runtime.as_posix()!r}\n", encoding="utf-8")
            (runtime / "config/test.yaml").write_bytes(b"untouched: true\r\n")
            dll = runtime / ".pixi/envs/default/Library/bin/gz-sim-detachable-joint-system.dll"
            dll.parent.mkdir(parents=True); dll.write_bytes(b"synthetic plugin marker: initial_attach")
            before = {p.relative_to(runtime).as_posix(): p.read_bytes() for p in runtime.rglob("*") if p.is_file()}
            output = root / "private_overlay"; receipt = prepare(runtime, output)
            self.assertEqual(receipt["scope"], "private additive simulation overlay; no simulator launched")
            self.assertTrue((output / "world.sdf").is_file())
            self.assertEqual((output / "config/test.yaml").read_bytes(), b"untouched: true\r\n")
            for name, data in before.items(): self.assertEqual((runtime/name).read_bytes(), data)
            for name, digest in receipt["copied_original_sha256"].items():
                self.assertEqual(hashlib.sha256(before[name]).hexdigest(), digest)
            self.assertIn(output.as_posix(), (output / "launch/ur_moveit_windows.launch.py").read_text(encoding="utf-8"))
            with self.assertRaises(ValueError): prepare(runtime, output)
            with self.assertRaises(ValueError): prepare(runtime, runtime / "nested_output")
            inherited = {"ROS_DOMAIN_ID": "42", "GZ_PARTITION": "ros2-sim2", "AMENT_PREFIX_PATH": "preserved-native-prefix"}
            _, actual = environment(output, "opencv_bin_pick_test", inherited)
            self.assertEqual(actual["ROS_DOMAIN_ID"], "130")
            self.assertEqual(actual["GZ_PARTITION"], "opencv_bin_pick_test")
            self.assertEqual(actual["AMENT_PREFIX_PATH"], inherited["AMENT_PREFIX_PATH"])
            self.assertEqual(inherited["ROS_DOMAIN_ID"], "42")
            with self.assertRaises(ValueError): environment(output, "ros2-sim2", inherited)


if __name__ == "__main__": unittest.main()
