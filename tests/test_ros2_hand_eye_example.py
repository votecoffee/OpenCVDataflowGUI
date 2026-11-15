"""Retained rendered calibration, independent review and preservation checks."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from opencv_dataflow.assistant_api import ProjectSession, make_registry, validate_graph
from opencv_dataflow.payloads import NO_VALUE
from opencv_dataflow.pointclouds import PointCloud

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/industrial integration/ros2_hand_eye_calibration"
PLUGIN = EXAMPLE / "hand_eye_review.py"
spec = importlib.util.spec_from_file_location("hand_eye_review_test", PLUGIN)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@unittest.skipUnless(importlib.util.find_spec("scipy"), "Optional SciPy refinement dependency")
class HandEyeExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = make_registry([PLUGIN])
        cls.session = ProjectSession.open(EXAMPLE / "ros2_hand_eye_refined.json", cls.registry)
        cls.receipt = cls.session.run()
        cls.fit = cls.session.graph.blocks["refine"].last_result.outputs
        cls.review = cls.session.graph.blocks["review"].last_result.outputs
        cls.samples = (
            cls.session.graph.blocks["captures"]
            .graph.blocks["samples"]
            .last_result.outputs["samples"]
        )

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    def test_complete_rendered_workflow_and_native_cloud(self):
        self.assertTrue(validate_graph(self.session.graph)["valid"])
        report = self.review["report"]
        self.assertTrue(report["passed"], report["failed_gates"])
        self.assertEqual([report["training_views"], report["held_out_views"]], [20, 5])
        self.assertTrue(all(s["corner_count"] == 24 for s in report["samples"]))
        self.assertLess(report["mount_error_mm"], 2.0)
        self.assertLess(report["held_out_max_reprojection_rms_px"], 1.0)
        self.assertIsInstance(self.review["cloud"], PointCloud)
        self.assertEqual(self.review["cloud"].colors.shape, self.review["cloud"].points.shape)
        self.assertEqual(self.session.graph.blocks["review"].preview_output, "cloud")
        self.assertEqual(self.fit["report"]["held_out_views_used"], 0)

    def test_failed_park_baseline_is_retained(self):
        with ProjectSession.open(EXAMPLE / "ros2_hand_eye_diverse.json", self.registry) as session:
            session.run()
            report = session.graph.blocks["review"].last_result.outputs["report"]
            self.assertFalse(report["passed"])
            self.assertIn("mount_error_mm", report["failed_gates"])

    def test_save_as_copies_all_captures_and_reopens(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "copy.json"
            self.session.save(path)
            self.assertEqual(
                len(list((Path(directory) / "captures_diverse").glob("*/image.png"))), 25
            )
            with ProjectSession.open(path, self.registry) as reopened:
                reopened.run()
                actual = reopened.graph.blocks["refine"].last_result.outputs["camera_to_gripper"]
                np.testing.assert_allclose(
                    actual["source_to_target"],
                    self.fit["camera_to_gripper"]["source_to_target"],
                    atol=1e-10,
                )

    def test_evaluation_truth_does_not_change_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "copy.json"
            with ProjectSession.open(
                EXAMPLE / "ros2_hand_eye_refined.json", self.registry
            ) as original:
                original.save(path)
            truth_path = Path(directory) / "captures_diverse/evaluation_only.json"
            truth = json.loads(truth_path.read_text())
            truth["tool0_from_camera_optical"][0][3] += 0.05
            truth_path.write_text(json.dumps(truth))
            with ProjectSession.open(path, self.registry) as altered:
                altered.run()
                result = altered.graph.blocks["refine"].last_result.outputs
                np.testing.assert_allclose(
                    result["camera_to_gripper"]["source_to_target"],
                    self.fit["camera_to_gripper"]["source_to_target"],
                    atol=1e-10,
                )
                self.assertFalse(
                    altered.graph.blocks["review"].last_result.outputs["report"]["passed"]
                )

    def test_refinement_rejects_held_out_input(self):
        with self.assertRaisesRegex(ValueError, "training captures only"):
            module.refine_mount(
                self.samples[-5:], self.fit["camera_to_gripper"], self.fit["target_to_base"]
            )

    def test_mismatched_stamp_is_rejected_even_with_updated_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = EXAMPLE / "captures_diverse"
            manifest = json.loads((source / "manifest.json").read_text())
            record = json.loads((source / manifest["captures"][0]["path"]).read_text())
            record["robot_transform"]["header"]["stamp"]["nanosec"] += 1
            manifest["captures"][0]["sha256"] = module.canonical(record)
            target = root / manifest["captures"][0]["path"]
            target.parent.mkdir(parents=True)
            target.write_text(json.dumps(record))
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "exact acquisition time"):
                module.load_sample(root / "manifest.json", 0)

    def test_missing_and_corrupt_capture_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = EXAMPLE / "captures_diverse"
            manifest = json.loads((source / "manifest.json").read_text())
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(FileNotFoundError):
                module.load_sample(root / "manifest.json", 0)
            record = json.loads((source / manifest["captures"][0]["path"]).read_text())
            target = root / manifest["captures"][0]["path"]
            target.parent.mkdir(parents=True)
            target.write_text(json.dumps(record))
            target.with_name("image.png").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "image checksum"):
                module.load_sample(root / "manifest.json", 0)

    def test_graphics_off_preserves_calibration(self):
        with ProjectSession.open(EXAMPLE / "ros2_hand_eye_refined.json", self.registry) as session:
            session.graph.set_graphics_mode("off")
            session.run()
            result = session.graph.blocks["review"].last_result.outputs
            self.assertIs(result["overlay"], NO_VALUE)
            self.assertTrue(result["report"]["passed"])
            np.testing.assert_allclose(
                session.graph.blocks["refine"].last_result.outputs["camera_to_gripper"][
                    "source_to_target"
                ],
                self.fit["camera_to_gripper"]["source_to_target"],
                atol=1e-10,
            )


if __name__ == "__main__":
    unittest.main()
