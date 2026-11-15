"""Export a native capture set losslessly, then author/replay the host example."""

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.assistant_api import ProjectSession, make_registry, validate_graph
from opencv_dataflow.toolblocks import ToolBlock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "hand_eye_example_review", HERE / "hand_eye_review.py"
)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def write(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )


def export(source, scene, output, robot_description=None):
    source = Path(source)
    output = Path(output)
    original = json.loads((source / "manifest.json").read_text())
    if not original["complete"] or len(original["captures"]) != 25:
        raise ValueError("Require a complete native 25-capture session")
    if hashlib.sha256(Path(scene).read_bytes()).hexdigest() != original["scene_sha256"]:
        raise ValueError("Evaluation scene does not match the captured workcell")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    chain = None
    if robot_description:
        sys.path.insert(
            0, str(HERE.parents[2] / "integrations/ros2/packages/opencv_dataflow_pick_backend")
        )
        from opencv_dataflow_pick_backend.kinematics import SerialChain

        chain = SerialChain(
            Path(robot_description).read_text(encoding="utf-8"), "base_link", "tool0"
        )
    manifest = copy.deepcopy(original)
    manifest["schema"] = "opencv.hand_eye_replay/v1"
    manifest["captures"] = []
    for entry in original["captures"]:
        record_raw = (source / entry["path"]).read_bytes()
        if hashlib.sha256(record_raw).hexdigest() != entry["sha256"]:
            raise ValueError("Native capture checksum differs")
        record = json.loads(record_raw)
        image = record["image"]
        raw = (source / entry["path"]).parent.joinpath(image["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != image["sha256"]:
            raise ValueError("Original image bytes changed")
        if image["encoding"] != "rgb8" or image["step"] != image["width"] * 3:
            raise ValueError(
                "This exporter expects tightly packed rgb8; do not silently drop padding"
            )
        rgb = np.frombuffer(raw, np.uint8).reshape(image["height"], image["width"], 3)
        ok, png = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        assert ok
        directory = output / record["id"]
        directory.mkdir()
        (directory / "image.png").write_bytes(png.tobytes())
        native_motion = record.pop("motion")
        write(directory / "motion.json", native_motion)
        image["source_bytes_sha256"] = image.pop("sha256")
        image["path"] = f"{record['id']}/image.png"
        image["png_sha256"] = hashlib.sha256(png.tobytes()).hexdigest()
        record["native_record_sha256"] = entry["sha256"]
        record["motion_path"] = f"{record['id']}/motion.json"
        record["motion_sha256"] = review.canonical(native_motion)
        if chain:
            feedback = record["joint_feedback"]
            _, skeleton = chain.forward(dict(zip(feedback["name"], feedback["position"])))
            record["robot_review_skeleton"] = skeleton.tolist()
            record["robot_review_kinematic_sha256"] = chain.sha256
        write(directory / "capture.json", record)
        manifest["captures"].append({**entry, "sha256": review.canonical(record)})
    write(output / "manifest.json", manifest)
    write(output / "evaluation_only.json", json.loads(Path(scene).read_text())["evaluation_only"])
    return manifest


def add(type_id, key, name, parameters=None):
    return dict(op="add", type=type_id, id=key, name=name, parameters=parameters or {})


def connect(a, out, b, inp):
    return dict(op="connect", source=a, output=out, target=b, input=inp)


def toolblock(registry, key, name, graph, outputs):
    definition = dict(
        toolblock_version=1,
        name=name,
        block_type="Calibration",
        subtype="ROS 2 hand-eye",
        graph=graph,
        inputs={},
        outputs={n: dict(block=b, port=p) for n, (b, p) in outputs.items()},
    )
    block = ToolBlock.from_definition(definition, registry, block_id=key)
    block.preview_output = "image" if "image" in outputs else ""
    item = block.to_project_item()
    block.close()
    return item


def build_project(destination, dataset="captures", refine=False):
    registry = make_registry([HERE / "hand_eye_review.py"])
    with ProjectSession(registry, directory=HERE) as captures:
        for i in range(25):
            key = f"view_{i + 1:02d}"
            role = "Train" if i < 20 else "HELD OUT"
            with ProjectSession(registry, directory=HERE) as sample:
                ops = [
                    add(
                        "example.hand_eye_capture",
                        "source",
                        "Saved image + exact-time robot pose",
                        {"index": i, "manifest_path": f"{dataset}/manifest.json"},
                    ),
                    add(
                        "calibration.charuco",
                        "corners",
                        "Identify 24 ChArUco corners",
                        {"minimum_corners": 12, "corner_refinement": "subpixel"},
                    ),
                    add(
                        "3d.pnp",
                        "target",
                        "Estimate board pose from pixels",
                        {
                            "source_frame": "charuco_board",
                            "ransac": False,
                            "minimum_inliers": 12,
                            "max_rms": 1.0,
                        },
                    ),
                    add("example.hand_eye_evidence", "evidence", "Retain paired observation"),
                    connect("source", "image", "corners", "image"),
                    connect("source", "camera", "corners", "camera"),
                    connect("corners", "image_points", "target", "image_points"),
                    connect("corners", "object_points", "target", "object_points"),
                    connect("source", "camera", "target", "camera"),
                    connect("source", "capture", "evidence", "capture"),
                    connect("source", "camera", "evidence", "camera"),
                    connect("source", "robot", "evidence", "robot"),
                    connect("target", "pose", "evidence", "target"),
                    connect("corners", "observation", "evidence", "observation"),
                    {"op": "arrange"},
                ]
                sample.apply(ops)
                item = toolblock(
                    registry,
                    key,
                    f"{role} {i + 1:02d}",
                    sample.graph.to_dict(),
                    {"sample": ("evidence", "sample"), "image": ("source", "image")},
                )
            item["position"] = [(i % 5) * 285, (i // 5) * 235]
            captures.apply([{"op": "insert", "block": item}])
        ops = [add("example.hand_eye_samples", "samples", "Frozen 20 training / 5 held-out split")]
        ops.extend(
            connect(f"view_{i + 1:02d}", "sample", "samples", f"sample_{i + 1}") for i in range(25)
        )
        captures.apply(ops)
        bundle = toolblock(
            registry,
            "captures",
            "Captured robot views (double-click to inspect)",
            captures.graph.to_dict(),
            {
                "robot": ("samples", "robot"),
                "target": ("samples", "target"),
                "samples": ("samples", "samples"),
                "training_samples": ("samples", "training_samples"),
                "image": ("view_01", "image"),
            },
        )
    with ProjectSession(registry, directory=HERE) as session:
        session.apply(
            [
                {"op": "insert", "block": bundle},
                add(
                    "calibration.hand_eye",
                    "fit",
                    "Fit camera mount — 20 teaching views",
                    {
                        "gripper_frame": "tool0",
                        "base_frame": "base_link",
                        "target_frame": "charuco_board",
                        "camera_frame": "hand_eye_camera_optical_frame",
                        "max_translation_rms_mm": 2.0,
                        "max_rotation_rms_degrees": 1.0,
                    },
                ),
                add(
                    "example.hand_eye_review",
                    "review",
                    "Check 5 unused views + reserved mount truth",
                    {"evaluation_path": f"{dataset}/evaluation_only.json"},
                ),
                connect("captures", "robot", "fit", "gripper_poses"),
                connect("captures", "target", "fit", "target_poses"),
                connect("captures", "samples", "review", "samples"),
                connect("fit", "camera_to_gripper", "review", "camera_to_gripper"),
                connect("fit", "target_to_base", "review", "target_to_base"),
                {"op": "arrange"},
                {
                    "op": "update",
                    "id": "review",
                    "fields": {"preview_output": "cloud", "node_size": "large"},
                },
                {"op": "update", "id": "captures", "fields": {"node_size": "large"}},
            ]
        )
        if refine:
            session.apply(
                [
                    add("example.hand_eye_refine", "refine", "Refine using training pixels only"),
                    {
                        "op": "disconnect",
                        "source": "fit",
                        "output": "camera_to_gripper",
                        "target": "review",
                        "input": "camera_to_gripper",
                    },
                    {
                        "op": "disconnect",
                        "source": "fit",
                        "output": "target_to_base",
                        "target": "review",
                        "input": "target_to_base",
                    },
                    connect("captures", "training_samples", "refine", "training_samples"),
                    connect("fit", "camera_to_gripper", "refine", "camera_to_gripper"),
                    connect("fit", "target_to_base", "refine", "target_to_base"),
                    connect("refine", "camera_to_gripper", "review", "camera_to_gripper"),
                    connect("refine", "target_to_base", "review", "target_to_base"),
                    {"op": "arrange"},
                ]
            )
        result = validate_graph(session.graph)
        if not result["valid"]:
            raise ValueError(result)
        session.save(destination)
    return registry


def replay(project, output):
    from opencv_dataflow.cloud_io import write_cloud
    from opencv_dataflow.ros2_poses import matrix_quaternion

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    with ProjectSession.open(project, make_registry([HERE / "hand_eye_review.py"])) as session:
        receipt = session.run()
        write(output / "run.json", receipt)
        result = session.graph.blocks["review"].last_result
        if result is None:
            raise ValueError("Review did not run; inspect the run receipt")
        report = result.outputs["report"]
        write(output / "assessment.json", report)
        cv2.imwrite(str(output / "held_out_overlay.png"), result.outputs["overlay"])
        cloud = result.outputs["cloud"]
        write_cloud(output / "calibration_review.ply", cloud)
        fit = session.graph.blocks[
            "refine" if "refine" in session.graph.blocks else "fit"
        ].last_result.outputs
        if "refine" in session.graph.blocks:
            write(output / "refinement.json", fit["report"])
        write(output / "tool0_from_camera_optical.json", fit["camera_to_gripper"])
        write(output / "base_from_board.json", fit["target_to_base"])
        t = np.array(fit["camera_to_gripper"]["source_to_target"])
        write(
            output / "estimated_camera_tf.json",
            dict(
                schema_version=1,
                transforms=[
                    dict(
                        parent_frame="tool0",
                        child_frame="hand_eye_estimated_camera_optical_frame",
                        translation_m=t[:3, 3].tolist(),
                        quaternion_xyzw=matrix_quaternion(t[:3, :3]).tolist(),
                    )
                ],
            ),
        )
        # Exportable views complement, rather than replace, the native flyaround output.
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xyz = cloud.points
        rgb = cloud.colors / 255
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(projection="3d")
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=1)
        span = np.ptp(xyz, axis=0).max() / 2
        centre = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
        ax.set(
            xlim=(centre[0] - span, centre[0] + span),
            ylim=(centre[1] - span, centre[1] + span),
            zlim=(centre[2] - span, centre[2] + span),
            xlabel="Base X (m)",
            ylabel="Base Y (m)",
            zlabel="Base Z (m)",
            title="Robot hand-eye calibration — simulated captures",
        )
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=25, azim=-55)
        fig.text(
            0.05,
            0.03,
            "Cyan: training | Magenta: held out | Gray: robot FK\n"
            "Yellow: estimated board | Green: reserved true board | RGB: pose axes",
            fontsize=10,
        )
        fig.savefig(output / "calibration_scene.png", dpi=150)
        plt.close(fig)
        print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))
        if not report["passed"]:
            raise ValueError("Frozen demonstration gates failed; retain results for review")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native-captures", type=Path)
    p.add_argument("--scene", type=Path, default=HERE / "assets/scene.json")
    p.add_argument("--replay-output", type=Path)
    p.add_argument("--build", action="store_true")
    p.add_argument("--dataset", default="captures")
    p.add_argument("--project", default="ros2_hand_eye_calibration.json")
    p.add_argument("--robot-description", type=Path)
    p.add_argument("--refine", action="store_true")
    args = p.parse_args()
    if args.native_captures:
        export(args.native_captures, args.scene, HERE / args.dataset, args.robot_description)
    project = HERE / args.project
    if args.build:
        build_project(project, args.dataset, args.refine)
    if args.replay_output:
        replay(project, args.replay_output)


if __name__ == "__main__":
    main()
