"""Explicitly loaded example adapters; built-in blocks perform detection and fitting."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_dataflow.core import Block, Port, Result
from opencv_dataflow.geometry3d_common import pose_matrix, pose_record, rigid
from opencv_dataflow.images import ImageArray
from opencv_dataflow.parameter_validation import NumericRange, ValidationError
from opencv_dataflow.payloads import NO_VALUE, FlowGroup, ValuePacket, frame
from opencv_dataflow.pointclouds import PointCloud
from opencv_dataflow.ros2_calibration import camera_from_info
from opencv_dataflow.ros2_poses import pose_from_ros


def canonical(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def member(root, relative):
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root.resolve()):
        raise ValidationError("Capture asset must stay in its dataset")
    return path


def stamp(message):
    value = message["header"]["stamp"]
    return value["sec"] * 1_000_000_000 + value["nanosec"]


def load_sample(manifest_path, index):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "opencv.hand_eye_replay/v1" or not manifest.get("complete"):
        raise ValidationError("Require a complete exported hand-eye capture set")
    entries = manifest["captures"]
    if not 0 <= index < len(entries):
        raise ValidationError("Capture index is outside the dataset")
    if len({c["id"] for c in entries}) != len(entries) or len(
        {c["stamp_ns"] for c in entries}
    ) != len(entries):
        raise ValidationError("Capture identities and acquisition stamps must be unique")
    entry = entries[index]
    root = manifest_path.parent.resolve()
    record = json.loads(member(root, entry["path"]).read_text(encoding="utf-8"))
    if canonical(record) != entry["sha256"]:
        raise ValidationError("Capture record checksum differs")
    image = record["image"]
    key = record["stamp_ns"]
    if (
        key <= 0
        or key != entry["stamp_ns"]
        or record["id"] != entry["id"]
        or record["role"] != entry["role"]
        or record["run_id"] != manifest["run_id"]
    ):
        raise ValidationError("Capture identity/run/role differs from the retained assignment")
    if not (
        key == stamp(image) == stamp(record["camera_info"]) == stamp(record["robot_transform"])
    ):
        raise ValidationError("Image, CameraInfo and robot pose must share exact acquisition time")
    if record["tf_lookup"] != "exact_acquisition_time" or not record["settled_capture"]:
        raise ValidationError("Require settled captures with exact-time TF")
    raw = member(root, image["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != image["png_sha256"]:
        raise ValidationError("Capture image checksum differs")
    pixels = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if pixels is None or image["encoding"] != "rgb8":
        raise ValidationError("Require retained rgb8 camera captures")
    rgb = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
    if (
        pixels.shape[:2] != (image["height"], image["width"])
        or hashlib.sha256(rgb.tobytes()).hexdigest() != image["source_bytes_sha256"]
    ):
        raise ValidationError("PNG does not reproduce original camera bytes")
    camera, _ = camera_from_info(record["camera_info"])
    robot, _ = pose_from_ros(
        record["robot_transform"], expected_parent="base_link", expected_child="tool0"
    )
    if (
        camera["frame"] != "hand_eye_camera_optical_frame"
        or image["header"]["frame_id"] != camera["frame"]
        or camera["image_size"] != [image["width"], image["height"]]
    ):
        raise ValidationError("Camera optical frame/grid mismatch")
    ticket = dict(
        model="flow_ticket_v1",
        kind="string",
        value=record["id"],
        session=manifest["run_id"],
        source="ros2_hand_eye_camera",
        sequence=index + 1,
    )
    metadata = dict(
        id=record["id"],
        role=record["role"],
        run_id=manifest["run_id"],
        stamp_ns=key,
        ticket=ticket,
        image_path=str(member(root, image["path"])),
        gates=manifest["gates"],
        skeleton=record.get("robot_review_skeleton", []),
    )
    payload = dict(
        ticket=ticket, spatial=frame([image["width"], image["height"]], root_id=record["id"])
    )
    return ImageArray(pixels, "bgr", payload), camera, robot, metadata, payload


class Capture(Block):
    type_id = "example.hand_eye_capture"
    title = "Saved robot-camera capture"
    category = "Calibration"
    outputs = (
        Port("image", "image"),
        Port("camera", "data"),
        Port("robot", "data"),
        Port("capture", "data"),
    )
    defaults = dict(manifest_path="captures/manifest.json", index=0)
    project_path_parameters = ("manifest_path",)
    project_asset_parameters = (*Block.project_asset_parameters, "manifest_path")
    parameter_ranges = {"index": NumericRange(0, 63, whole=True)}
    parameter_help = {
        "index": "Retained capture index; training/held-out assignments are fixed in the manifest."
    }
    process_inputs_readonly = True

    def project_assets(self):
        assets = super().project_assets()
        manifest_path = Path(self.parameters["manifest_path"])
        root = self.project_dir or Path.cwd()
        manifest = json.loads((root / manifest_path).read_text(encoding="utf-8"))
        for entry in manifest["captures"]:
            path = manifest_path.parent / entry["path"]
            assets.append(path.as_posix())
            record = json.loads((root / path).read_text(encoding="utf-8"))
            assets.extend(
                (manifest_path.parent / record[key]).as_posix() for key in ["motion_path"]
            )
            assets.append((manifest_path.parent / record["image"]["path"]).as_posix())
        return assets

    def process(self, inputs):
        path = Path(self.parameters["manifest_path"])
        if not path.is_absolute():
            path = self.project_dir / path
        image, camera, robot, capture, payload = load_sample(path, self.parameters["index"])
        return Result(
            dict(image=image, camera=camera, robot=robot, capture=capture),
            payloads={
                p.name: (payload if p.name == "image" else {"ticket": capture["ticket"]})
                for p in self.outputs
            },
        )


class Evidence(Block):
    type_id = "example.hand_eye_evidence"
    title = "Retain image-derived pose"
    category = "Calibration"
    inputs = (
        Port("capture", "data"),
        Port("camera", "data"),
        Port("robot", "data"),
        Port("target", "data"),
        Port("observation", "data"),
    )
    outputs = (Port("sample", "data"),)
    process_inputs_readonly = True

    def process(self, inputs):
        return Result({"sample": dict(inputs)})


class Samples(Block):
    type_id = "example.hand_eye_samples"
    title = "Training / held-out samples"
    category = "Calibration"
    inputs = tuple(Port(f"sample_{i + 1}", "data") for i in range(25))
    outputs = (
        Port("robot", "any"),
        Port("target", "any"),
        Port("samples", "data"),
        Port("training_samples", "data"),
    )
    outcome_passthrough = True
    per_packet_outcomes = True
    process_inputs_readonly = True

    def process(self, inputs):
        samples = [inputs[p.name] for p in self.inputs]
        identities = [s["capture"]["id"] for s in samples]
        if len(set(identities)) != 25 or len({s["capture"]["run_id"] for s in samples}) != 1:
            raise ValidationError("Require 25 unique captures from one run")
        training = [s for s in samples if s["capture"]["role"] == "train"]
        held = [s for s in samples if s["capture"]["role"] == "held_out"]
        if len(training) != 20 or len(held) != 5:
            raise ValidationError("Require the frozen 20 training / 5 held-out split")
        groups = {
            key: FlowGroup(
                tuple(ValuePacket(s[key], {"ticket": s["capture"]["ticket"]}) for s in training),
                tuple(s["capture"]["id"] for s in training),
            )
            for key in ("robot", "target")
        }
        return Result({**groups, "samples": samples, "training_samples": training})


def refine_mount(samples, initial_mount, initial_board, max_evaluations=200):
    """Refine both rigid poses against training pixels; K/D remain fixed."""
    from scipy.optimize import least_squares

    if not 3 <= len(samples) <= 64 or any(s["capture"]["role"] != "train" for s in samples):
        raise ValidationError("Reprojection refinement accepts training captures only")
    initial_mount, initial_board = rigid(initial_mount), rigid(initial_board)

    def pack(t):
        return np.r_[cv2.Rodrigues(t[:3, :3])[0].reshape(3), t[:3, 3]]

    def unpack(x):
        return pose_matrix(cv2.Rodrigues(x[:3])[0], x[3:])

    prepared = []
    for s in samples:
        if (
            s["robot"]["source_frame"] != "tool0"
            or s["robot"]["target_frame"] != "base_link"
            or s["robot"]["units"] != "m"
            or s["camera"]["units"] != "m"
        ):
            raise ValidationError(
                "Refinement requires the retained tool0/base_link metric convention"
            )
        prepared.append(
            (
                rigid(s["robot"]),
                np.array(s["observation"]["object_points"]),
                np.array(s["observation"]["image_points"]),
                np.array(s["camera"]["camera_matrix"]),
                np.array(s["camera"]["distortion"]),
            )
        )

    def residual(x):
        mount, board = unpack(x[:6]), unpack(x[6:])
        errors = []
        for robot, xyz, xy, k, d in prepared:
            pose = np.linalg.inv(robot @ mount) @ board
            z = (xyz @ pose[:3, :3].T + pose[:3, 3])[:, 2]
            if np.any(z <= 0.01):
                return np.full(sum(len(v[1]) * 2 for v in prepared), 1e6)
            predicted = cv2.projectPoints(xyz, cv2.Rodrigues(pose[:3, :3])[0], pose[:3, 3], k, d)[
                0
            ].reshape(-1, 2)
            errors.extend((predicted - xy).reshape(-1))
        return np.array(errors)

    initial = np.r_[pack(initial_mount), pack(initial_board)]
    fit = least_squares(
        residual,
        initial,
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        max_nfev=max_evaluations,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    initial_rms = float(np.sqrt(np.mean(residual(initial) ** 2) * 2))
    final_rms = float(np.sqrt(np.mean(residual(fit.x) ** 2) * 2))
    if not fit.success or not np.isfinite(fit.x).all() or final_rms > initial_rms + 1e-9:
        raise ValidationError("Reprojection refinement did not converge to a finite improvement")
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    if singular[-1] < singular[0] * 1e-10:
        raise ValidationError("Training motion does not constrain the refinement")
    report = dict(
        model="hand_eye_reprojection_refinement_v1",
        training_views=len(samples),
        held_out_views_used=0,
        truth_used=False,
        fixed_camera_intrinsics=True,
        method="training_pixel_soft_l1",
        initial_rms_px=initial_rms,
        final_rms_px=final_rms,
        evaluations=fit.nfev,
        jacobian_condition_number=float(singular[0] / singular[-1]),
    )
    return unpack(fit.x[:6]), unpack(fit.x[6:]), report


class Refine(Block):
    type_id = "example.hand_eye_refine"
    title = "Refine mount from training pixels"
    category = "Calibration"
    inputs = (
        Port("training_samples", "data"),
        Port("camera_to_gripper", "data"),
        Port("target_to_base", "data"),
    )
    outputs = (
        Port("camera_to_gripper", "data"),
        Port("target_to_base", "data"),
        Port("report", "data"),
    )
    defaults = {"max_evaluations": 200, "max_rms_px": 1.0}
    parameter_ranges = {
        "max_evaluations": NumericRange(10, 2000, whole=True),
        "max_rms_px": NumericRange(0.001, 10),
    }
    process_inputs_readonly = True

    def process(self, inputs):
        mount, board, report = refine_mount(
            inputs["training_samples"],
            inputs["camera_to_gripper"],
            inputs["target_to_base"],
            self.parameters["max_evaluations"],
        )
        passed = report["final_rms_px"] <= self.parameters["max_rms_px"]
        return Result(
            dict(
                camera_to_gripper=pose_record(mount, "hand_eye_camera_optical_frame", "tool0", "m")
                if passed
                else NO_VALUE,
                target_to_base=pose_record(board, "charuco_board", "base_link", "m")
                if passed
                else NO_VALUE,
                report=report,
            ),
            passed=passed,
            fail_reasons=[] if passed else ["hand_eye_refinement_rms"],
            message=(
                f"Training pixel RMS {report['initial_rms_px']:.3f} -> {report['final_rms_px']:.3f}"
            ),
        )


def angle(rotation):
    return float(
        np.degrees(
            np.arctan2(
                np.linalg.norm(
                    [
                        rotation[2, 1] - rotation[1, 2],
                        rotation[0, 2] - rotation[2, 0],
                        rotation[1, 0] - rotation[0, 1],
                    ]
                ),
                np.trace(rotation) - 1,
            )
        )
    )


def assess(samples, mount, board, truth):
    mount, board = rigid(mount), rigid(board)
    rows = []
    for sample in samples:
        robot = rigid(sample["robot"])
        measured = rigid(sample["target"])
        predicted = np.linalg.inv(robot @ mount) @ board
        observation = sample["observation"]
        xyz = np.array(observation["object_points"])
        xy = np.array(observation["image_points"])
        camera = sample["camera"]
        projection = cv2.projectPoints(
            xyz,
            cv2.Rodrigues(predicted[:3, :3])[0],
            predicted[:3, 3],
            np.array(camera["camera_matrix"]),
            np.array(camera["distortion"]),
        )[0].reshape(-1, 2)
        reconstructed = robot @ mount @ measured
        rows.append(
            dict(
                id=sample["capture"]["id"],
                role=sample["capture"]["role"],
                corner_count=len(xy),
                translation_error_mm=float(
                    np.linalg.norm(reconstructed[:3, 3] - board[:3, 3]) * 1000
                ),
                rotation_error_degrees=angle(board[:3, :3].T @ reconstructed[:3, :3]),
                reprojection_rms_px=float(np.sqrt(np.mean(np.sum((projection - xy) ** 2, axis=1)))),
                observed_xy=xy.tolist(),
                predicted_xy=projection.tolist(),
            )
        )
    held = [r for r in rows if r["role"] == "held_out"]
    expected = rigid(truth["tool0_from_camera_optical"])
    summary = dict(
        model="ros2_hand_eye_assessment_v1",
        simulation_only=True,
        training_views=20,
        held_out_views=5,
        mount_error_mm=float(np.linalg.norm(mount[:3, 3] - expected[:3, 3]) * 1000),
        mount_error_degrees=angle(expected[:3, :3].T @ mount[:3, :3]),
        held_out_translation_rms_mm=float(
            np.sqrt(np.mean([r["translation_error_mm"] ** 2 for r in held]))
        ),
        held_out_rotation_rms_degrees=float(
            np.sqrt(np.mean([r["rotation_error_degrees"] ** 2 for r in held]))
        ),
        held_out_max_reprojection_rms_px=max(r["reprojection_rms_px"] for r in held),
        samples=rows,
        truth_used_for_fitting=False,
        physical_calibration_qualified=False,
    )
    gates = samples[0]["capture"]["gates"]
    tests = {
        "mount_error_mm": "max_mount_error_mm",
        "mount_error_degrees": "max_mount_error_degrees",
        "held_out_translation_rms_mm": "max_held_out_translation_rms_mm",
        "held_out_rotation_rms_degrees": "max_held_out_rotation_rms_degrees",
        "held_out_max_reprojection_rms_px": "max_reprojection_rms_px",
    }
    summary["gates"] = gates
    summary["failed_gates"] = [key for key, limit in tests.items() if summary[key] > gates[limit]]
    summary["passed"] = not summary["failed_gates"]
    return summary


class Review(Block):
    type_id = "example.hand_eye_review"
    title = "Held-out calibration review"
    category = "Calibration"
    inputs = (
        Port("samples", "data"),
        Port("camera_to_gripper", "data"),
        Port("target_to_base", "data"),
    )
    outputs = (Port("report", "data"), Port("cloud", "cloud"), Port("overlay", "image"))
    defaults = {"evaluation_path": "captures/evaluation_only.json"}
    project_path_parameters = ("evaluation_path",)
    project_asset_parameters = (*Block.project_asset_parameters, "evaluation_path")
    process_inputs_readonly = True

    def process(self, inputs):
        path = Path(self.parameters["evaluation_path"])
        if not path.is_absolute():
            path = self.project_dir / path
        truth = json.loads(path.read_text(encoding="utf-8"))
        samples = inputs["samples"]
        mount = rigid(inputs["camera_to_gripper"])
        board = rigid(inputs["target_to_base"])
        report = assess(samples, mount, board, truth)
        points, colors = [], []

        def line(a, b, color):
            points.extend(np.linspace(a, b, 50))
            colors.extend([color] * 50)

        def axes(t, scale=0.05):
            for i, color in enumerate(([255, 70, 70], [70, 255, 70], [80, 140, 255])):
                line(t[:3, 3], t[:3, 3] + t[:3, i] * scale, color)

        axes(np.eye(4), 0.15)
        axes(board, 0.09)
        board_corners = np.array(
            [[0, 0, 0], [0.21, 0, 0], [0.21, 0.15, 0], [0, 0.15, 0], [0, 0, 0]]
        )
        board_xyz = board_corners @ board[:3, :3].T + board[:3, 3]
        for a, b in zip(board_xyz, board_xyz[1:]):
            line(a, b, [245, 220, 60])
        expected_board = rigid(truth["base_from_board"])
        truth_xyz = board_corners @ expected_board[:3, :3].T + expected_board[:3, 3]
        for a, b in zip(truth_xyz, truth_xyz[1:]):
            line(a, b, [50, 245, 100])
        for sample in samples:
            camera = rigid(sample["robot"]) @ mount
            color = [40, 200, 220] if sample["capture"]["role"] == "train" else [240, 70, 220]
            axes(camera, 0.025)
            line(camera[:3, 3], (board_xyz[0] + board_xyz[2]) / 2, color)
            measured = rigid(sample["robot"]) @ mount @ rigid(sample["target"])
            xyz = (
                np.array(sample["observation"]["object_points"]) @ measured[:3, :3].T
                + measured[:3, 3]
            )
            points.extend(xyz)
            colors.extend([color] * len(xyz))
        skeleton = np.array(samples[-1]["capture"].get("skeleton", []))
        if len(skeleton):
            for a, b in zip(skeleton, skeleton[1:]):
                line(a, b, [160, 165, 180])
        cloud = PointCloud(
            np.asarray(points),
            units="m",
            frame="base_link",
            colors=np.asarray(colors),
            metadata={
                "legend": {
                    "cyan": "training camera rays / reconstructed corners",
                    "magenta": "held-out camera rays / corners",
                    "yellow": "estimated board outline",
                    "green": "reserved true board outline",
                    "gray": "measured-joint model FK skeleton (last view)",
                    "RGB": "XYZ pose axes",
                },
                "simulation_only": True,
            },
        )
        overlay = NO_VALUE
        if self.generate_graphics:
            selected = [s for s in samples if s["capture"]["role"] == "held_out"][0]
            row = next(r for r in report["samples"] if r["id"] == selected["capture"]["id"])
            overlay = cv2.imread(selected["capture"]["image_path"])
            predicted = np.linalg.inv(rigid(selected["robot"]) @ mount) @ board
            cv2.drawFrameAxes(
                overlay,
                np.array(selected["camera"]["camera_matrix"]),
                np.array(selected["camera"]["distortion"]),
                cv2.Rodrigues(predicted[:3, :3])[0],
                predicted[:3, 3],
                0.045,
                2,
            )
            for observed, predicted in zip(row["observed_xy"], row["predicted_xy"]):
                a, b = tuple(np.rint(observed).astype(int)), tuple(np.rint(predicted).astype(int))
                cv2.circle(overlay, a, 4, (0, 220, 0), 1)
                cv2.drawMarker(overlay, b, (255, 0, 220), cv2.MARKER_CROSS, 9, 1)
                # Magnified residual vectors aid review; endpoint crosses remain true scale.
                tip = tuple(
                    np.rint(np.array(observed) + (np.array(predicted) - observed) * 30).astype(int)
                )
                cv2.line(overlay, a, tip, (0, 170, 255), 1)
            for i, text in enumerate(
                [
                    f"HELD OUT {row['id']} | predicted RMS {row['reprojection_rms_px']:.3f} px",
                    "Green: observed | Magenta: predicted | Orange: residual x30",
                    f"Mount error: {report['mount_error_mm']:.3f} mm / "
                    f"{report['mount_error_degrees']:.3f} deg",
                ]
            ):
                cv2.putText(
                    overlay,
                    text,
                    (25, 35 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (35, 35, 35),
                    2,
                    cv2.LINE_AA,
                )
        return Result(
            dict(
                report=report,
                cloud=cloud,
                overlay=ImageArray(overlay, "bgr") if overlay is not NO_VALUE else NO_VALUE,
            ),
            passed=report["passed"],
            fail_reasons=report["failed_gates"],
            message=(
                f"Held-out RMS {report['held_out_max_reprojection_rms_px']:.3f} px; "
                f"mount {report['mount_error_mm']:.3f} mm"
            ),
        )


def register(registry):
    for cls in (Capture, Evidence, Samples, Refine, Review):
        registry.register(cls)
