"""Private paired snapshots and frozen GDRNPP review; no worker analysis requests."""

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .completed import BINDING, SCHEMA, json_bytes, report_digest
from .selection import describe_review

COLORS = dict(
    unknown=(255, 165, 30),
    supported=(40, 240, 80),
    contradicted=(255, 50, 70),
    occluded=(150, 100, 250),
    outside_target=(65, 160, 255),
)


def freeze(array):
    result = np.array(array, copy=True)
    result.flags.writeable = False
    return result


def write_cloud(path, cloud):
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment stored scene and predicted verification samples; display pose axes; units=m\n"
        f"element vertex {len(cloud.points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    data = np.empty(len(cloud.points), dtype=[("xyz", "<f4", (3,)), ("rgb", "u1", (3,))])
    data["xyz"], data["rgb"] = cloud.points, cloud.colors
    Path(path).write_bytes(header.encode() + data.tobytes())


@dataclass(frozen=True)
class CompletedRecord:
    identity_json: bytes
    parameters_json: bytes
    ports: tuple

    @property
    def identity(self):
        return json.loads(self.identity_json)

    @property
    def parameters(self):
        return json.loads(self.parameters_json)

    def __deepcopy__(self, memo):
        return self


def retain_completed(block, result, outputs):
    if not (block.run_profile.input_previews and block.run_profile.output_previews):
        return {}
    identity = result.get("completed")
    if identity is None:  # Older workers may still execute; explain unavailable review on capture.
        return {}
    if identity.get("schema") != SCHEMA or identity.get("report_sha256") != report_digest(
        outputs["report"]
    ):
        raise ValueError("Worker completion receipt does not identify the emitted report")
    if identity.get("archive_sha256") != block.parameters["sha256"]:
        raise ValueError("Worker completion receipt identifies a different applied recipe")
    from opencv_dataflow.payloads import NO_VALUE

    ports = tuple(key for key, value in outputs.items() if value is not NO_VALUE)
    record = CompletedRecord(json_bytes(identity), json_bytes(block.parameters), ports)
    block._completed_record = record
    return {key: {BINDING: identity} for key in ports}


class CaptureJob:
    """Created on GUI thread, transferred off-thread, released on every outcome."""

    def __init__(self, record, bundle, lease):
        self.record, self.bundle, self.lease = record, bundle, lease
        self.closed = False

    def close(self):
        if not self.closed:
            self.closed = True
            self.lease.release()

    def fetch(self, *, cancelled=lambda: False, progress=lambda text: None):
        try:
            if self.closed:
                raise RuntimeError("Capture job has closed")
            identity = self.record.identity

            def call(operation, **extra):
                if cancelled():
                    raise RuntimeError("Completed review capture cancelled")
                meta, arrays = self.lease.client.call(operation, dict(identity=identity, **extra))
                if meta.get("identity") != identity:
                    raise ValueError("Worker returned another execution's evidence")
                return meta, {k: freeze(v) for k, v in arrays.items()}

            progress("Copying executed sources and recipe")
            meta, source = call("capture_completed")
            if report_digest(meta["report"]) != identity["report_sha256"]:
                raise ValueError("Stored worker report differs from the paired completion")
            report = self.bundle.outputs["report"].value
            if report_digest(report) != identity["report_sha256"]:
                raise ValueError("Paired report differs from the completed worker record")
            raw = source["recipe_bytes"].tobytes()[: meta["archive_bytes"]]
            if hashlib.sha256(raw).hexdigest() != identity["archive_sha256"]:
                raise ValueError("Transferred recipe differs from the executed archive")
            if meta["setup"]["content_sha256"] != identity["content_sha256"]:
                raise ValueError("Transferred model content differs from the completion")
            if self.record.parameters["mode"] == "Connected RGB-D":
                from .dataflow import connected_capture

                request, normalized = connected_capture(
                    {k: p.value for k, p in self.bundle.inputs.items()},
                    {k: p.payload for k, p in self.bundle.inputs.items()},
                    depth_scale_to_m=self.record.parameters["depth_scale_to_m"],
                )
                info = request["metadata"]
                if (
                    info["capture_id"] != meta["capture"]["capture_id"]
                    or info["camera_frame"] != meta["capture"]["camera_frame"]
                ):
                    raise ValueError("Paired inputs identify another capture or camera frame")
                for key, value in normalized.items():
                    if not np.array_equal(value, source[key]):
                        raise ValueError(f"Paired {key} differs from the worker's consumed source")
                # Compare optional score/source defaults without importing ML in the host.
                actual = [
                    {"object_id": 6, "score": None, "source": "external_box", **d}
                    for d in info["detections"]
                ]
                for detection in actual:
                    detection["box"] = np.asarray(detection["box"], np.float64).tolist()
                if json_bytes(actual) != json_bytes(meta["capture"]["detections"]):
                    raise ValueError("Paired boxes differ from the consumed proposal list")
            candidates = []
            for index in range(identity["candidates"]):
                progress(f"Copying candidate {index + 1}/{identity['candidates']}")
                detail, evidence = call("capture_completed_candidate", index=index)
                if (
                    detail.get("index") != index
                    or detail.get("candidate_id") != report["candidates"][index]["id"]
                ):
                    raise ValueError("Candidate evidence is out of order or stale")
                validate_evidence(report["candidates"][index], evidence)
                candidates.append(evidence)
            _, graphics = call("capture_completed_graphics")
            for port, key in (("overlay", "overlay"), ("gallery", "gallery")):
                if port in self.bundle.outputs and not np.array_equal(
                    self.bundle.outputs[port].value, graphics.get(key)
                ):
                    raise ValueError("Stored emitted image differs from the paired output")
            if "cloud" in self.bundle.outputs:
                cloud = self.bundle.outputs["cloud"].value
                if not np.array_equal(cloud.points, graphics.get("points")) or not np.array_equal(
                    cloud.colors, graphics.get("colors")
                ):
                    raise ValueError("Stored emitted cloud differs from the paired output")
            if cancelled():
                raise RuntimeError("Completed review capture cancelled")
            return FrozenReview(
                self.record,
                self.bundle,
                copy.deepcopy(meta),
                source,
                tuple(candidates),
                graphics,
                raw,
            )
        finally:
            self.close()


def validate_evidence(candidate, evidence):
    if not evidence:
        return  # Explicit unavailable state; no synthesized geometry.
    points = evidence.get("points")
    fit = candidate.get("depth_fit") or candidate.get("fit") or {}
    if points is None or points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Invalid stored fit points")
    count = len(points)
    if fit.get("rendered_samples") != count:
        raise ValueError("Stored sample count differs from the original fit denominator")
    if not count:
        return
    for name in (
        "pixels",
        "inside",
        "observed_depth_m",
        "target",
        "known",
        "delta_m",
        "supported",
        "contradicted",
        "occluded",
        "outside_target",
    ):
        if name not in evidence or evidence[name].shape != (
            (count, 2) if name == "pixels" else (count,)
        ):
            raise ValueError("Incomplete stored fit evidence")
    for name, metric in (("supported", "coverage"), ("contradicted", "contradiction")):
        if not np.isclose(evidence[name].sum() / count, fit[metric], rtol=0, atol=1e-12):
            raise ValueError("Stored flags differ from the original fit fractions")


@dataclass(frozen=True)
class FrozenReview:
    record: CompletedRecord
    bundle: object
    metadata: dict
    arrays: dict
    candidates: tuple
    graphics: dict
    recipe_bytes: bytes

    @property
    def report(self):
        return copy.deepcopy(self.bundle.outputs["report"].value)

    def colors(self, index):
        e = self.candidates[index]
        rgb = np.tile(COLORS["unknown"], (len(e.get("points", ())), 1)).astype(np.uint8)
        # Flags overlap; contradictions retain their meaning even outside the target box.
        for name in ("outside_target", "occluded", "supported", "contradicted"):
            if name in e:
                rgb[e[name]] = COLORS[name]
        return rgb

    def image(self, index, *, reference=False):
        image = self.arrays["reference_rgb" if reference else "rgb"].copy()
        if reference:
            return image
        if index is None:
            return image
        candidate = self.report["candidates"][index]
        box = candidate.get("crop", {}).get("source_xyxy")
        if box:
            lo, hi = np.rint(np.array(box).reshape(2, 2)).astype(int)
            cv2.rectangle(image, tuple(lo), tuple(hi - 1), (255, 230, 60), 1)
        e = self.candidates[index]
        if "pixels" in e:
            inside = e["inside"]
            x, y = e["pixels"][inside].T
            # Multiple rendered samples may hit one pixel. This is a display projection;
            # fractions always use the original samples, never colored pixel counts.
            image[y, x] = self.colors(index)[inside]
        pose = np.asarray(candidate["pose"]["source_to_target"])
        axes = np.r_[np.zeros((1, 3)), np.eye(3) * 0.02] @ pose[:3, :3].T + pose[:3, 3]
        if np.all(axes[:, 2] > 0):
            projected = axes @ self.arrays["camera"].T
            uv = np.rint(projected[:, :2] / projected[:, 2:3]).astype(int)
            for i, color in enumerate(((255, 0, 0), (0, 255, 0), (50, 120, 255))):
                cv2.arrowedLine(image, tuple(uv[0]), tuple(uv[i + 1]), color, 1, cv2.LINE_AA)
        return image

    def cloud(self, index, *, scene=True, axes=True):
        from opencv_dataflow.pointclouds import PointCloud

        points, colors = [], []
        if scene and "scene_points" in self.arrays:
            points.append(self.arrays["scene_points"])
            colors.append(self.arrays["scene_colors"])
        if index is not None and "points" in self.candidates[index]:
            points.append(self.candidates[index]["points"])
            colors.append(self.colors(index))
        if index is not None and axes:
            pose = np.asarray(self.report["candidates"][index]["pose"]["source_to_target"])
            for axis, color in enumerate(((255, 0, 0), (0, 255, 0), (50, 120, 255))):
                points.append(np.linspace(pose[:3, 3], pose[:3, 3] + 0.02 * pose[:3, axis], 50))
                colors.append(np.tile(color, (50, 1)).astype(np.uint8))
        if not points:
            return None
        return PointCloud(
            np.concatenate(points).astype(np.float32),
            "m",
            self.report["target_frame"],
            colors=np.concatenate(colors),
            metadata=dict(
                stored_review=True,
                measured_scene="scene_points" in self.arrays,
                predicted_geometry="original verification crop samples",
                pose_axes="derived display at stored pose; axis length 0.02 m",
                completion=self.record.identity,
            ),
        )

    def sample(self, index, row):
        e = self.candidates[index]
        if type(row) is not int or not 0 <= row < len(e.get("points", ())):
            raise ValueError("Select a stored verification sample")
        detail = {key: value[row].tolist() for key, value in e.items()}
        detail.update(
            sample=row,
            candidate_id=self.report["candidates"][index]["id"],
            scope="Original verification sample; flags may overlap",
        )
        return detail

    def export(self, directory):
        """Write a new independent review folder; original evidence and hashes are retained."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=False)
        (target / "executed.gdrnpp").write_bytes(self.recipe_bytes)
        np.savez_compressed(
            target / "sources.npz", **{k: v for k, v in self.arrays.items() if k != "recipe_bytes"}
        )
        np.savez_compressed(target / "emitted_graphics.npz", **self.graphics)
        originals = {
            k: np.asarray(p.value)
            for k, p in self.bundle.inputs.items()
            if isinstance(p.value, np.ndarray)
        }
        np.savez_compressed(target / "original_inputs.npz", **originals)
        for index, evidence in enumerate(self.candidates):
            np.savez_compressed(target / f"candidate_{index + 1:02d}.npz", **evidence)
            cloud = self.cloud(index)
            if cloud is not None:
                write_cloud(target / f"candidate_{index + 1:02d}.ply", cloud)
            cv2.imwrite(
                str(target / f"candidate_{index + 1:02d}.png"), self.image(index)[..., ::-1]
            )
        parameters = self.record.parameters
        parameters["models_dir"] = (
            "<machine-local path omitted; checkpoint identity is in setup binding>"
        )
        manifest = dict(
            schema="simple_gdrnpp_frozen_review/v1",
            identity=self.record.identity,
            report=self.report,
            setup=self.metadata["setup"],
            capture=self.metadata["capture"],
            parameters=parameters,
            path_redactions=["parameters.models_dir"],
            publication={
                k: getattr(self.bundle, k)
                for k in (
                    "snapshot_id",
                    "block_id",
                    "execution_count",
                    "output_revision",
                    "passed",
                    "score",
                    "message",
                )
            },
            payloads={
                side: {k: p.payload for k, p in getattr(self.bundle, side).items()}
                for side in ("inputs", "outputs")
            },
            original_inputs={
                k: (
                    dict(
                        dtype=p.value.dtype.str,
                        shape=list(p.value.shape),
                        encoding=getattr(p.value, "encoding", None),
                        units=getattr(p.value, "units", None),
                    )
                    if isinstance(p.value, np.ndarray)
                    else p.value
                )
                for k, p in self.bundle.inputs.items()
            },
            scope=(
                "Frozen sources and fit samples; images are display projections. "
                "No rerender or analysis."
            ),
        )
        (target / "review.json").write_bytes(json_bytes(manifest) + b"\n")
        return target


def prepare_capture(block, context):
    record = block._completed_record
    if record is None or block._lease is None:
        raise ValueError(
            "No retained completed evidence. Run in Debug with an updated worker first."
        )
    ports = (
        ()
        if record.parameters["mode"] == "Retained reference"
        else ("image", "depth", "camera", "detections", "capture_id")
    )
    bundle = context.snapshot_processed_execution(input_ports=ports, output_ports=record.ports)
    for packet in bundle.outputs.values():
        if packet.payload.get(BINDING) != record.identity:
            raise ValueError("Paired outputs belong to a different completed execution")
    report = bundle.outputs["report"].value
    if report_digest(report) != record.identity["report_sha256"]:
        raise ValueError("Paired report differs from the original completion")
    poses = [c["pose"] for c in report["candidates"] if c["fit"]["status"] == "accepted"]
    review, table = describe_review(report, record.parameters["review_candidate"])
    expected = dict(accepted_count=len(poses), poses=poses, review=review, candidate_table=table)
    if report["best_pose"] is not None:
        expected["pose"] = report["best_pose"]
    for key, value in expected.items():
        if key not in bundle.outputs or json_bytes(bundle.outputs[key].value) != json_bytes(value):
            raise ValueError("Paired pose/summary outputs differ from the original completion")
    score = max((c["fit"]["coverage"] for c in report["candidates"]), default=0.0)
    if bundle.passed != bool(poses) or bundle.score != score:
        raise ValueError("Paired judgment differs from the original completion")
    return CaptureJob(record, bundle, block._lease.acquire())
