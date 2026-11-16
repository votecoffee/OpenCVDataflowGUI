"""Light host adapter. Numeric inference and heavy dependencies stay in an isolated worker."""

import copy
import re
from pathlib import Path

import numpy as np

from opencv_dataflow.core import Block, Port, Result
from opencv_dataflow.images import ImageArray, opencv_image
from opencv_dataflow.parameter_validation import NumericRange, ValidationError
from opencv_dataflow.payloads import NO_VALUE, ValuePacket, frame, require_same_frame
from opencv_dataflow.pointclouds import PointCloud

from .bridge import WorkerClient, project_recipe_path
from .selection import describe_review

CAPTURE_PORTS = ("image", "depth", "camera", "detections", "capture_id")


def connected_capture(inputs, payloads, *, depth_scale_to_m=1):
    if any(name not in inputs for name in CAPTURE_PORTS):
        raise ValueError(
            "Connect image, registered depth, camera, detections and capture ID together"
        )
    image = opencv_image(inputs["image"])
    raw = inputs["depth"]
    camera = inputs["camera"]
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Use an unambiguous uint8 color image")
    if (
        raw.ndim != 2
        or raw.shape != image.shape[:2]
        or raw.dtype not in (np.uint16, np.float32, np.float64)
    ):
        raise ValueError(
            "Connect registered single-channel uint16/float depth matching the color grid"
        )
    require_same_frame(
        [
            ValuePacket(inputs["image"], payloads.get("image", {})),
            ValuePacket(raw, payloads.get("depth", {})),
        ],
        message="Color and depth declare different coordinate grids; register them first",
    )
    if not isinstance(camera, dict):
        raise ValueError("Camera must declare matrix, image_size and frame")
    if (
        camera.get("lens", "pinhole") != "pinhole"
        or np.any(camera.get("distortion", []))
        or tuple(camera.get("image_size", ())) != image.shape[1::-1]
    ):
        raise ValueError("Use rectified intrinsics with the matching color/depth image size")
    key = "camera_matrix" if "camera_matrix" in camera else "matrix"
    k = np.array(camera[key], np.float64)
    units = getattr(raw, "units", "unknown")
    scale = {"m": 1.0, "mm": 0.001}.get(units, depth_scale_to_m)
    if isinstance(scale, bool) or not np.isfinite(scale) or not 1e-12 <= scale <= 1:
        raise ValueError("Declare a positive stored-depth to meter scale")
    capture_id = inputs["capture_id"]
    if not isinstance(capture_id, str) or not capture_id.strip():
        raise ValueError("Name the current capture")
    detections = inputs["detections"]
    if not isinstance(detections, list) or len(detections) > 64:
        raise ValueError("Connect up to 64 explicit detection records")
    metadata = dict(
        capture_id=capture_id, model_frame=None, camera_frame=camera["frame"], detections=detections
    )
    return dict(kind="arrays", metadata=metadata), dict(
        rgb=np.ascontiguousarray(image[..., ::-1]),
        depth=np.asarray(raw, np.float32) * scale,
        camera=k,
        mask=np.ones(raw.shape, np.uint8),
    )


class SimpleGDRNPP(Block):
    type_id = "3d.simple_gdrnpp"
    title = "SimpleGDRNPP"
    category = block_type = "3D vision"
    subtype = "Learned object pose"
    composes_result_image = True
    process_inputs_readonly = True
    parallel_safe = False
    preview_safe = False
    plugin_dialog_label = "Set up trained object…"
    inputs = tuple(
        Port(name, kind, required=False, required_when_connected=True)
        for name, kind in [
            ("image", "image"),
            ("depth", "depth"),
            ("camera", "data"),
            ("detections", "data"),
            ("capture_id", "text"),
        ]
    )
    outputs = (
        Port("report", "data"),
        Port("pose", "data", required=False),
        Port("poses", "data"),
        Port("accepted_count", "integer"),
        Port("overlay", "image", required=False),
        Port("cloud", "cloud", required=False),
        Port("review", "data"),
        Port("candidate_table", "data"),
        Port("gallery", "image", required=False),
    )
    defaults = dict(
        recipe_path="",
        sha256="",
        models_dir="",
        mode="Connected RGB-D",
        depth_scale_to_m=1.0,
        render_overlay=True,
        render_cloud=True,
        review_candidate=0,
        render_gallery=False,
    )
    parameter_choices = {"mode": ("Connected RGB-D", "Retained reference")}
    parameter_ranges = {
        "depth_scale_to_m": NumericRange(1e-12, 1),
        "review_candidate": NumericRange(0, 64, whole=True),
    }
    project_path_parameters = ("models_dir",)
    parameter_help = {
        "recipe_path": "Project-relative sealed .gdrnpp setup staged by the native popup.",
        "sha256": "Pinned complete setup archive hash.",
        "models_dir": "External verified checkpoint directory; weights remain outside the project.",
        "mode": "Use connected current inputs, or intentionally rerun the retained reference.",
        "depth_scale_to_m": (
            "For untagged stored depth only. Tagged m/mm depth already carries its units."
        ),
        "render_overlay": "Generate the candidate pose/fit image when graphics are enabled.",
        "render_cloud": "Generate measured scene and predicted CAD/axes for native flyaround.",
        "review_candidate": (
            "0 shows the highest-support accepted pose, or strongest rejected proposal. "
            "1..64 selects that proposal for graphics; accepted pose outputs stay unchanged."
        ),
        "render_gallery": (
            "Optional image of all numbered proposals; adds rendering work per candidate."
        ),
    }
    input_help = {
        "camera": "Rectified pinhole dict: matrix, image_size, frame, optional zero distortion.",
        "detections": "Object-6 instances: identifier, XYXY pixel box with exclusive upper bounds.",
        "depth": "Already registered axial depth; no sensor registration or synchronization.",
        "capture_id": "Explicit current acquisition identity; no implicit reference-pose reuse.",
    }
    output_help = {
        "pose": "Best depth-accepted pose only; absent on rejection. Not a grasp qualification.",
        "poses": "All depth-accepted poses. Raw proposals remain separately in report.",
        "cloud": "Review-only measured scene, predicted CAD and axes; RGB colors explain fit.",
        "review": "Selected proposal identity/status and whether it is the best accepted pose.",
        "candidate_table": "Proposal numbers, IDs, support/contradiction and selection.",
        "gallery": "Optional diagnostic montage of all proposals; uses its own pixel grid.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = None
        self.preview_output = "cloud"

    @property
    def filmstrip_output(self):
        return "overlay"

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["recipe_path"]:
            try:
                project_recipe_path(self.project_dir, p["recipe_path"])
            except ValueError as e:
                raise ValidationError(str(e), "recipe_path") from e
            if re.fullmatch("[0-9a-f]{64}", p["sha256"]) is None:
                raise ValidationError("Stage a setup with its complete SHA-256", "sha256")
        for name in ("render_overlay", "render_cloud", "render_gallery"):
            if type(p[name]) is not bool:
                raise ValidationError("Rendering options must be boolean", name)
        if type(p["review_candidate"]) is not int:
            raise ValidationError("Review candidate must be an integer", "review_candidate")
        if not isinstance(p["models_dir"], str):
            raise ValidationError("Choose an external model directory", "models_dir")

    def process(self, inputs):
        p = self.parameters
        try:
            base = dict(
                kind="recipe",
                path=str(project_recipe_path(self.project_dir, p["recipe_path"])),
                sha256=p["sha256"],
            )
            if not p["models_dir"]:
                raise ValueError("Choose the verified external model folder in Setup")
            path = Path(p["models_dir"])
            if not path.is_absolute():
                path = Path(self.project_dir) / path
            data = dict(
                base=base,
                models_dir=str(path),
                graphics=self.generate_graphics
                and (p["render_overlay"] or p["render_cloud"] or p["render_gallery"]),
                want_image=p["render_overlay"],
                want_cloud=p["render_cloud"],
                want_gallery=p["render_gallery"],
                review_candidate=p["review_candidate"],
            )
            arrays = {}
            if p["mode"] == "Connected RGB-D":
                data["current"], arrays = connected_capture(
                    inputs, self.input_payloads, depth_scale_to_m=p["depth_scale_to_m"]
                )
            elif inputs:
                raise ValueError(
                    "Retained-reference mode must not silently ignore connected current inputs"
                )
            if self._client is None:
                self._client = WorkerClient()
            result, graphics = self._client.call(
                "preview", data, arrays, cancelled=getattr(self, "_cancelled", lambda: False)
            )
        except ValueError as error:
            raise ValidationError(str(error)) from error
        report = result["report"]
        selection, table = describe_review(report, p["review_candidate"])
        if (
            any(k in graphics for k in ("overlay", "points"))
            and result.get("candidate_index") != selection["candidate_index"]
        ):
            raise ValidationError(
                "Update the isolated worker: returned graphics select another proposal"
            )
        if data["graphics"] and p["render_gallery"] and "gallery" not in graphics:
            raise ValidationError("Update the isolated worker to create the candidate gallery")
        accepted = [c["pose"] for c in report["candidates"] if c["fit"]["status"] == "accepted"]
        contexts = {}
        image = NO_VALUE
        cloud = NO_VALUE
        gallery = NO_VALUE
        if "overlay" in graphics:
            image = ImageArray(graphics["overlay"], "rgb")
            contexts["overlay"] = (
                copy.deepcopy(self.input_payloads.get("image", {}))
                if p["mode"] == "Connected RGB-D"
                else {
                    "spatial": frame(
                        report["input"]["source_size"],
                        f"gdrnpp-reference:{report['setup']['content_sha256']}",
                    )
                }
            )
        if "points" in graphics:
            cloud = PointCloud(
                graphics["points"],
                "m",
                report["target_frame"],
                colors=graphics["colors"],
                metadata={
                    "derived_review": True,
                    "components": ["measured scene", "predicted visible CAD", "pose axes"],
                    "capture_id": report["capture_id"],
                    "review": copy.deepcopy(selection),
                },
            )
        if "gallery" in graphics:
            gallery = ImageArray(graphics["gallery"], "rgb")
            contexts["gallery"] = {
                "spatial": frame(gallery.shape[1::-1], f"gdrnpp-gallery:{report['capture_id']}")
            }
        viewing = (
            f"reviewing {selection['candidate_number']}/{selection['candidate_count']} "
            f"{selection['candidate_id']} ({selection['status']})"
            if selection["candidate_index"] is not None
            else "no proposals to review"
        )
        return Result(
            dict(
                report=report,
                pose=report["best_pose"] or NO_VALUE,
                poses=accepted,
                accepted_count=len(accepted),
                overlay=image,
                cloud=cloud,
                review=selection,
                candidate_table=table,
                gallery=gallery,
            ),
            payloads=contexts,
            passed=bool(accepted),
            score=max((c["fit"]["coverage"] for c in report["candidates"]), default=0.0),
            message=(
                f"{len(accepted)} depth-accepted poses; {viewing}; "
                "score is maximum measured support"
            ),
            fail_reasons=[] if accepted else ["no_depth_accepted_pose"],
        )

    def create_plugin_dialog(self, parent, parameters, context):
        from .qt_setup import GDRNPPSetupDialog

        return GDRNPPSetupDialog(parent, parameters, context)

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None

    def reset_runtime(self):
        self.close()


def register(registry):
    registry.register(SimpleGDRNPP)
