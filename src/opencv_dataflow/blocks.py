"""Built-in acquisition, G-API processing, inspection, and output blocks."""

from __future__ import annotations

import copy
import csv
import json
import sqlite3
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np

from .capture_settings import Camera, RecordedCapture
from .codec_metadata import write_metadata_file
from .core import Block, Graph, Port, Registry, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .image_codecs import (
    CODEC_CHOICES,
    CODEC_DEFAULTS,
    CODEC_HELP,
    CODEC_RANGES,
    MAX_ENCODED_BYTES,
    codec_image,
    codec_options,
    encode_image_buffer,
)
from .image_metadata import decode_file_image
from .images import COLOR_FORMATS, PIXEL_DEPTHS, format_image, image_format, opencv_image
from .objects import collection, contour_item
from .parameter_validation import NumericRange, ValidationError
from .parameters import BORDERS, CONTOUR_APPROXIMATIONS, THRESHOLD_MODES, scalar_value
from .payloads import NO_VALUE, ValuePacket, moved_payload, require_same_frame
from .replay import ReplaySource
from .selection import SELECTION_INPUTS, RegionalOperation, contains, selection_mask


class Comment(Block):
    """A saved canvas note for instructions, context, or source attribution."""

    type_id, title, category = "layout.comment", "Comment", "Annotations"
    subtype = "Layout"
    layout_only = True


class DemoImage(Block):
    block_type, subtype = "Input", "Synthetic"
    type_id, title, category = "input.demo", "Demo image", "Input"
    outputs = (Port("image"),)

    def process(self, inputs):
        image = np.full((360, 480, 3), 28, np.uint8)
        cv2.circle(image, (140, 180), 72, (210, 190, 70), -1)
        cv2.rectangle(image, (270, 105), (395, 250), (90, 205, 160), -1)
        cv2.putText(image, "DATAFLOW", (26, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
        return Result({"image": image}, message="Synthetic inspection target")


class ImageFile(ReplaySource):
    block_type, subtype = "Input", "Files"
    type_id, title, category = "input.file", "Image file", "Input"
    outputs = (Port("image"), Port("info", "data"), Port("metadata", "any", required=False))
    defaults = {
        "path": "",
        "source_mode": "single",
        "paths": [],
        "directory": "",
        "recursive": False,
        "repeat": "stop",
        "filename_ticket": True,
        "metadata_mode": "off",
        "color_format": "bgr",
        "pixel_depth": "uint8",
        "path_base": "working_directory",
    }
    parameter_choices = {
        "source_mode": ("single", "files", "directory"),
        "repeat": ("stop", "loop"),
        "metadata_mode": ("off", "read", "apply_orientation"),
        "color_format": COLOR_FORMATS,
        "pixel_depth": PIXEL_DEPTHS,
        "path_base": ("working_directory", "project"),
    }
    parameter_help = {
        "paths": "Ordered image paths; one image per activation in files mode.",
        "directory": "Snapshot supported image filenames in sorted order.",
        "repeat": "Stop at end, or loop. Reset flow state to restart.",
        "filename_ticket": "Use filename tickets unless an explicit ticket mode is selected.",
        "metadata_mode": "Report metadata, or also apply EXIF orientation to JPEG/PNG images.",
    }
    extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
        ".jp2",
        ".pbm",
        ".pgm",
        ".ppm",
        ".exr",
        ".hdr",
        ".pic",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_outputs = [name for name in self.exposed_outputs if name != "metadata"]
        self.reset_runtime()

    def reset_runtime(self):
        self._sequence = {
            "signature": None,
            "files": [],
            "cursor": 0,
            "finished": False,
            "selected": None,
        }

    def snapshot_runtime(self):
        return copy.deepcopy(self._sequence)

    def restore_runtime(self, snapshot):
        self._sequence = copy.deepcopy(snapshot)

    def copy_runtime_from(self, other):
        self.restore_runtime(other.snapshot_runtime())

    @property
    def replayable(self):
        return self.parameters["source_mode"] != "single"

    @property
    def exhausted(self):
        return self.finite_source and self._sequence["finished"]

    def resolve_path(self, value):
        path = Path(value)
        base = self.parameters["path_base"]
        if base not in ("working_directory", "project"):
            raise ValueError("Path base must be working_directory or project")
        if base == "project" and not path.is_absolute():
            if self.project_dir is None:
                raise ValueError(
                    "Save or open the project before using a project-relative image path"
                )
            path = self.project_dir / path
        return path

    def issue_source_ticket(self):
        mode = self.parameters["source_mode"]
        state = self._sequence
        signature = json.dumps(self.parameters, sort_keys=True)
        if signature != state["signature"]:
            self.reset_runtime()
            state = self._sequence
            state["signature"] = signature
            state["finished"] = True
            if mode == "single":
                files = [self.resolve_path(self.parameters["path"])]
            elif mode == "files":
                paths = self.parameters["paths"]
                if not isinstance(paths, list) or any(
                    not isinstance(p, str) or not p for p in paths
                ):
                    raise ValueError("Image paths must be a list of nonempty strings")
                files = [self.resolve_path(p) for p in paths]
            elif mode == "directory":
                directory = self.resolve_path(self.parameters["directory"])
                if not directory.is_dir():
                    raise ValueError(f"Image directory does not exist: {directory}")
                candidates = (
                    directory.rglob("*") if self.parameters["recursive"] else directory.iterdir()
                )
                files = sorted(
                    (p for p in candidates if p.is_file() and p.suffix.lower() in self.extensions),
                    key=lambda p: (str(p).casefold(), str(p)),
                )
            else:
                raise ValueError("Unknown image source mode")
            state["files"] = [str(p) for p in files]
            if not files:
                raise ValueError("No images selected or found")
            state["finished"] = False
        if state["finished"] and mode != "single":
            state["selected"] = None
            return None
        index = state["cursor"] % len(state["files"])
        state["selected"] = state["files"][index]
        state["cursor"] += 1
        state["finished"] = self.finite_source and state["cursor"] >= len(state["files"])
        if self.replayable and not self.finite_source:
            state["cursor"] %= len(state["files"])
        ticket = super().issue_source_ticket()
        if self.parameters["filename_ticket"] and self.ticket_settings["mode"] == "auto_number":
            ticket.update(
                kind="string",
                value=Path(state["selected"]).name,
                session=f"{ticket['session']}:file:{ticket['sequence']}",
            )
        return ticket

    def process(self, inputs):
        selected = self._sequence["selected"]
        if selected is None:
            return Result(
                {"image": NO_VALUE, "info": NO_VALUE, "metadata": NO_VALUE},
                message="Image sequence complete",
            )
        path = Path(selected)
        buffer = self.read_bytes(path)
        include_chunks = "metadata" in self.exposed_outputs
        decoded = decode_file_image(
            buffer, self.parameters["metadata_mode"], include_chunks=include_chunks
        )
        image, metadata = decoded[:2]
        chunks = decoded[2] if include_chunks else None
        if image is None:
            raise ValueError(f"Unable to decode image: {path}")
        output = format_image(
            image, self.parameters["color_format"], self.parameters["pixel_depth"]
        )
        info = {
            "filename": path.name,
            "path": str(path),
            "width": int(output.shape[1]),
            "height": int(output.shape[0]),
            "channels": 1 if output.ndim == 2 else int(output.shape[2]),
            "color_format": image_format(output),
            "pixel_depth": str(output.dtype),
            "bits_per_channel": output.dtype.itemsize * 8,
            "source_color_format": image_format(image),
            "source_pixel_depth": str(image.dtype),
        }
        if metadata is not None:
            info["metadata"] = metadata
        return Result(
            {"image": output, "info": info, "metadata": chunks if chunks is not None else NO_VALUE},
            graphics={"decoded source": image},
            message=f"{path.name} · {info['color_format']} / {output.dtype}",
        )

    def read_bytes(self, path):
        return path.read_bytes()


class GAPIBlock(Block):
    """Cache a native GComputation per parameter set; never silently fall back."""

    category = "G-API"
    block_type, subtype = "Advanced", "G-API"
    inputs, outputs = (Port("image"),), (Port("image"),)
    input_depths = (np.uint8, np.uint16, np.float32)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._computation = None
        self._key = None

    def build(self, image):
        raise NotImplementedError

    def process(self, inputs):
        source = opencv_image(inputs["image"])
        if source.dtype not in self.input_depths:
            names = ", ".join(np.dtype(dtype).name for dtype in self.input_depths)
            raise ValueError(f"{self.title} requires pixel depth {names}; got {source.dtype}")
        key = json.dumps(self.parameters, sort_keys=True)
        if self._computation is None or self._key != key:
            image = cv2.GMat()
            output = self.build(image)
            self._computation = cv2.GComputation([image], [output])
            self._key = key
        image = self._computation.apply([source])
        if isinstance(image, (tuple, list)):
            image = image[0]
        return Result({"image": image}, message="OpenCV G-API / CPU")


class Grayscale(GAPIBlock):
    type_id, title = "gapi.gray", "Grayscale"
    input_depths = (np.uint8,)

    def build(self, image):
        return cv2.gapi.BGR2Gray(image)


class Blur(RegionalOperation, GAPIBlock):
    type_id, title = "gapi.blur", "Gaussian blur"
    defaults = {
        "kernel": 5,
        "sigma": 1.2,
        "kernel_y": 0,
        "sigma_y": 0.0,
        "border": "reflect",
        "border_value": [0.0, 0.0, 0.0, 0.0],
    }
    parameter_choices = {"border": tuple(BORDERS)}
    parameter_help = {
        "kernel_y": "0 uses the legacy square kernel size.",
        "sigma_y": "0 uses sigma X, following OpenCV.",
        "border_value": "Used only by the constant border mode.",
    }

    def build(self, image):
        kernel = int(self.parameters["kernel"])
        if kernel < 0 or (kernel != 0 and kernel % 2 != 1):
            raise ValueError("Kernel must be a positive odd integer")
        sigma = float(self.parameters["sigma"])
        if sigma < 0:
            raise ValueError("Sigma must be nonnegative")
        ky = int(self.parameters["kernel_y"]) or kernel
        sy = float(self.parameters["sigma_y"])
        if ky < 0 or (ky != 0 and ky % 2 != 1) or not np.isfinite([sigma, sy]).all() or sy < 0:
            raise ValueError("Gaussian kernel Y must be positive odd and sigmas finite/nonnegative")
        if (kernel == 0 and sigma <= 0) or (ky == 0 and sigma <= 0 and sy <= 0):
            raise ValueError("Automatic Gaussian kernel sizing requires positive sigma")
        return cv2.gapi.gaussianBlur(
            image,
            (kernel, ky),
            sigma,
            sy,
            BORDERS[self.parameters["border"]],
            scalar_value(self.parameters["border_value"]),
        )


class Canny(RegionalOperation, GAPIBlock):
    region_background = "zero"
    type_id, title = "gapi.canny", "Canny edges"
    input_depths = (np.uint8,)
    defaults = {"low": 60.0, "high": 150.0, "aperture": "3", "l2_gradient": False}
    parameter_choices = {"aperture": ("3", "5", "7")}

    def build(self, image):
        low, high = float(self.parameters["low"]), float(self.parameters["high"])
        if not 0 <= low <= high:
            raise ValueError("Thresholds must satisfy 0 <= low <= high")
        return cv2.gapi.Canny(
            image, low, high, int(self.parameters["aperture"]), self.parameters["l2_gradient"]
        )


class Threshold(RegionalOperation, Block):
    region_background = "zero"
    block_type, subtype = "Image processing", "Segmentation"
    type_id, title, category = "cv.threshold", "Threshold", "Processing"
    inputs = (
        *SELECTION_INPUTS,
        Port("value", "number", required=False, required_when_connected=True),
    )
    outputs = (Port("image"), Port("threshold", "number"))
    input_help = {"value": "Overrides the saved threshold, in grayscale pixel units."}
    defaults = {"value": 127.0, "invert": False, "max_value": 255.0, "mode": "binary"}
    parameter_choices = {"mode": tuple(THRESHOLD_MODES)}
    parameter_ranges = {"value": NumericRange(), "max_value": NumericRange(0)}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_inputs.remove("value")

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if p["invert"] and p["mode"] != "binary":
            raise ValidationError(
                "Legacy invert applies to binary mode; otherwise choose an inverse mode",
                "invert",
                "mode",
            )

    def process(self, inputs):
        image = opencv_image(inputs["image"])
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        value = NumericRange().check(inputs.get("value", self.parameters["value"]), "Threshold")
        maximum = float(self.parameters["max_value"])
        name = self.parameters["mode"]
        if self.parameters["invert"]:
            name = "binary_inv"
        mode = THRESHOLD_MODES[name]
        level, mask = cv2.threshold(image, value, maximum, mode)
        return Result({"image": mask, "threshold": level})


class Contours(Block):
    parameter_help = {
        "region_mode": "Center keeps whole objects; clip measures only their selected pixels."
    }
    block_type, subtype = "Inspection", "Shapes"
    type_id, title, category = "inspect.contours", "Find contours", "Inspection"
    inputs = SELECTION_INPUTS
    outputs = (
        Port("overlay"),
        Port("mask"),
        Port("count", "number"),
        Port("data", "data"),
        Port("collection", "data"),
    )
    defaults = {
        "min_area": 100.0,
        "expected_count": 2,
        "retrieval": "external",
        "approximation": "simple",
        "offset": [0, 0],
        "region_mode": "center",
    }
    parameter_choices = {
        "retrieval": ("external", "list", "ccomp", "tree"),
        "approximation": tuple(CONTOUR_APPROXIMATIONS),
        "region_mode": ("center", "clip"),
    }
    parameter_ranges = {
        "min_area": NumericRange(0),
        "expected_count": NumericRange(0, whole=True),
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        message = "Contour offset must be a whole-pixel [x,y] pair"
        try:
            offset = np.asarray(p["offset"], float)
        except (ValueError, TypeError) as exc:
            raise ValidationError(message, "offset") from exc
        if (
            offset.shape != (2,)
            or not np.isfinite(offset).all()
            or np.any(offset != np.rint(offset))
        ):
            raise ValidationError(message, "offset")

    def process(self, inputs):
        image = opencv_image(inputs["image"])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        retrieval = {
            "external": cv2.RETR_EXTERNAL,
            "list": cv2.RETR_LIST,
            "ccomp": cv2.RETR_CCOMP,
            "tree": cv2.RETR_TREE,
        }
        offset = np.asarray(self.parameters["offset"], float)
        selection = selection_mask(inputs, gray)
        search = gray
        if selection is not None and self.parameters["region_mode"] == "clip":
            search = np.where(selection != 0, gray, 0).astype(gray.dtype)
        contours, hierarchy, extraction_info = self.extract_contours(
            search,
            retrieval[self.parameters["retrieval"]],
            CONTOUR_APPROXIMATIONS[self.parameters["approximation"]],
            tuple(offset.astype(int)),
        )
        minimum = float(self.parameters["min_area"])
        expected = int(self.parameters["expected_count"])
        contour_ids = {id(c): index for index, c in enumerate(contours)}
        selected = [c for c in contours if cv2.contourArea(c) >= minimum]
        if selection is not None and self.parameters["region_mode"] == "center":
            retained = []
            for contour in selected:
                moments = cv2.moments(contour)
                center = (
                    [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]
                    if moments["m00"]
                    else contour.reshape(-1, 2).mean(axis=0)
                )
                if contains(selection, np.asarray(center) - offset):
                    retained.append(contour)
            selected = retained
        overlay = (
            cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if self.parameters.get("draw_overlay", True) else NO_VALUE
        )
        mask = np.zeros_like(gray) if self.parameters.get("draw_mask", True) else NO_VALUE
        local = []
        if overlay is not NO_VALUE or mask is not NO_VALUE or self.generate_graphics:
            local = [
                (c - offset.astype(np.int32).reshape(1, 1, 2)).astype(np.int32) for c in selected
            ]
        if overlay is not NO_VALUE:
            cv2.drawContours(overlay, local, -1, (80, 230, 140), 2)
        if mask is not NO_VALUE:
            cv2.drawContours(mask, local, -1, 255, -1)
        count = len(selected)
        score = max(0.0, 1.0 - abs(count - expected) / max(expected, 1))
        data = {
            "count": count,
            "areas": [float(cv2.contourArea(c)) for c in selected],
            "contours": [c.reshape(-1, 2).tolist() for c in selected],
            "offset": offset.tolist(),
            **extraction_info,
        }
        payload = moved_payload(
            self.input_payloads["image"],
            np.array([[1, 0, -offset[0]], [0, 1, -offset[1]], [0, 0, 1]]),
            [gray.shape[1], gray.shape[0]],
        )
        items = [
            contour_item(c, contour_ids[id(c)], hierarchy[0][contour_ids[id(c)]]) for c in selected
        ]
        return Result(
            {
                "overlay": overlay,
                "mask": mask,
                "count": count,
                "data": data,
                "collection": collection("contours", items, gray, payload, offset),
            },
            payloads={"collection": payload, "data": payload},
            graphics={"contours": overlay} if overlay is not NO_VALUE else {},
            drawings={
                "contours": GraphicDrawing(
                    tuple(
                        graphic_object(
                            "polyline", contour.reshape(-1, 2), (80, 230, 140), closed=True
                        )
                        for contour in local
                    )
                )
            } if self.generate_graphics else {},
            passed=count == expected,
            score=score,
            message=f"{count} contours / expected {expected}",
        )

    def extract_contours(self, image, retrieval, approximation, offset):
        contours, hierarchy = cv2.findContours(image, retrieval, approximation, offset=offset)
        return contours, hierarchy, {}


class SaveImage(Block):
    block_type, subtype = "Output", "Images"
    type_id, title, category = "output.image", "Save image", "Output"
    inputs = (Port("image"), Port("metadata", "any", required=False, required_when_connected=True))
    defaults = {"path": "output.png", **CODEC_DEFAULTS}
    parameter_choices = CODEC_CHOICES
    parameter_help = CODEC_HELP
    parameter_ranges = CODEC_RANGES
    input_help = {
        "metadata": "PNG/JPEG ImageMetadata; verified before publishing, existing files preserved.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_inputs = ["image"]

    def process(self, inputs):
        path = Path(self.parameters["path"])
        if path.exists():
            raise FileExistsError("Choose a new path; existing images are preserved")
        if "metadata" in inputs:
            try:
                require_same_frame(
                    [inputs["image"],
                     ValuePacket(inputs["metadata"], self.input_payloads["metadata"])]
                )
            except ValueError as exc:
                raise ValidationError("Metadata and image coordinate frames differ") from exc
            image = codec_image(
                inputs["image"], path.suffix.lower(), self.parameters, allow_signed_tiff=True
            )
            write_metadata_file(
                path, image, codec_options(path.suffix.lower(), self.parameters),
                inputs["metadata"], MAX_ENCODED_BYTES,
            )
            return Result(message=f"Saved {path.name} with verified metadata")
        encoded = encode_image_buffer(
            inputs["image"], path.suffix.lower(), self.parameters, allow_signed_tiff=True
        )
        with path.open("xb") as stream:
            stream.write(encoded.tobytes())
        return Result(message=f"Saved {path.name}")


class WriteResults(Block):
    block_type, subtype = "Output", "Structured data"
    type_id, title, category = "output.results", "Write results", "Output"
    inputs = (Port("data", "data"),)
    defaults = {"path": "results.csv", "format": "csv"}
    parameter_choices = {"format": ("csv", "xml", "sqlite")}

    def process(self, inputs):
        path = Path(self.parameters["path"])
        payload = json.dumps(inputs["data"], allow_nan=False)
        mode = self.parameters["format"].lower()
        if mode == "csv":
            # A fixed JSON payload column prevents schema drift and formula injection.
            with path.open("a+", newline="", encoding="utf-8") as stream:
                stream.seek(0)
                if stream.read(8) not in ("", "payload\n", "payload\r"):
                    raise ValueError("Existing CSV has an incompatible header")
                stream.seek(0, 2)
                writer = csv.writer(stream)
                if stream.tell() == 0:
                    writer.writerow(["payload"])
                writer.writerow([payload])
        elif mode == "xml":
            root = ET.Element("results")
            ET.SubElement(root, "payload").text = payload
            with path.open("xb") as stream:
                stream.write(ET.tostring(root, encoding="utf-8", xml_declaration=True))
        elif mode == "sqlite":
            with sqlite3.connect(path) as database:
                database.execute(
                    "CREATE TABLE IF NOT EXISTS results "
                    "(id INTEGER PRIMARY KEY, payload TEXT NOT NULL)"
                )
                database.execute("INSERT INTO results(payload) VALUES (?)", (payload,))
        else:
            raise ValueError("Format must be csv, xml, or sqlite")
        return Result(message=f"Wrote {mode}: {path.name}")


def builtin_registry():
    from .affine_features import AffineFeatures
    from .aliked_features import ALIKED_BLOCKS
    from .alpha_composition import AlphaOver
    from .analysis_tools import ANALYSIS_BLOCKS
    from .animation_blocks import ANIMATION_BLOCKS
    from .annotation import AnnotationLayer
    from .annoy_blocks import ANNOY_BLOCKS
    from .aruco_dictionary import CreateArucoDictionary, LoadArucoDictionary, SaveArucoDictionary
    from .aruco_board_recovery import RecoverArucoBoard
    from .aruco_markers import DetectArucoMarkers
    from .background_reference import CaptureBackgroundReference, SaveBackgroundReference
    from .basic_processing import BASIC_PROCESSING_BLOCKS
    from .batch_inference import BATCH_INFERENCE_BLOCKS
    from .binary_stereo import BinaryStereoDisparity
    from .binary_stereo_costs import InspectBinaryStereoCosts
    from .blob_detection import SimpleBlobDetection
    from .blur_estimation import EstimateMotionBlur
    from .calibration import (
        CalibrationGrid,
        ColorBalance,
        LensCalibration,
        LoadCalibration,
        PixelToPhysical,
        SaveCalibration,
        Undistort,
    )
    from .calibration_revisions import DeclareCalibrationCapture
    from .calibration_validation import (
        FitPlanarCalibration,
        InspectPlanarDistance,
        ValidateLensCalibration,
        ValidatePlanarCalibration,
    )
    from .calipers import BeadCalipers
    from .camera_health import CameraHealth
    from .camera_pair_registration import RegisterCameraPair
    from .camera_projector import CAMERA_PROJECTOR_BLOCKS
    from .cascade_detection import DetectCascade
    from .channel_transform import ChannelTransform, RouteChannels
    from .charuco import DetectCharucoBoard
    from .chessboard_quality import CheckChessboardQuality
    from .chromatic_aberration import ChromaticModel, CorrectChromaticAberration, LoadChromaticModel
    from .circle_fitting import FitCircleArc
    from .classification_evaluation import CLASSIFICATION_EVALUATION_BLOCKS
    from .cloud_cylinder import MeasureCloudCylinder
    from .cloud_datum import MeasureCloudDatum
    from .cloud_extents import MeasureOrientedCloud
    from .cloud_features import RegisterCloudFeatures
    from .cloud_fixture import PerturbCloudFixture
    from .cloud_io import CLOUD_IO_BLOCKS
    from .cloud_recording import RecordedCloud
    from .cloud_merge import MergeCloudScans
    from .cloud_octree import QueryCloudOctree
    from .cloud_clustering import ClusterPointCloud
    from .cloud_model_fitting import FitCloudModel
    from .cloud_comparison import CompareCloudDimensions
    from .cloud_outliers import RejectCloudOutliers
    from .cloud_tiles import TiledCloudRadius
    from .cloud_region_growing import GrowCloudRegions
    from .cloud_surface_quality import MeasureCloudSurfaceQuality
    from .cloud_projection import ProjectCloudVisibility
    from .cloud_section import MeasureCloudSection
    from .cloud_sampling import SampleCloud
    from .cloud_tools import CLOUD_BLOCKS
    from .cluster_models import CLUSTER_MODEL_BLOCKS
    from .clustering import KMeansImage, KMeansSamples
    from .code_readers import CODE_READER_BLOCKS
    from .codec_probe import ProbeImageFile
    from .coded_rig_observations import LoadCodedRigObservations
    from .collection_tools import COLLECTION_BLOCKS
    from .collinear_rectification import RectifyCollinearCameras
    from .color_calibration import COLOR_CALIBRATION_BLOCKS
    from .color_measurement import COLOR_MEASUREMENT_BLOCKS
    from .color_tsdf import FuseColorTSDF, PackPosedRGBD
    from .color_tracking import ColorDistributionTracker
    from .confidence_calibration import CONFIDENCE_BLOCKS
    from .connection_health import CONNECTION_HEALTH_BLOCKS
    from .contour_analysis import CONTOUR_ANALYSIS_BLOCKS
    from .contour_raster import RasterizeContours
    from .control_flow import CONTROL_BLOCKS
    from .convex_overlap import ConvexPolygonOverlap
    from .conveyor_tracking import ConveyorDemo, TrackConveyorParts
    from .coordination import CoordinateInputs, CounterInput
    from .depth import RGBD, DepthFile, DepthSample, FillDepthHoles
    from .depth_components import MeasureDepthComponents
    from .depth_evaluation import EvaluateDepth
    from .depth_fusion import FuseRegisteredDepth
    from .depth_encoding import NormalizeMetricDepth, QuantizeMetricDepth
    from .depth_plane import SeparateDepthPlane
    from .depth_region_projection import ReprojectDepthRegion
    from .depth_silhouette import MeasureDepthSilhouette
    from .depth_smoothing import SmoothDepth
    from .depth_trajectory import TrackDepthFusion
    from .detection_decode import DecodeDetections
    from .detection_evaluation import EvaluateDetections
    from .detection_fusion import FuseTileDetections
    from .detection_votes import GroupDetectionVotes
    from .differential_exposure import DifferentialExposure
    from .disk_features import DISK_BLOCKS
    from .disparity_evaluation import EvaluateDisparity
    from .disparity_speckles import FilterDisparitySpeckles
    from .disparity_wls import FilterExternalDisparity
    from .distortion_fixture import DistortImageFixture
    from .dnn_blocks import DNN_BLOCKS
    from .dnn_profiling import DNN_PROFILING_BLOCKS
    from .ecc_registration import AlignECC
    from .edge_agreement import CompareEdgeMaps
    from .epipolar import CheckEpipolarCorrespondences
    from .epipolar_refinement import RefineEpipolarCorrespondences
    from .example_plugins import invert as invert_example
    from .example_plugins import mean_inspection as mean_example
    from .examples3d import StereoDemo
    from .exposure import AlignExposures, FuseExposures
    from .fanuc import FANUC_BLOCKS
    from .feature_covariance import (
        FitFeatureCovariance,
        LoadFeatureCovariance,
        MahalanobisDistance,
        SaveFeatureCovariance,
    )
    from .feature_evaluation import EvaluateImageFeatures
    from .features import FEATURE_BLOCKS
    from .fiducial_targets import GenerateFiducialTarget
    from .fisheye_calibration import FisheyeCalibration
    from .flat_field_reference import BuildFlatFieldReference
    from .flow_files import SaveFlowFile
    from .flow_refinement import RefineDenseFlow
    from .flow_warp import LoadFlowFile, WarpDenseFlow
    from .focus import MeasureFocus
    from .focus_stacking import FocusStack
    from .foreground_debounce import ForegroundDebounce
    from .fourier_components import FourierComponents, ReconstructFourierImage
    from .fourier_correlation import FourierCorrelation
    from .frame_interpolation import InterpolateFrames
    from .frequency_response import MeasureFrequencyResponse
    from .generalized_hough import FindGuilTemplate, FindHoughTemplate
    from .geometry3d import GEOMETRY3D_BLOCKS
    from .gram_matrix import GramMatrix
    from .gray_level_cooccurrence import GrayLevelCooccurrence
    from .hand_eye import HandEyeCalibration
    from .hash_tsdf import FuseHashedTSDF
    from .hdr import (
        CalibrateCameraResponse,
        CalibrateRobertson,
        MergeHDR,
        MergeRobertson,
        ToneMapHDR,
    )
    from .hdr_motion import MergeMotionHDR
    from .histogram import Histogram
    from .histogram_comparison import CompareHistograms, HistogramBackProjection
    from .hit_miss import HitOrMiss
    from .hog import HOGFeatures
    from .hog_detection import HOG_DETECTION_BLOCKS
    from .hog_training import TrainHOGDetector
    from .illumination import AdaptiveBackgroundCorrection, FlatFieldCorrection
    from .image_blend import BlendWeightMaps
    from .image_codecs import DecodeImage, EncodeImage
    from .image_colorize import ColorizeScalar
    from .image_correlation import CompareAlignedImages
    from .image_hash import CompareImageHashes, ComputeImageHash
    from .image_layout import JoinImages, PasteImagePatch, TileImage
    from .image_mixtures import IMAGE_EM_BLOCKS
    from .image_noise import AddImageNoise, RandomImage
    from .image_norm import MeasureImageNorm
    from .image_quality import CompareImageQuality, MeasureBRISQUE
    from .image_queue import ImageQueue
    from .image_statistics import ImageSummaryStatistics
    from .image_stitching import StitchImagePair
    from .image_tracking import TrackImageParts
    from .image_validity import CheckImageValidity, ReplaceNonfiniteSamples
    from .input_shifts import INPUT_SHIFT_BLOCKS
    from .instance_segmentation import INSTANCE_BLOCKS
    from .instance_tracking import TrackInstanceMasks
    from .judgments import FinalJudgment
    from .knn_blocks import KNN_BLOCKS
    from .label_measurement import MaskFromLabelCollection, MeasureLabelRegions
    from .landmarks import LANDMARK_BLOCKS
    from .laplacian_pyramids import LAPLACIAN_PYRAMID_BLOCKS
    from .learned_flow import LearnedFlow
    from .learned_restoration import RestoreImage
    from .learned_stereo import LearnedStereo
    from .learned_tracking import VitSequence
    from .lens_map_files import LoadLensMaps, SaveLensMaps
    from .lightglue_blocks import LIGHTGLUE_BLOCKS
    from .line_pair import MeasureLinePair
    from .linear_program import SolveLinearProgram
    from .linear_system import SolveLinearSystem
    from .linemod_pose import MatchLINEMODPose, TrainLINEMODPose
    from .local_anomaly import LOCAL_ANOMALY_BLOCKS
    from .local_binary_patterns import LocalBinaryPatterns
    from .local_features import LOCAL_FEATURE_BLOCKS
    from .local_threshold import LocalThreshold
    from .logix import LOGIX_BLOCKS
    from .logpolar_registration import LogPolarRegistration
    from .mask_boundary import CompareMaskBoundary
    from .mask_clearance import MaskClearance
    from .mask_resampling import ResampleMask
    from .match_review import ReviewFeatureMatches
    from .matched_transform import EstimateMatchedTransform
    from .matrix_eigen import MatrixEigen
    from .matrix_inverse import InvertMatrix
    from .matrix_multiply import MultiplyMatrices
    from .matrix_structure import (
        CompleteSymmetricMatrix,
        CreateDiagonalMatrix,
        MeasureMatrixDiagonal,
    )
    from .marked_checkerboard import DetectMarkedCheckerboard
    from .matrix_svd import MatrixSVD
    from .metadata_blocks import CreateImageMetadata
    from .mil_tracking import MILSequence
    from .monocular_depth import MonocularDepth
    from .mosaic_layout_io import LoadMosaicLayout, SaveMosaicLayout
    from .motion_trajectory import CaptureMotionTrajectory, ReplayMotionTrajectory
    from .motion_trajectory_io import LoadMotionTrajectory, SaveMotionTrajectory
    from .mqtt import MQTT_BLOCKS
    from .multi_inference import MULTI_INFERENCE_BLOCKS
    from .multicamera_calibration import (
        CalibrateMulticameraRig,
        LoadMulticameraObservations,
        ValidateMulticameraRig,
    )
    from .multipage_blocks import MULTIPAGE_BLOCKS
    from .multiscale_ridges import MultiscaleRidges
    from .multiscale_super_resolution import MultiScaleSuperResolution
    from .native_background import NATIVE_BACKGROUND_BLOCKS
    from .native_index import NATIVE_INDEX_BLOCKS
    from .native_lightglue import NATIVE_LIGHTGLUE_BLOCKS
    from .native_trackers import NATIVE_TRACKER_BLOCKS
    from .nearest_rows import FindNearestRows
    from .neighbor_restoration import RestoreFromNeighbors
    from .network_video import NetworkVideo
    from .ocr import OCR_BLOCKS
    from .ocr_batch import RecognizeTextBatch
    from .ocr_constraints import RecognizeConstrainedText
    from .offline_stabilization import StabilizeRecordedSequence
    from .omnidir_calibration import OmnidirectionalCalibration
    from .one_class_svm import ONE_CLASS_SVM_BLOCKS
    from .organized_mesh import MeshOrganizedDepth
    from .organized_planes import FindOrganizedPlanes
    from .oriented_edges import FindOrientedEdges
    from .overlay_graphics import OverlayResultGraphics, SetFilmstripImage
    from .paced_video import PacedVideoFile
    from .panorama_points import PanoramaPoints
    from .parallel_inspection import ParallelInspect, RepeatedPartsDemo
    from .path_continuity import InspectPathContinuity
    from .path_strip import UnwrapPathStrip
    from .patterns import MatchShapePattern, PatternDemoImages, TrainShapePattern
    from .phase_shift import PHASE_SHIFT_BLOCKS
    from .phase_unwrap import UnwrapFringePhase
    from .photo_filters import PHOTO_FILTER_BLOCKS
    from .pick_selection import SelectPickCandidate
    from .pixel_moments import PixelMoments
    from .planar_motion import RecoverPlanarCameraMotion
    from .planar_pose import PlanarPoseHypotheses
    from .plugins import (
        BIN_PICKING_TYPES,
        GEOMETRIC_MATCHER_TYPES,
        LOCATE_3D_TYPES,
        MEASURE_3D_TYPES,
        PATH_WIDTH_TYPES,
        PIXEL_SEGMENTATION_TYPES,
        REFERENCE_INSPECT_TYPES,
        load_installed_3d_locate,
        load_installed_3d_measure,
        load_installed_bin_picking,
        load_installed_geometric_matcher,
        load_installed_path_width,
        load_installed_pixel_segmentation,
        load_installed_reference_inspect,
        register_simple_providers,
    )
    from .point_set_lines import HoughPointSetLines
    from .polar_image import PolarImageWarp
    from .projection_matrix import InspectProjectionMatrix
    from .projector_rectification import InverseRectifyProjectorPattern
    from .polynomial_roots import SolvePolynomialRoots
    from .pose_crop import CropPose
    from .poses import POSE_BLOCKS
    from .principal_components import (
        FitPrincipalComponents,
        LoadPrincipalComponents,
        SavePrincipalComponents,
        TransformPrincipalComponents,
    )
    from .processing import PROCESSING_BLOCKS
    from .profile_resampling import ResampleImageProfile
    from .projection import IntensityProfile, PathIntensityProfile
    from .prompt_mask_decode import DecodePromptMasks
    from .prompt_segmentation import EfficientSAMInference, SegmentationPrompts
    from .ptcloud3d import PTCLOUD3D_BLOCKS
    from .pyramids import PYRAMID_BLOCKS
    from .qr_append import ReadQRAppend
    from .qr_encoding import EncodeQRCode
    from .quasi_dense_stereo import QuasiDenseDisparity
    from .raster_hough_lines import HoughPolarLines
    from .raw_sensor import RAW_SENSOR_BLOCKS
    from .recipe_blocks import RecipeField, RecipeSelect
    from .refresh_tracking import REFRESH_TRACKING_BLOCKS
    from .region_contrast import MeasureRegionContrast
    from .region_proposals import GraphSegments, SelectiveSearchProposals
    from .regions import REGION_BLOCKS
    from .registered_replay import ReplayRegisteredViews
    from .registration import REGISTRATION_BLOCKS
    from .regression import REGRESSION_BLOCKS
    from .remap_maps import BuildLensMaps, LensMapDisplacement, UndistortLensPoints
    from .restoration import RESTORATION_BLOCKS
    from .rgbd_recording import RecordedRGBD
    from .rgbd_registration import RegisterRGBD
    from .rgbd_warp import WarpRGBDByPose
    from .ridge_response import RidgeResponse
    from .robot_world import RobotWorldHandEye
    from .ros2_blocks import ROS2_BLOCKS
    from .rotated_detection import DecodeRotatedDetections
    from .rotating_panorama import StitchRotatingPanorama
    from .rtrees_blocks import RTREES_BLOCKS
    from .run_contours import RunContours
    from .scan_stitching import StitchScanSequence
    from .scanline_motion import EstimateScanlineMotion
    from .scanner_range_image import ScannerRangeImage
    from .scene_scale import RecoverSceneScale
    from .segmentation import SEGMENTATION_BLOCKS
    from .segmentation_fusion import FuseTileSegmentation
    from .selection_remap import RemapSelection, RemapSelectionPacked
    from .semantic_segmentation import DecodeSegmentation
    from .sequence_tools import SEQUENCE_BLOCKS
    from .shadow_compensation import CompensateReferenceShadows
    from .shape_fitting import (
        BoundingConvexPolygon,
        BoundingRotatedRectangle,
        FitShape,
        InspectWorkplaneEllipse,
        InspectWorkplaneRectangle,
        RotatedRectangleOverlap,
    )
    from .silhouette_tracking import TrackObjectSilhouette
    from .simple_classifier.dataflow import ClassifyImage
    from .simpleitk_blocks import SIMPLEITK_BLOCKS
    from .skeleton import PruneSkeleton, Skeletonize
    from .skimage_regions import RegionProperties
    from .skimage_thresholds import GlobalThreshold, MultiOtsu
    from .skimage_thresholds import LocalThreshold as SkimageLocalThreshold
    from .slanted_edge import MeasureSlantedEdge
    from .soft_nms import SuppressDetections
    from .spanning_tree import MinimumSpanningTree
    from .sparse_multiview import MakeLandmarkView, ReconstructSparseViews
    from .spectral import SPECTRAL_BLOCKS
    from .stabilization import StabilizeCameraMotion
    from .stabilization_fill import FillStabilizationBorders
    from .stack_blur import StackBlur
    from .sparse_depth import BackprojectDepthPixels
    from .stereo3d import STEREO3D_BLOCKS
    from .stereo_color import MatchStereoColor
    from .stereo_rectified_camera import StereoRectifiedCamera
    from .stereo_validation import ValidateStereoCalibration
    from .stream_stabilization import StabilizeStream
    from .strip_defects import GroupStripDefects, JudgeStripDefects
    from .strip_graphics import MapStripGraphics
    from .strip_width import InspectStripWidth
    from .structure_tensor import StructureTensor
    from .subpixel_crop import CropSubpixel
    from .super_resolution import SuperResolution
    from .superpixels import SuperpixelsLSC, SuperpixelsSEEDS, SuperpixelsSLIC
    from .superpoint_features import SuperPointONNXFeatures
    from .svm_blocks import SVM_BLOCKS
    from .table_sort import SortNumericTable
    from .target_calibration import LoadTargetCalibrationObservations, RefineCalibrationTarget
    from .template_instances import FindTemplateInstances
    from .tensor_comparison import CompareImageTensors
    from .text_rules import VerifyText
    from .texture_patterns import MatchTexturePattern, TrainTexturePattern
    from .tiff_files import TiffFile, WriteTiffFile
    from .tiff_page_access import TiffPageAccess
    from .tiled_mosaic import RegisterScanLayout, RenderMosaicTile
    from .tilted_statistics import TiltedWindowStatistics
    from .tof_depth_cleanup import CleanToFDepth
    from .tracking_appearance import DescribeBoxAppearance
    from .translation_registration import EstimateTranslation
    from .triangulation import DelaunayTriangulation
    from .mesh_comparison import CompareScanToMesh
    from .mesh_inspection import InspectTriangleMesh
    from .mesh_projection import ProjectMesh
    from .mesh_simplification import SimplifyTriangleMesh
    from .mesh_texture import BakeMeshTexture
    from .tsdf_mesh import TSDF_MESH_BLOCKS
    from .uncalibrated_stereo import RectifyUncalibratedStereo
    from .video_buffer import DecodeVideo
    from .video_files import VideoFile, WriteVideo
    from .video_time import VideoAtTime
    from .video_timing import VideoTiming
    from .visible_support import VisibleSupport
    from .vision3d_modules import VISION3D_MODULE_BLOCKS
    from .visual_words import VISUAL_WORD_BLOCKS
    from .voronoi import VoronoiCells
    from .watershed_seeds import WatershedMaskSeeds, WatershedPointSeeds
    from .weighted_moments import IntensityMoments
    from .wide_stereo import CalibrateWideStereo
    from .workplane import RectifyWorkplane
    from .yuv_formats import DecodePackedYuv, RawYuvFile
    from .zxing_reader import ReadZXingBarcode

    registry = Registry()
    for block in (
        *ANIMATION_BLOCKS,
        ProbeImageFile,
        *MULTIPAGE_BLOCKS,
        HitOrMiss,
        Comment,
        DemoImage,
        RandomImage,
        *LOGIX_BLOCKS,
        *FANUC_BLOCKS,
        *MQTT_BLOCKS,
        *CONNECTION_HEALTH_BLOCKS,
        *ROS2_BLOCKS,
        ImageFile,
        VideoFile,
        RecordedCapture,
        PacedVideoFile,
        NetworkVideo,
        VideoAtTime,
        VideoTiming,
        RemapSelection,
        RemapSelectionPacked,
        StitchRotatingPanorama,
        PanoramaPoints,
        VisibleSupport,
        DecodeVideo,
        TiffFile,
        TiffPageAccess,
        RawYuvFile,
        *RAW_SENSOR_BLOCKS,
        Camera,
        Grayscale,
        Blur,
        Canny,
        *BASIC_PROCESSING_BLOCKS,
        StackBlur,
        DecodePackedYuv,
        EncodeImage,
        DecodeImage,
        CreateImageMetadata,
        AnnotationLayer,
        OverlayResultGraphics,
        SetFilmstripImage,
        Threshold,
        Contours,
        RunContours,
        SaveImage,
        WriteTiffFile,
        WriteVideo,
        WriteResults,
        Histogram,
        HOGFeatures,
        *HOG_DETECTION_BLOCKS,
        TrainHOGDetector,
        DetectCascade,
        CalibrationGrid,
        DetectArucoMarkers,
        CreateArucoDictionary,
        LoadArucoDictionary,
        SaveArucoDictionary,
        DetectCharucoBoard,
        DetectMarkedCheckerboard,
        GenerateFiducialTarget,
        LensCalibration,
        RegisterCameraPair,
        LoadTargetCalibrationObservations,
        RefineCalibrationTarget,
        ValidateLensCalibration,
        CheckChessboardQuality,
        HandEyeCalibration,
        RobotWorldHandEye,
        BuildLensMaps,
        InverseRectifyProjectorPattern,
        LensMapDisplacement,
        UndistortLensPoints,
        LoadLensMaps,
        SaveLensMaps,
        Undistort,
        ColorBalance,
        *COLOR_CALIBRATION_BLOCKS,
        PixelToPhysical,
        InspectPlanarDistance,
        FitPlanarCalibration,
        RectifyWorkplane,
        ValidatePlanarCalibration,
        DepthFile,
        RGBD,
        DepthSample,
        FillDepthHoles,
        SmoothDepth,
        NormalizeMetricDepth,
        QuantizeMetricDepth,
        CleanToFDepth,
        EvaluateDepth,
        FuseRegisteredDepth,
        ReprojectDepthRegion,
        SeparateDepthPlane,
        MeasureDepthComponents,
        MeasureDepthSilhouette,
        LoadCalibration,
        SaveCalibration,
        *PROCESSING_BLOCKS,
        LocalThreshold,
        LocalBinaryPatterns,
        GrayLevelCooccurrence,
        JoinImages,
        StitchImagePair,
        StitchScanSequence,
        RegisterScanLayout,
        LoadMosaicLayout,
        SaveMosaicLayout,
        CaptureMotionTrajectory,
        ReplayMotionTrajectory,
        LoadMotionTrajectory,
        SaveMotionTrajectory,
        RestoreFromNeighbors,
        FillStabilizationBorders,
        EstimateMotionBlur,
        EstimateScanlineMotion,
        TrackImageParts,
        DescribeBoxAppearance,
        ResampleImageProfile,
        *REFRESH_TRACKING_BLOCKS,
        CaptureBackgroundReference,
        SaveBackgroundReference,
        ReplayRegisteredViews,
        RenderMosaicTile,
        PasteImagePatch,
        CropSubpixel,
        CropPose,
        PolarImageWarp,
        UnwrapPathStrip,
        MapStripGraphics,
        TileImage,
        *FEATURE_BLOCKS,
        *REGISTRATION_BLOCKS,
        AffineFeatures,
        ReviewFeatureMatches,
        AlignECC,
        *RESTORATION_BLOCKS,
        *PHOTO_FILTER_BLOCKS,
        AlignExposures,
        FuseExposures,
        CalibrateCameraResponse,
        CalibrateRobertson,
        MergeHDR,
        MergeRobertson,
        MergeMotionHDR,
        ToneMapHDR,
        *SPECTRAL_BLOCKS,
        FourierComponents,
        ReconstructFourierImage,
        FourierCorrelation,
        MeasureFrequencyResponse,
        *ANALYSIS_BLOCKS,
        BlendWeightMaps,
        AlphaOver,
        InterpolateFrames,
        RefineDenseFlow,
        LoadFlowFile,
        SaveFlowFile,
        WarpDenseFlow,
        AddImageNoise,
        DistortImageFixture,
        ColorizeScalar,
        FindHoughTemplate,
        FindGuilTemplate,
        ChannelTransform,
        RouteChannels,
        IntensityProfile,
        PathIntensityProfile,
        InspectPathContinuity,
        ImageSummaryStatistics,
        MeasureRegionContrast,
        IntensityMoments,
        MeasureImageNorm,
        CheckImageValidity,
        ReplaceNonfiniteSamples,
        TiltedWindowStatistics,
        EstimateTranslation,
        *REGION_BLOCKS,
        *PYRAMID_BLOCKS,
        *LAPLACIAN_PYRAMID_BLOCKS,
        *COLLECTION_BLOCKS,
        SimpleBlobDetection,
        HoughPointSetLines,
        HoughPolarLines,
        *CONTOUR_ANALYSIS_BLOCKS,
        RasterizeContours,
        FitCircleArc,
        CompareEdgeMaps,
        *CODE_READER_BLOCKS,
        EncodeQRCode,
        ReadQRAppend,
        ReadZXingBarcode,
        VerifyText,
        *SEGMENTATION_BLOCKS,
        MeasureLabelRegions,
        RegionProperties,
        ChromaticModel,
        LoadChromaticModel,
        CorrectChromaticAberration,
        MultiOtsu,
        GlobalThreshold,
        SkimageLocalThreshold,
        MaskFromLabelCollection,
        GraphSegments,
        RidgeResponse,
        StructureTensor,
        MultiscaleRidges,
        SuperpixelsSLIC,
        SuperpixelsLSC,
        SuperpixelsSEEDS,
        SelectiveSearchProposals,
        Skeletonize,
        PruneSkeleton,
        WatershedPointSeeds,
        WatershedMaskSeeds,
        FlatFieldCorrection,
        CompensateReferenceShadows,
        BuildFlatFieldReference,
        AdaptiveBackgroundCorrection,
        CompareImageQuality,
        CompareMaskBoundary,
        CompareAlignedImages,
        DifferentialExposure,
        CameraHealth,
        MeasureFocus,
        LogPolarRegistration,
        FocusStack,
        MeasureSlantedEdge,
        ComputeImageHash,
        CompareImageHashes,
        MeasureBRISQUE,
        CompareHistograms,
        *COLOR_MEASUREMENT_BLOCKS,
        ClassifyImage,
        HistogramBackProjection,
        FitShape,
        BoundingRotatedRectangle,
        InspectWorkplaneRectangle,
        InspectWorkplaneEllipse,
        MeasureLinePair,
        BoundingConvexPolygon,
        DelaunayTriangulation,
        VoronoiCells,
        MinimumSpanningTree,
        RotatedRectangleOverlap,
        ConvexPolygonOverlap,
        FindTemplateInstances,
        SelectPickCandidate,
        BeadCalipers,
        MaskClearance,
        ResampleMask,
        InspectStripWidth,
        GroupStripDefects,
        JudgeStripDefects,
        TrainShapePattern,
        MatchShapePattern,
        TrainTexturePattern,
        MatchTexturePattern,
        FindOrientedEdges,
        PatternDemoImages,
        *POSE_BLOCKS,
        *CONTROL_BLOCKS,
        CoordinateInputs,
        CounterInput,
        ParallelInspect,
        RepeatedPartsDemo,
        ImageQueue,
        RecipeSelect,
        RecipeField,
        *SEQUENCE_BLOCKS,
        *SIMPLEITK_BLOCKS,
        ForegroundDebounce,
        StabilizeCameraMotion,
        StabilizeRecordedSequence,
        StabilizeStream,
        *NATIVE_BACKGROUND_BLOCKS,
        *NATIVE_TRACKER_BLOCKS,
        MILSequence,
        ColorDistributionTracker,
        PixelMoments,
        TrackConveyorParts,
        ConveyorDemo,
        FinalJudgment,
        *DNN_BLOCKS,
        *DNN_PROFILING_BLOCKS,
        *REGRESSION_BLOCKS,
        *LOCAL_ANOMALY_BLOCKS,
        *INPUT_SHIFT_BLOCKS,
        *BATCH_INFERENCE_BLOCKS,
        *MULTI_INFERENCE_BLOCKS,
        DecodeDetections,
        FuseTileDetections,
        SuppressDetections,
        GroupDetectionVotes,
        CompareImageTensors,
        DecodeRotatedDetections,
        DecodeSegmentation,
        FuseTileSegmentation,
        SegmentationPrompts,
        EfficientSAMInference,
        DecodePromptMasks,
        *INSTANCE_BLOCKS,
        TrackInstanceMasks,
        *LANDMARK_BLOCKS,
        FitFeatureCovariance,
        LoadFeatureCovariance,
        SaveFeatureCovariance,
        EvaluateImageFeatures,
        SolveLinearSystem,
        SolveLinearProgram,
        SolvePolynomialRoots,
        CreateDiagonalMatrix,
        CompleteSymmetricMatrix,
        MeasureMatrixDiagonal,
        MultiplyMatrices,
        InvertMatrix,
        MatrixSVD,
        MatrixEigen,
        GramMatrix,
        FindNearestRows,
        SortNumericTable,
        FitPrincipalComponents,
        TransformPrincipalComponents,
        SavePrincipalComponents,
        LoadPrincipalComponents,
        MahalanobisDistance,
        KMeansSamples,
        KMeansImage,
        *CLUSTER_MODEL_BLOCKS,
        *VISUAL_WORD_BLOCKS,
        *IMAGE_EM_BLOCKS,
        *OCR_BLOCKS,
        RecognizeTextBatch,
        RecognizeConstrainedText,
        SuperResolution,
        MultiScaleSuperResolution,
        RestoreImage,
        MonocularDepth,
        LearnedStereo,
        LearnedFlow,
        VitSequence,
        *ANNOY_BLOCKS,
        *NATIVE_INDEX_BLOCKS,
        *SVM_BLOCKS,
        *ONE_CLASS_SVM_BLOCKS,
        *KNN_BLOCKS,
        *RTREES_BLOCKS,
        *CLASSIFICATION_EVALUATION_BLOCKS,
        *CONFIDENCE_BLOCKS,
        EvaluateDetections,
        *LIGHTGLUE_BLOCKS,
        *LOCAL_FEATURE_BLOCKS,
        SuperPointONNXFeatures,
        *NATIVE_LIGHTGLUE_BLOCKS,
        *DISK_BLOCKS,
        *ALIKED_BLOCKS,
        *GEOMETRY3D_BLOCKS,
        InspectProjectionMatrix,
        PlanarPoseHypotheses,
        RecoverArucoBoard,
        RecoverPlanarCameraMotion,
        MakeLandmarkView,
        ReconstructSparseViews,
        RecoverSceneScale,
        EstimateMatchedTransform,
        DeclareCalibrationCapture,
        FisheyeCalibration,
        OmnidirectionalCalibration,
        CheckEpipolarCorrespondences,
        RefineEpipolarCorrespondences,
        RectifyUncalibratedStereo,
        ValidateStereoCalibration,
        LoadMulticameraObservations,
        LoadCodedRigObservations,
        CalibrateMulticameraRig,
        ValidateMulticameraRig,
        *STEREO3D_BLOCKS,
        BackprojectDepthPixels,
        MatchStereoColor,
        StereoRectifiedCamera,
        RectifyCollinearCameras,
        RecordedRGBD,
        RegisterRGBD,
        WarpRGBDByPose,
        BinaryStereoDisparity,
        InspectBinaryStereoCosts,
        QuasiDenseDisparity,
        FilterDisparitySpeckles,
        FilterExternalDisparity,
        EvaluateDisparity,
        CalibrateWideStereo,
        *CLOUD_BLOCKS,
        SampleCloud,
        ScannerRangeImage,
        PerturbCloudFixture,
        RegisterCloudFeatures,
        *CLOUD_IO_BLOCKS,
        RecordedCloud,
        QueryCloudOctree,
        RejectCloudOutliers,
        TiledCloudRadius,
        CompareCloudDimensions,
        MeasureOrientedCloud,
        MeasureCloudSection,
        MeasureCloudDatum,
        MergeCloudScans,
        ClusterPointCloud,
        GrowCloudRegions,
        FitCloudModel,
        MeasureCloudCylinder,
        MeasureCloudSurfaceQuality,
        ProjectCloudVisibility,
        *VISION3D_MODULE_BLOCKS,
        *PTCLOUD3D_BLOCKS,
        FuseHashedTSDF,
        PackPosedRGBD,
        FuseColorTSDF,
        TrackDepthFusion,
        *PHASE_SHIFT_BLOCKS,
        UnwrapFringePhase,
        TrackObjectSilhouette,
        TrainLINEMODPose,
        MatchLINEMODPose,
        *CAMERA_PROJECTOR_BLOCKS,
        *TSDF_MESH_BLOCKS,
        ProjectMesh,
        BakeMeshTexture,
        InspectTriangleMesh,
        SimplifyTriangleMesh,
        MeshOrganizedDepth,
        FindOrganizedPlanes,
        CompareScanToMesh,
        StereoDemo,
    ):
        registry.register(block)
    registry.add_provider(GEOMETRIC_MATCHER_TYPES, load_installed_geometric_matcher)
    registry.add_provider(PIXEL_SEGMENTATION_TYPES, load_installed_pixel_segmentation)
    registry.add_provider(PATH_WIDTH_TYPES, load_installed_path_width)
    registry.add_provider(REFERENCE_INSPECT_TYPES, load_installed_reference_inspect)
    registry.add_provider(LOCATE_3D_TYPES, load_installed_3d_locate)
    registry.add_provider(MEASURE_3D_TYPES, load_installed_3d_measure)
    registry.add_provider(BIN_PICKING_TYPES, load_installed_bin_picking)
    register_simple_providers(registry)
    registry.add_provider((invert_example.Invert.type_id,), invert_example.register)
    registry.add_provider((mean_example.MeanInspection.type_id,), mean_example.register)
    return registry


def demo_graph(registry):
    graph = Graph()
    chain = [
        graph.add(registry.create(t))
        for t in ("input.demo", "cv.gray", "cv.gaussian_blur", "cv.threshold", "inspect.contours")
    ]
    chain[3].parameters["value"] = 100
    chain[4].parameters["min_area"] = 1000.0
    for index, block in enumerate(chain):
        block.position = [index * 265.0, (index % 2) * 45.0]
    for source, target in zip(chain, chain[1:]):
        graph.connect(source.id, "image", target.id, "image")
    return graph
