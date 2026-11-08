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

from .core import Block, Graph, Port, Registry, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .image_codecs import CODEC_CHOICES, CODEC_DEFAULTS, CODEC_HELP, codec_options
from .image_metadata import decode_file_image
from .images import COLOR_FORMATS, PIXEL_DEPTHS, format_image, image_format, opencv_image
from .objects import collection, contour_item
from .parameter_validation import NumericRange, ValidationError
from .parameters import BORDERS, CONTOUR_APPROXIMATIONS, THRESHOLD_MODES, scalar_value
from .payloads import NO_VALUE, moved_payload
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
    outputs = (Port("image"), Port("info", "data"))
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
            return Result({"image": NO_VALUE, "info": NO_VALUE}, message="Image sequence complete")
        path = Path(selected)
        buffer = self.read_bytes(path)
        image, metadata = decode_file_image(buffer, self.parameters["metadata_mode"])
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
            {"image": output, "info": info},
            graphics={"decoded source": image},
            message=f"{path.name} · {info['color_format']} / {output.dtype}",
        )

    def read_bytes(self, path):
        return path.read_bytes()


class Camera(Block):
    block_type, subtype = "Input", "Cameras"
    type_id, title, category = "input.camera", "Camera", "Input"
    outputs = (Port("image"),)
    defaults = {
        "device": 0,
        "backend": "auto",
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "buffer_size": 0,
    }
    parameter_choices = {
        "backend": ("auto", "dshow", "msmf", "v4l2", "gstreamer", "ffmpeg", "avfoundation")
    }
    parameter_ranges = {
        **{key: NumericRange(0, 16384, whole=True) for key in ("width", "height", "buffer_size")},
        "fps": NumericRange(0),
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._capture = None
        self._device = None

    def process(self, inputs):
        device = int(self.parameters["device"])
        p = self.parameters
        width, height, buffer_size = (int(p[k]) for k in ("width", "height", "buffer_size"))
        fps = float(p["fps"])
        settings = (device, p["backend"], width, height, fps, buffer_size)
        if self._capture is None or settings != self._device:
            self.close()
            backends = {
                "auto": cv2.CAP_ANY,
                "dshow": cv2.CAP_DSHOW,
                "msmf": cv2.CAP_MSMF,
                "v4l2": cv2.CAP_V4L2,
                "gstreamer": cv2.CAP_GSTREAMER,
                "ffmpeg": cv2.CAP_FFMPEG,
                "avfoundation": cv2.CAP_AVFOUNDATION,
            }
            self._capture = (
                cv2.VideoCapture(device)
                if p["backend"] == "auto"
                else cv2.VideoCapture(device, backends[p["backend"]])
            )
            self._device = settings
            for prop, value in (
                (cv2.CAP_PROP_FRAME_WIDTH, width),
                (cv2.CAP_PROP_FRAME_HEIGHT, height),
                (cv2.CAP_PROP_FPS, fps),
                (cv2.CAP_PROP_BUFFERSIZE, buffer_size),
            ):
                if value and not self._capture.set(prop, value):
                    self.close()
                    raise ValueError("Camera backend rejected a requested acquisition setting")
        if not self._capture.isOpened():
            self.close()
            raise RuntimeError(f"Cannot open camera {device}")
        ok, image = self._capture.read()
        if not ok:
            raise RuntimeError("Camera did not return a frame")
        return Result({"image": image}, message=f"Camera {device}")

    def close(self):
        if self._capture is not None:
            self._capture.release()
            self._capture = None


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
        contours, hierarchy = cv2.findContours(
            search,
            retrieval[self.parameters["retrieval"]],
            CONTOUR_APPROXIMATIONS[self.parameters["approximation"]],
            offset=tuple(offset.astype(int)),
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
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        mask = np.zeros_like(gray)
        local = [(c - offset.astype(np.int32).reshape(1, 1, 2)).astype(np.int32) for c in selected]
        cv2.drawContours(overlay, local, -1, (80, 230, 140), 2)
        cv2.drawContours(mask, local, -1, 255, -1)
        count = len(selected)
        score = max(0.0, 1.0 - abs(count - expected) / max(expected, 1))
        data = {
            "count": count,
            "areas": [float(cv2.contourArea(c)) for c in selected],
            "contours": [c.reshape(-1, 2).tolist() for c in selected],
            "offset": offset.tolist(),
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
            graphics={"contours": overlay},
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


class SaveImage(Block):
    block_type, subtype = "Output", "Images"
    type_id, title, category = "output.image", "Save image", "Output"
    inputs = (Port("image"),)
    defaults = {"path": "output.png", **CODEC_DEFAULTS}
    parameter_choices = CODEC_CHOICES
    parameter_help = CODEC_HELP

    def process(self, inputs):
        path = Path(self.parameters["path"])
        if path.exists():
            raise FileExistsError("Choose a new path; existing images are preserved")
        options = codec_options(path.suffix.lower(), self.parameters)
        ok, encoded = cv2.imencode(path.suffix, opencv_image(inputs["image"]), options)
        if not ok:
            raise ValueError("Image encoding failed")
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
    from .analysis_tools import ANALYSIS_BLOCKS
    from .annotation import AnnotationLayer
    from .annoy_blocks import ANNOY_BLOCKS
    from .aruco_markers import DetectArucoMarkers
    from .basic_processing import BASIC_PROCESSING_BLOCKS
    from .blob_detection import SimpleBlobDetection
    from .calibration import (
        CalibrationGrid,
        ColorBalance,
        LensCalibration,
        LoadCalibration,
        PixelToPhysical,
        SaveCalibration,
        Undistort,
    )
    from .calibration_validation import ValidateLensCalibration
    from .calipers import BeadCalipers
    from .channel_transform import ChannelTransform, RouteChannels
    from .charuco import DetectCharucoBoard
    from .chessboard_quality import CheckChessboardQuality
    from .cloud_io import CLOUD_IO_BLOCKS
    from .cloud_tools import CLOUD_BLOCKS
    from .clustering import KMeansImage, KMeansSamples
    from .code_readers import CODE_READER_BLOCKS
    from .collection_tools import COLLECTION_BLOCKS
    from .color_calibration import COLOR_CALIBRATION_BLOCKS
    from .color_measurement import COLOR_MEASUREMENT_BLOCKS
    from .contour_analysis import CONTOUR_ANALYSIS_BLOCKS
    from .control_flow import CONTROL_BLOCKS
    from .convex_overlap import ConvexPolygonOverlap
    from .conveyor_tracking import ConveyorDemo, TrackConveyorParts
    from .coordination import CoordinateInputs, CounterInput
    from .depth import RGBD, DepthFile, DepthSample, FillDepthHoles
    from .dnn_blocks import DNN_BLOCKS
    from .ecc_registration import AlignECC
    from .edge_agreement import CompareEdgeMaps
    from .epipolar import CheckEpipolarCorrespondences
    from .example_plugins import invert as invert_example
    from .examples3d import StereoDemo
    from .exposure import AlignExposures, FuseExposures
    from .fanuc import FANUC_BLOCKS
    from .feature_covariance import FitFeatureCovariance, MahalanobisDistance
    from .features import FEATURE_BLOCKS
    from .focus import MeasureFocus
    from .fourier_correlation import FourierCorrelation
    from .generalized_hough import FindGuilTemplate, FindHoughTemplate
    from .geometry3d import GEOMETRY3D_BLOCKS
    from .gram_matrix import GramMatrix
    from .gray_level_cooccurrence import GrayLevelCooccurrence
    from .hand_eye import HandEyeCalibration
    from .hdr import (
        CalibrateCameraResponse,
        CalibrateRobertson,
        MergeHDR,
        MergeRobertson,
        ToneMapHDR,
    )
    from .histogram import Histogram
    from .histogram_comparison import CompareHistograms, HistogramBackProjection
    from .hog import HOGFeatures
    from .illumination import AdaptiveBackgroundCorrection, FlatFieldCorrection
    from .image_blend import BlendWeightMaps
    from .image_codecs import DecodeImage, EncodeImage
    from .image_colorize import ColorizeScalar
    from .image_hash import CompareImageHashes, ComputeImageHash
    from .image_layout import JoinImages, PasteImagePatch, TileImage
    from .image_noise import AddImageNoise, RandomImage
    from .image_norm import MeasureImageNorm
    from .image_quality import CompareImageQuality, MeasureBRISQUE
    from .image_queue import ImageQueue
    from .image_statistics import ImageSummaryStatistics
    from .image_stitching import StitchImagePair
    from .image_validity import CheckImageValidity, ReplaceNonfiniteSamples
    from .judgments import FinalJudgment
    from .label_measurement import MaskFromLabelCollection, MeasureLabelRegions
    from .lens_map_files import LoadLensMaps, SaveLensMaps
    from .lightglue_blocks import LIGHTGLUE_BLOCKS
    from .linear_program import SolveLinearProgram
    from .linear_system import SolveLinearSystem
    from .local_binary_patterns import LocalBinaryPatterns
    from .local_threshold import LocalThreshold
    from .logix import LOGIX_BLOCKS
    from .mask_clearance import MaskClearance
    from .match_review import ReviewFeatureMatches
    from .matrix_eigen import MatrixEigen
    from .matrix_inverse import InvertMatrix
    from .matrix_multiply import MultiplyMatrices
    from .matrix_structure import (
        CompleteSymmetricMatrix,
        CreateDiagonalMatrix,
        MeasureMatrixDiagonal,
    )
    from .matrix_svd import MatrixSVD
    from .mqtt import MQTT_BLOCKS
    from .multiscale_ridges import MultiscaleRidges
    from .native_background import NATIVE_BACKGROUND_BLOCKS
    from .native_trackers import NATIVE_TRACKER_BLOCKS
    from .nearest_rows import FindNearestRows
    from .ocr import OCR_BLOCKS
    from .offline_stabilization import StabilizeRecordedSequence
    from .oriented_edges import FindOrientedEdges
    from .overlay_graphics import OverlayResultGraphics, SetFilmstripImage
    from .parallel_inspection import ParallelInspect, RepeatedPartsDemo
    from .path_continuity import InspectPathContinuity
    from .path_strip import UnwrapPathStrip
    from .patterns import MatchShapePattern, PatternDemoImages, TrainShapePattern
    from .photo_filters import PHOTO_FILTER_BLOCKS
    from .pick_selection import SelectPickCandidate
    from .plugins import (
        GEOMETRIC_MATCHER_TYPES,
        PATH_WIDTH_TYPES,
        PIXEL_SEGMENTATION_TYPES,
        load_installed_geometric_matcher,
        load_installed_path_width,
        load_installed_pixel_segmentation,
    )
    from .point_set_lines import HoughPointSetLines
    from .polar_image import PolarImageWarp
    from .polynomial_roots import SolvePolynomialRoots
    from .poses import POSE_BLOCKS
    from .principal_components import FitPrincipalComponents, TransformPrincipalComponents
    from .processing import PROCESSING_BLOCKS
    from .projection import IntensityProfile, PathIntensityProfile
    from .ptcloud3d import PTCLOUD3D_BLOCKS
    from .pyramids import PYRAMID_BLOCKS
    from .raster_hough_lines import HoughPolarLines
    from .recipe_blocks import RecipeField, RecipeSelect
    from .region_proposals import GraphSegments, SelectiveSearchProposals
    from .regions import REGION_BLOCKS
    from .registration import REGISTRATION_BLOCKS
    from .remap_maps import BuildLensMaps, LensMapDisplacement, UndistortLensPoints
    from .restoration import RESTORATION_BLOCKS
    from .ridge_response import RidgeResponse
    from .scan_stitching import StitchScanSequence
    from .segmentation import SEGMENTATION_BLOCKS
    from .sequence_tools import SEQUENCE_BLOCKS
    from .shape_fitting import (
        BoundingConvexPolygon,
        BoundingRotatedRectangle,
        FitShape,
        RotatedRectangleOverlap,
    )
    from .simple_classifier.dataflow import ClassifyImage
    from .skeleton import PruneSkeleton, Skeletonize
    from .spanning_tree import MinimumSpanningTree
    from .spectral import SPECTRAL_BLOCKS
    from .stabilization import StabilizeCameraMotion
    from .stereo3d import STEREO3D_BLOCKS
    from .strip_defects import GroupStripDefects
    from .strip_graphics import MapStripGraphics
    from .strip_width import InspectStripWidth
    from .subpixel_crop import CropSubpixel
    from .superpixels import SuperpixelsLSC, SuperpixelsSEEDS, SuperpixelsSLIC
    from .table_sort import SortNumericTable
    from .template_instances import FindTemplateInstances
    from .text_rules import VerifyText
    from .texture_patterns import MatchTexturePattern, TrainTexturePattern
    from .tiff_files import TiffFile, WriteTiffFile
    from .tilted_statistics import TiltedWindowStatistics
    from .translation_registration import EstimateTranslation
    from .triangulation import DelaunayTriangulation
    from .video_files import VideoFile, WriteVideo
    from .vision3d_modules import VISION3D_MODULE_BLOCKS
    from .voronoi import VoronoiCells
    from .watershed_seeds import WatershedMaskSeeds, WatershedPointSeeds
    from .yuv_formats import DecodePackedYuv, RawYuvFile
    from .zxing_reader import ReadZXingBarcode

    registry = Registry()
    for block in (
        Comment,
        DemoImage,
        RandomImage,
        *LOGIX_BLOCKS,
        *FANUC_BLOCKS,
        *MQTT_BLOCKS,
        ImageFile,
        VideoFile,
        TiffFile,
        RawYuvFile,
        Camera,
        Grayscale,
        Blur,
        Canny,
        *BASIC_PROCESSING_BLOCKS,
        DecodePackedYuv,
        EncodeImage,
        DecodeImage,
        AnnotationLayer,
        OverlayResultGraphics,
        SetFilmstripImage,
        Threshold,
        Contours,
        SaveImage,
        WriteTiffFile,
        WriteVideo,
        WriteResults,
        Histogram,
        HOGFeatures,
        CalibrationGrid,
        DetectArucoMarkers,
        DetectCharucoBoard,
        LensCalibration,
        ValidateLensCalibration,
        CheckChessboardQuality,
        HandEyeCalibration,
        BuildLensMaps,
        LensMapDisplacement,
        UndistortLensPoints,
        LoadLensMaps,
        SaveLensMaps,
        Undistort,
        ColorBalance,
        *COLOR_CALIBRATION_BLOCKS,
        PixelToPhysical,
        DepthFile,
        RGBD,
        DepthSample,
        FillDepthHoles,
        LoadCalibration,
        SaveCalibration,
        *PROCESSING_BLOCKS,
        LocalThreshold,
        LocalBinaryPatterns,
        GrayLevelCooccurrence,
        JoinImages,
        StitchImagePair,
        StitchScanSequence,
        PasteImagePatch,
        CropSubpixel,
        PolarImageWarp,
        UnwrapPathStrip,
        MapStripGraphics,
        TileImage,
        *FEATURE_BLOCKS,
        *REGISTRATION_BLOCKS,
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
        ToneMapHDR,
        *SPECTRAL_BLOCKS,
        FourierCorrelation,
        *ANALYSIS_BLOCKS,
        BlendWeightMaps,
        AddImageNoise,
        ColorizeScalar,
        FindHoughTemplate,
        FindGuilTemplate,
        ChannelTransform,
        RouteChannels,
        IntensityProfile,
        PathIntensityProfile,
        InspectPathContinuity,
        ImageSummaryStatistics,
        MeasureImageNorm,
        CheckImageValidity,
        ReplaceNonfiniteSamples,
        TiltedWindowStatistics,
        EstimateTranslation,
        *REGION_BLOCKS,
        *PYRAMID_BLOCKS,
        *COLLECTION_BLOCKS,
        SimpleBlobDetection,
        HoughPointSetLines,
        HoughPolarLines,
        *CONTOUR_ANALYSIS_BLOCKS,
        CompareEdgeMaps,
        *CODE_READER_BLOCKS,
        ReadZXingBarcode,
        VerifyText,
        *SEGMENTATION_BLOCKS,
        MeasureLabelRegions,
        MaskFromLabelCollection,
        GraphSegments,
        RidgeResponse,
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
        AdaptiveBackgroundCorrection,
        CompareImageQuality,
        MeasureFocus,
        ComputeImageHash,
        CompareImageHashes,
        MeasureBRISQUE,
        CompareHistograms,
        *COLOR_MEASUREMENT_BLOCKS,
        ClassifyImage,
        HistogramBackProjection,
        FitShape,
        BoundingRotatedRectangle,
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
        InspectStripWidth,
        GroupStripDefects,
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
        StabilizeCameraMotion,
        StabilizeRecordedSequence,
        *NATIVE_BACKGROUND_BLOCKS,
        *NATIVE_TRACKER_BLOCKS,
        TrackConveyorParts,
        ConveyorDemo,
        FinalJudgment,
        *DNN_BLOCKS,
        FitFeatureCovariance,
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
        MahalanobisDistance,
        KMeansSamples,
        KMeansImage,
        *OCR_BLOCKS,
        *ANNOY_BLOCKS,
        *LIGHTGLUE_BLOCKS,
        *GEOMETRY3D_BLOCKS,
        CheckEpipolarCorrespondences,
        *STEREO3D_BLOCKS,
        *CLOUD_BLOCKS,
        *CLOUD_IO_BLOCKS,
        *VISION3D_MODULE_BLOCKS,
        *PTCLOUD3D_BLOCKS,
        StereoDemo,
    ):
        registry.register(block)
    registry.add_provider(GEOMETRIC_MATCHER_TYPES, load_installed_geometric_matcher)
    registry.add_provider(PIXEL_SEGMENTATION_TYPES, load_installed_pixel_segmentation)
    registry.add_provider(PATH_WIDTH_TYPES, load_installed_path_width)
    registry.add_provider((invert_example.Invert.type_id,), invert_example.register)
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
