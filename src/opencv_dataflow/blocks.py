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
from .images import COLOR_FORMATS, PIXEL_DEPTHS, format_image, image_format, opencv_image
from .objects import collection, contour_item
from .parameters import BORDERS, scalar_value
from .payloads import NO_VALUE, moved_payload
from .selection import SELECTION_INPUTS, RegionalOperation, contains, selection_mask


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


class ImageFile(Block):
    block_type, subtype = "Input", "Files"
    type_id, title, category = "input.file", "Image file", "Input"
    outputs = (Port("image"), Port("info", "data"))
    file_source = True
    defaults = {
        "path": "",
        "source_mode": "single",
        "paths": [],
        "directory": "",
        "recursive": False,
        "repeat": "stop",
        "filename_ticket": True,
        "color_format": "bgr",
        "pixel_depth": "uint8",
        "path_base": "working_directory",
    }
    parameter_choices = {
        "source_mode": ("single", "files", "directory"),
        "repeat": ("stop", "loop"),
        "color_format": COLOR_FORMATS,
        "pixel_depth": PIXEL_DEPTHS,
        "path_base": ("working_directory", "project"),
    }
    parameter_help = {
        "paths": "Ordered image paths; one image per activation in files mode.",
        "directory": "Snapshot supported image filenames in sorted order.",
        "repeat": "Stop at end, or loop. Reset flow state to restart.",
        "filename_ticket": "Use filename tickets unless an explicit ticket mode is selected.",
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
    def finite_source(self):
        return self.parameters["source_mode"] != "single" and self.parameters["repeat"] == "stop"

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
        image = cv2.imdecode(np.frombuffer(self.read_bytes(path), np.uint8), cv2.IMREAD_UNCHANGED)
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._capture = None
        self._device = None

    def process(self, inputs):
        device = int(self.parameters["device"])
        from .processing import finite, integer

        p = self.parameters
        width, height, buffer_size = (
            integer(p[k], k, 0, 16384) for k in ("width", "height", "buffer_size")
        )
        fps = finite(p["fps"], "FPS", 0)
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
    inputs = SELECTION_INPUTS
    region_background = "zero"
    block_type, subtype = "Image processing", "Segmentation"
    type_id, title, category = "cv.threshold", "Threshold", "Processing"
    inputs, outputs = SELECTION_INPUTS, (Port("image"), Port("threshold", "number"))
    defaults = {"value": 127.0, "invert": False, "max_value": 255.0, "mode": "binary"}
    parameter_choices = {"mode": ("binary", "binary_inv", "trunc", "tozero", "tozero_inv")}

    def process(self, inputs):
        image = opencv_image(inputs["image"])
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        value, maximum = float(self.parameters["value"]), float(self.parameters["max_value"])
        if not np.isfinite([value, maximum]).all() or maximum < 0:
            raise ValueError("Threshold/max value must be finite and max value nonnegative")
        name = self.parameters["mode"]
        if self.parameters["invert"]:
            if name != "binary":
                raise ValueError(
                    "Legacy invert applies to binary mode; otherwise choose an inverse mode"
                )
            name = "binary_inv"
        mode = {
            "binary": cv2.THRESH_BINARY,
            "binary_inv": cv2.THRESH_BINARY_INV,
            "trunc": cv2.THRESH_TRUNC,
            "tozero": cv2.THRESH_TOZERO,
            "tozero_inv": cv2.THRESH_TOZERO_INV,
        }[name]
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
        "approximation": ("none", "simple", "tc89_l1", "tc89_kcos"),
        "region_mode": ("center", "clip"),
    }

    def process(self, inputs):
        image = opencv_image(inputs["image"])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        retrieval = {
            "external": cv2.RETR_EXTERNAL,
            "list": cv2.RETR_LIST,
            "ccomp": cv2.RETR_CCOMP,
            "tree": cv2.RETR_TREE,
        }
        approximation = {
            "none": cv2.CHAIN_APPROX_NONE,
            "simple": cv2.CHAIN_APPROX_SIMPLE,
            "tc89_l1": cv2.CHAIN_APPROX_TC89_L1,
            "tc89_kcos": cv2.CHAIN_APPROX_TC89_KCOS,
        }
        offset = np.asarray(self.parameters["offset"], float)
        if (
            offset.shape != (2,)
            or not np.isfinite(offset).all()
            or np.any(offset != np.rint(offset))
        ):
            raise ValueError("Contour offset must be a whole-pixel [x,y] pair")
        selection = selection_mask(inputs, gray)
        search = gray
        if selection is not None and self.parameters["region_mode"] == "clip":
            search = np.where(selection != 0, gray, 0).astype(gray.dtype)
        contours, hierarchy = cv2.findContours(
            search,
            retrieval[self.parameters["retrieval"]],
            approximation[self.parameters["approximation"]],
            offset=tuple(offset.astype(int)),
        )
        minimum = float(self.parameters["min_area"])
        expected = int(self.parameters["expected_count"])
        if minimum < 0 or expected < 0:
            raise ValueError("Minimum area and expected count must be nonnegative")
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
            passed=count == expected,
            score=score,
            message=f"{count} contours / expected {expected}",
        )


class SaveImage(Block):
    block_type, subtype = "Output", "Images"
    type_id, title, category = "output.image", "Save image", "Output"
    inputs = (Port("image"),)
    defaults = {
        "path": "output.png",
        "jpeg_quality": -1,
        "jpeg_progressive": False,
        "jpeg_optimize": False,
        "jpeg_restart_interval": 0,
        "jpeg_luma_quality": -1,
        "jpeg_chroma_quality": -1,
        "png_compression": -1,
        "png_strategy": "native",
        "png_bilevel": False,
        "webp_quality": -1,
    }
    parameter_choices = {
        "png_strategy": ("native", "default", "filtered", "huffman", "rle", "fixed")
    }
    parameter_help = {
        "jpeg_quality": "-1 preserves the codec default; otherwise 0..100. JPEG only.",
        "png_compression": "-1 preserves the codec default; otherwise 0..9. PNG only.",
        "webp_quality": "-1 preserves the codec default; otherwise 1..100. WebP only.",
    }

    def process(self, inputs):
        path = Path(self.parameters["path"])
        if path.exists():
            raise FileExistsError("Choose a new path; existing images are preserved")
        from .processing import integer

        options = []
        ext = path.suffix.lower()
        if ext in (".jpg", ".jpeg"):
            for key, flag in (
                ("jpeg_quality", cv2.IMWRITE_JPEG_QUALITY),
                ("jpeg_luma_quality", cv2.IMWRITE_JPEG_LUMA_QUALITY),
                ("jpeg_chroma_quality", cv2.IMWRITE_JPEG_CHROMA_QUALITY),
            ):
                value = integer(self.parameters[key], key, -1, 100)
                if value >= 0:
                    options.extend([flag, value])
            options.extend(
                [
                    cv2.IMWRITE_JPEG_PROGRESSIVE,
                    int(self.parameters["jpeg_progressive"]),
                    cv2.IMWRITE_JPEG_OPTIMIZE,
                    int(self.parameters["jpeg_optimize"]),
                    cv2.IMWRITE_JPEG_RST_INTERVAL,
                    integer(self.parameters["jpeg_restart_interval"], "Restart interval", 0, 65535),
                ]
            )
        elif ext == ".png":
            compression = integer(self.parameters["png_compression"], "PNG compression", -1, 9)
            if compression >= 0:
                options.extend([cv2.IMWRITE_PNG_COMPRESSION, compression])
            strategy = self.parameters["png_strategy"]
            if strategy != "native":
                value = {
                    "default": cv2.IMWRITE_PNG_STRATEGY_DEFAULT,
                    "filtered": cv2.IMWRITE_PNG_STRATEGY_FILTERED,
                    "huffman": cv2.IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY,
                    "rle": cv2.IMWRITE_PNG_STRATEGY_RLE,
                    "fixed": cv2.IMWRITE_PNG_STRATEGY_FIXED,
                }[strategy]
                options.extend([cv2.IMWRITE_PNG_STRATEGY, value])
            if self.parameters["png_bilevel"]:
                options.extend([cv2.IMWRITE_PNG_BILEVEL, 1])
        elif ext == ".webp":
            quality = integer(self.parameters["webp_quality"], "WebP quality", -1, 100)
            if quality >= 0:
                if quality == 0:
                    raise ValueError("WebP quality must be 1..100 or -1 for the native default")
                options.extend([cv2.IMWRITE_WEBP_QUALITY, quality])
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
    from .annoy_blocks import ANNOY_BLOCKS
    from .basic_processing import BASIC_PROCESSING_BLOCKS
    from .calibration import (
        CalibrationGrid,
        ColorBalance,
        LensCalibration,
        LoadCalibration,
        PixelToPhysical,
        SaveCalibration,
        Undistort,
    )
    from .cloud_io import CLOUD_IO_BLOCKS
    from .cloud_tools import CLOUD_BLOCKS
    from .code_readers import CODE_READER_BLOCKS
    from .collection_tools import COLLECTION_BLOCKS
    from .contour_analysis import CONTOUR_ANALYSIS_BLOCKS
    from .control_flow import CONTROL_BLOCKS
    from .coordination import CoordinateInputs, CounterInput
    from .depth import RGBD, DepthFile, DepthSample
    from .dnn_blocks import DNN_BLOCKS
    from .ecc_registration import AlignECC
    from .examples3d import StereoDemo
    from .features import FEATURE_BLOCKS
    from .geometry3d import GEOMETRY3D_BLOCKS
    from .histogram import Histogram
    from .histogram_comparison import CompareHistograms, HistogramBackProjection
    from .image_quality import CompareImageQuality
    from .image_queue import ImageQueue
    from .judgments import FinalJudgment
    from .lightglue_blocks import LIGHTGLUE_BLOCKS
    from .oriented_edges import FindOrientedEdges
    from .parallel_inspection import ParallelInspect, RepeatedPartsDemo
    from .patterns import MatchShapePattern, PatternDemoImages, TrainShapePattern
    from .poses import POSE_BLOCKS
    from .processing import PROCESSING_BLOCKS
    from .ptcloud3d import PTCLOUD3D_BLOCKS
    from .pyramids import PYRAMID_BLOCKS
    from .recipe_blocks import RecipeField, RecipeSelect
    from .regions import REGION_BLOCKS
    from .registration import REGISTRATION_BLOCKS
    from .restoration import RESTORATION_BLOCKS
    from .segmentation import SEGMENTATION_BLOCKS
    from .sequence_tools import SEQUENCE_BLOCKS
    from .shape_fitting import FitShape
    from .stereo3d import STEREO3D_BLOCKS
    from .template_instances import FindTemplateInstances
    from .texture_patterns import MatchTexturePattern, TrainTexturePattern
    from .vision3d_modules import VISION3D_MODULE_BLOCKS

    registry = Registry()
    for block in (
        DemoImage,
        ImageFile,
        Camera,
        Grayscale,
        Blur,
        Canny,
        *BASIC_PROCESSING_BLOCKS,
        Threshold,
        Contours,
        SaveImage,
        WriteResults,
        Histogram,
        CalibrationGrid,
        LensCalibration,
        Undistort,
        ColorBalance,
        PixelToPhysical,
        DepthFile,
        RGBD,
        DepthSample,
        LoadCalibration,
        SaveCalibration,
        *PROCESSING_BLOCKS,
        *FEATURE_BLOCKS,
        *REGISTRATION_BLOCKS,
        AlignECC,
        *RESTORATION_BLOCKS,
        *ANALYSIS_BLOCKS,
        *REGION_BLOCKS,
        *PYRAMID_BLOCKS,
        *COLLECTION_BLOCKS,
        *CONTOUR_ANALYSIS_BLOCKS,
        *CODE_READER_BLOCKS,
        *SEGMENTATION_BLOCKS,
        CompareImageQuality,
        CompareHistograms,
        HistogramBackProjection,
        FitShape,
        FindTemplateInstances,
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
        FinalJudgment,
        *DNN_BLOCKS,
        *ANNOY_BLOCKS,
        *LIGHTGLUE_BLOCKS,
        *GEOMETRY3D_BLOCKS,
        *STEREO3D_BLOCKS,
        *CLOUD_BLOCKS,
        *CLOUD_IO_BLOCKS,
        *VISION3D_MODULE_BLOCKS,
        *PTCLOUD3D_BLOCKS,
        StereoDemo,
    ):
        registry.register(block)
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
