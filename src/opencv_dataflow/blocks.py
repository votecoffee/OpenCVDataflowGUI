"""Built-in acquisition, G-API processing, inspection, and output blocks."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np

from .core import Block, Graph, Port, Registry, Result
from .images import COLOR_FORMATS, PIXEL_DEPTHS, format_image, image_format, opencv_image


class DemoImage(Block):
    type_id, title, category = "input.demo", "Demo image", "Input"
    outputs = (Port("image"),)

    def process(self, inputs):
        image = np.full((360, 480, 3), 28, np.uint8)
        cv2.circle(image, (140, 180), 72, (210, 190, 70), -1)
        cv2.rectangle(image, (270, 105), (395, 250), (90, 205, 160), -1)
        cv2.putText(image, "DATAFLOW", (26, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
        return Result({"image": image}, message="Synthetic inspection target")


class ImageFile(Block):
    type_id, title, category = "input.file", "Image file", "Input"
    outputs = (Port("image"), Port("info", "data"))
    defaults = {"path": "", "color_format": "bgr", "pixel_depth": "uint8"}
    parameter_choices = {"color_format": COLOR_FORMATS, "pixel_depth": PIXEL_DEPTHS}

    def process(self, inputs):
        path = Path(self.parameters["path"])
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Unable to decode image: {path}")
        output = format_image(
            image, self.parameters["color_format"], self.parameters["pixel_depth"]
        )
        info = {
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


class Camera(Block):
    type_id, title, category = "input.camera", "Camera", "Input"
    outputs = (Port("image"),)
    defaults = {"device": 0}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._capture = None
        self._device = None

    def process(self, inputs):
        device = int(self.parameters["device"])
        if self._capture is None or device != self._device:
            self.close()
            self._capture = cv2.VideoCapture(device)
            self._device = device
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
            self._computation = cv2.GComputation(cv2.GIn(image), cv2.GOut(output))
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


class Blur(GAPIBlock):
    type_id, title = "gapi.blur", "Gaussian blur"
    defaults = {"kernel": 5, "sigma": 1.2}

    def build(self, image):
        kernel = int(self.parameters["kernel"])
        if kernel < 1 or kernel % 2 != 1:
            raise ValueError("Kernel must be a positive odd integer")
        sigma = float(self.parameters["sigma"])
        if sigma < 0:
            raise ValueError("Sigma must be nonnegative")
        return cv2.gapi.gaussianBlur(image, (kernel, kernel), sigma)


class Canny(GAPIBlock):
    type_id, title = "gapi.canny", "Canny edges"
    input_depths = (np.uint8,)
    defaults = {"low": 60.0, "high": 150.0}

    def build(self, image):
        low, high = float(self.parameters["low"]), float(self.parameters["high"])
        if not 0 <= low <= high:
            raise ValueError("Thresholds must satisfy 0 <= low <= high")
        return cv2.gapi.Canny(image, low, high)


class Threshold(Block):
    type_id, title, category = "cv.threshold", "Threshold", "Processing"
    inputs, outputs = (Port("image"),), (Port("image"), Port("threshold", "number"))
    defaults = {"value": 127, "invert": False}

    def process(self, inputs):
        image = opencv_image(inputs["image"])
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        value = int(self.parameters["value"])
        if not 0 <= value <= 255:
            raise ValueError("Threshold must be between 0 and 255")
        mode = cv2.THRESH_BINARY_INV if self.parameters["invert"] else cv2.THRESH_BINARY
        level, mask = cv2.threshold(image, value, 255, mode)
        return Result({"image": mask, "threshold": level})


class Contours(Block):
    type_id, title, category = "inspect.contours", "Find contours", "Inspection"
    inputs = (Port("image"),)
    outputs = (Port("overlay"), Port("mask"), Port("count", "number"), Port("data", "data"))
    defaults = {"min_area": 100.0, "expected_count": 2}

    def process(self, inputs):
        image = opencv_image(inputs["image"])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        minimum = float(self.parameters["min_area"])
        expected = int(self.parameters["expected_count"])
        if minimum < 0 or expected < 0:
            raise ValueError("Minimum area and expected count must be nonnegative")
        selected = [c for c in contours if cv2.contourArea(c) >= minimum]
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        mask = np.zeros_like(gray)
        cv2.drawContours(overlay, selected, -1, (80, 230, 140), 2)
        cv2.drawContours(mask, selected, -1, 255, -1)
        count = len(selected)
        score = max(0.0, 1.0 - abs(count - expected) / max(expected, 1))
        data = {"count": count, "areas": [float(cv2.contourArea(c)) for c in selected]}
        return Result(
            {"overlay": overlay, "mask": mask, "count": count, "data": data},
            graphics={"contours": overlay},
            passed=count == expected,
            score=score,
            message=f"{count} contours / expected {expected}",
        )


class SaveImage(Block):
    type_id, title, category = "output.image", "Save image", "Output"
    inputs = (Port("image"),)
    defaults = {"path": "output.png"}

    def process(self, inputs):
        path = Path(self.parameters["path"])
        if path.exists():
            raise FileExistsError("Choose a new path; existing images are preserved")
        ok, encoded = cv2.imencode(path.suffix, opencv_image(inputs["image"]))
        if not ok:
            raise ValueError("Image encoding failed")
        with path.open("xb") as stream:
            stream.write(encoded.tobytes())
        return Result(message=f"Saved {path.name}")


class WriteResults(Block):
    type_id, title, category = "output.results", "Write results", "Output"
    inputs = (Port("data", "data"),)
    defaults = {"path": "results.csv", "format": "csv"}

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
    registry = Registry()
    for block in (
        DemoImage,
        ImageFile,
        Camera,
        Grayscale,
        Blur,
        Canny,
        Threshold,
        Contours,
        SaveImage,
        WriteResults,
    ):
        registry.register(block)
    return registry


def demo_graph(registry):
    graph = Graph()
    chain = [
        graph.add(registry.create(t))
        for t in ("input.demo", "gapi.gray", "gapi.blur", "cv.threshold", "inspect.contours")
    ]
    chain[3].parameters["value"] = 100
    chain[4].parameters["min_area"] = 1000.0
    for index, block in enumerate(chain):
        block.position = [index * 265.0, (index % 2) * 45.0]
    for source, target in zip(chain, chain[1:]):
        graph.connect(source.id, "image", target.id, "image")
    return graph
