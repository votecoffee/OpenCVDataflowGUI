"""Gaussian image sets and optional port-payload tools."""

import copy

import cv2
import numpy as np

from .core import Block, Port, Result
from .images import ImageArray, image_format
from .parameters import BORDERS
from .payloads import ImagePyramid, metadata, moved_payload, validate_payload
from .processing import image_value, integer


class BuildPyramid(Block):
    type_id, title, category = "cv.image_pyramid", "Gaussian image pyramid set", "Image processing"
    block_type, subtype = "Image processing", "Pyramids"
    inputs = (Port("image"),)
    outputs = (Port("images", "pyramid"), Port("image"), Port("info", "data"))
    defaults = {"levels": 4, "border": "reflect"}
    parameter_choices = {"border": tuple(k for k in BORDERS if k != "constant")}
    parameter_help = {"levels": "Total images including level 0 (largest), from 1 to 9."}

    def process(self, inputs):
        source = inputs["image"]
        image = image_value(source, (np.uint8, np.uint16, np.int16, np.float32, np.float64))
        count = integer(self.parameters["levels"], "Levels", 1, 9)
        if image.shape[0] * image.shape[1] > 64_000_000:
            raise ValueError("Pyramid base is limited to 64 million pixels")
        largest_size = [image.shape[1], image.shape[0]]
        levels = []
        info = []
        for level in range(count):
            if level:
                if image.shape[:2] == (1, 1):
                    raise ValueError("Requested pyramid has more levels than this image supports")
                image = cv2.pyrDown(image, borderType=BORDERS[self.parameters["border"]])
            matrix = np.diag([2.0**level, 2.0**level, 1.0])
            size = [image.shape[1], image.shape[0]]
            payload = moved_payload(metadata(source), matrix, size)
            payload["pyramid"] = {
                "set_id": self.id,
                "level": level,
                "largest_size": largest_size,
                "to_largest": matrix.tolist(),
                "level_count": count,
            }
            levels.append(ImageArray(image.copy(), image_format(source), payload))
            info.append({"level": level, "size": size, "to_largest": matrix.tolist()})
        images = ImagePyramid(tuple(levels))
        return Result(
            {"images": images, "image": images.largest, "info": info},
            message=f"{count} levels, largest {largest_size}",
        )


class SelectPyramidLevel(Block):
    type_id, title, category = "cv.pyramid_level", "Select pyramid level", "Image processing"
    block_type, subtype = "Image processing", "Pyramids"
    inputs = (Port("images", "pyramid"),)
    outputs = (Port("image"), Port("transform", "data"))
    defaults = {"level": "largest"}
    parameter_choices = {"level": ("largest", *map(str, range(9)))}

    def process(self, inputs):
        levels = inputs["images"].levels
        index = 0 if self.parameters["level"] == "largest" else int(self.parameters["level"])
        if index >= len(levels):
            raise ValueError(f"Pyramid has no level {index}")
        image = levels[index]
        payload = metadata(image)
        transform = {
            "source_to_target": payload["pyramid"]["to_largest"],
            "source_size": [image.shape[1], image.shape[0]],
            "target_size": payload["pyramid"]["largest_size"],
        }
        return Result(
            {"image": image, "transform": transform},
            payloads={"image": payload, "transform": payload},
            message=f"Selected pyramid level {index}",
        )


class AttachPayload(Block):
    type_id, title, category = "core.attach_payload", "Attach port payload", "Data"
    block_type, subtype = "Data", "Payloads"
    inputs = (Port("value", "any"), Port("payload", "data", required=False))
    outputs = (Port("value", "any"),)
    defaults = {"payload": {}}

    def process(self, inputs):
        payload = copy.deepcopy(self.input_payloads.get("value", {}))
        extra = inputs.get("payload", self.parameters["payload"])
        validate_payload(extra)
        payload.update(copy.deepcopy(extra))
        return Result({"value": inputs["value"]}, payloads={"value": payload})


class ReadPayload(Block):
    type_id, title, category = "core.read_payload", "Read port payload / coordinates", "Data"
    block_type, subtype = "Data", "Payloads"
    inputs = (Port("value", "any"),)
    outputs = (Port("value", "any"), Port("payload", "data"), Port("transform", "data"))
    defaults = {"reference": "largest"}
    parameter_choices = {"reference": ("largest", "root")}

    def process(self, inputs):
        payload = copy.deepcopy(self.input_payloads.get("value", {}))
        spatial, pyramid = payload.get("spatial"), payload.get("pyramid")
        transform = {}
        if spatial:
            largest = self.parameters["reference"] == "largest" and pyramid is not None
            transform = {
                "source_to_target": pyramid["to_largest"] if largest else spatial["to_root"],
                "source_size": spatial["image_size"],
                "target_size": pyramid["largest_size"] if largest else spatial["root_size"],
            }
        return Result(
            {"value": inputs["value"], "payload": payload, "transform": transform},
            payloads={"value": payload, "payload": {}, "transform": payload},
        )


PYRAMID_BLOCKS = (BuildPyramid, SelectPyramidLevel, AttachPayload, ReadPayload)
