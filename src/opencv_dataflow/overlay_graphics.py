"""Compose selected result annotations in the final image's coordinate frame."""

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import draw_object, layer_transform
from .images import DepthArray, ImageArray, image_format, opencv_image, preview_rgb
from .tickets import ticket_key

GRAPHICS_INPUTS = tuple(f"graphics_{index}" for index in range(1, 9))


class OverlayResultGraphics(Block):
    type_id = "cv.overlay_result_graphics"
    title, category = "Overlay result graphics", "Image processing"
    block_type, subtype = "Image processing", "Compositing"
    inputs = (Port("image"), *(Port(name, "graphics", required=False) for name in GRAPHICS_INPUTS))
    outputs = (Port("image"),)
    defaults = {"excluded_graphics": {}}
    supports_graphics_selector = True
    composes_result_image = True
    tolerates_unavailable_inputs = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exposed_inputs = ["image", *GRAPHICS_INPUTS[:2]]

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        values = self.parameters if parameters is None else parameters
        excluded = values["excluded_graphics"]
        if (
            not isinstance(excluded, dict)
            or not excluded.keys() <= set(GRAPHICS_INPUTS)
            or any(
                not isinstance(names, list) or any(not isinstance(name, str) for name in names)
                for names in excluded.values()
            )
        ):
            raise ValueError("Excluded graphics must list layer names by graphics input")

    def process(self, inputs):
        source = inputs["image"]
        if isinstance(source, DepthArray) or source.dtype == np.int16:
            canvas = cv2.cvtColor(preview_rgb(source), cv2.COLOR_RGB2BGR)
            encoding = "bgr"
        else:
            encoding = image_format(source)
            base = opencv_image(source)
            canvas = np.repeat(base[..., None], 3, axis=2) if base.ndim == 2 else base[..., :3]
            canvas = np.array(canvas, order="C", copy=True)
        target = self.input_payloads.get("image", {})
        ticket = target.get("ticket")
        selected, skipped = 0, 0
        for slot in GRAPHICS_INPUTS:
            if slot not in inputs:
                continue
            excluded = self.parameters["excluded_graphics"].get(slot, [])
            for layer in inputs[slot].layers:
                if layer.name in excluded:
                    continue
                layer_ticket = layer.payload.get("ticket", self.input_payloads[slot].get("ticket"))
                if ticket is not None and layer_ticket is not None:
                    if ticket_key(ticket) != ticket_key(layer_ticket):
                        skipped += 1
                        continue
                transform = layer_transform(layer, target)
                if transform is None:
                    skipped += 1
                    continue
                for graphic in layer.objects:
                    draw_object(canvas, graphic, transform)
                selected += 1
        if encoding in ("rgb", "rgba"):
            canvas = canvas[..., ::-1]
        if encoding in ("rgba", "bgra"):
            canvas = np.concatenate((canvas, np.asarray(source)[..., 3:]), axis=2)
        output_format = encoding if encoding in ("rgb", "rgba", "bgra") else "bgr"
        return Result(
            {"image": ImageArray(canvas, output_format)},
            message=(
                f"Overlaid {selected} layers · skipped {skipped} "
                "unaligned or different-ticket layers"
            ),
        )
