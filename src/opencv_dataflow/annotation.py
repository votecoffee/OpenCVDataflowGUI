"""Author reusable annotations in the coordinate frame of an inspection image."""

from .core import Block, Port, Result
from .graphics_layers import RESULT_GRAPHICS, GraphicDrawing, graphic_circle, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import ValuePacket, require_same_frame

COORDINATE = NumericRange(-500_000, 500_000)
COLOR_BYTE = NumericRange(0, 255, whole=True)


def annotation_point(value, field):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValidationError("Annotation point must be an [x, y] pair", field)
    if any(type(coordinate) not in (int, float) for coordinate in value):
        raise ValidationError("Annotation coordinates must be numbers", field)
    return tuple(
        COORDINATE.check(coordinate, "Annotation coordinate", field) for coordinate in value
    )


def annotation_text(value, field):
    if not isinstance(value, str) or not value.strip() or len(value) > 160:
        raise ValidationError("Annotation text must contain 1 to 160 characters", field)
    return value


class AnnotationLayer(Block):
    type_id, title, category = "cv.annotation_layer", "Add annotation layer", "Image processing"
    block_type, subtype = "Image processing", "Drawing"
    inputs = (
        Port("image"),
        Port("points", "data", required=False, required_when_connected=True),
        Port("text", "text", required=False, required_when_connected=True),
    )
    outputs = (Port(RESULT_GRAPHICS, "graphics", required=False),)
    defaults = {
        "shape": "rectangle",
        "layer_name": "annotation",
        "origin": [8.0, 8.0],
        "end": [64.0, 42.0],
        "radius": 12.0,
        "label": "Inspection",
        "color_rgb": [255, 197, 66],
        "thickness": 2,
        "filled": False,
    }
    parameter_choices = {"shape": ("line", "rectangle", "circle", "label")}
    parameter_ranges = {
        "radius": NumericRange(0.5, 100_000),
        "thickness": NumericRange(1, 32, whole=True),
    }
    parameter_help = {
        "origin": "Start, corner, circle center, or label anchor in source pixels.",
        "end": "Second endpoint for a line, or opposite corner for a rectangle.",
        "radius": "Circle radius in source pixels.",
        "label": "Text displayed by the label shape. A wired text input takes precedence.",
        "color_rgb": "Three whole RGB values between 0 and 255.",
        "thickness": "Stroke width in rendered pixels for lines and outlines.",
        "filled": "Fill the rectangle or circle.",
    }
    input_help = {
        "image": "Reference for the annotation's pixel frame and ticket.",
        "points": "One [x, y] for a circle or label, or two for a line or rectangle.",
        "text": "Optional label text from an upstream tool.",
    }
    process_inputs_readonly = True
    parallel_safe = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        annotation_point(p["origin"], "origin")
        annotation_point(p["end"], "end")
        annotation_text(p["layer_name"], "layer_name")
        if p["shape"] == "label":
            annotation_text(p["label"], "label")
        color = p["color_rgb"]
        if not isinstance(color, (list, tuple)) or len(color) != 3:
            raise ValidationError("Color must be [red, green, blue]", "color_rgb")
        for value in color:
            COLOR_BYTE.check(value, "Color component", "color_rgb")
        if type(p["filled"]) is not bool:
            raise ValidationError("Filled must be true or false", "filled")

    def process(self, inputs):
        if not self.generate_graphics:
            return Result({})
        p = self.parameters
        shape = p["shape"]
        origin, end = p["origin"], p["end"]
        if "points" in inputs:
            context = self.input_payloads["points"]
            require_same_frame(
                (inputs["image"], ValuePacket(None, context)),
                message="Transform annotation points to the image frame first",
            )
            points = inputs["points"]
            count = 2 if shape in ("line", "rectangle") else 1
            if not isinstance(points, (list, tuple)) or len(points) != count:
                raise ValidationError(f"{shape.capitalize()} requires {count} point(s)", "points")
            origin = annotation_point(points[0], "points")
            if count == 2:
                end = annotation_point(points[1], "points")

        color = tuple(int(value) for value in reversed(p["color_rgb"]))
        thickness = int(p["thickness"])
        if shape == "circle":
            item = graphic_circle(
                *origin, float(p["radius"]), color, thickness=thickness, filled=p["filled"]
            )
        elif shape == "rectangle":
            x, y = origin
            right, bottom = end
            kind = "filled_polygon" if p["filled"] else "polyline"
            item = graphic_object(
                kind,
                (origin, (right, y), end, (x, bottom)),
                color,
                closed=True,
                thickness=thickness,
            )
        elif shape == "label":
            label = annotation_text(inputs.get("text", p["label"]), "text")
            item = graphic_object("label", (origin,), color, text=label)
        else:
            item = graphic_object("polyline", (origin, end), color, thickness=thickness)
        return Result(
            {},
            drawings={p["layer_name"]: GraphicDrawing((item,))},
            message=f"{shape.capitalize()} · {p['layer_name']}",
        )
