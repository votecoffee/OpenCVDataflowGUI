"""Binary morphology on sorted half-open horizontal runs."""

import numpy as np

from .core import Block, Port, Result
from .parameter_validation import NumericRange, ValidationError
from .parameters import anchor_value
from .payloads import check_spatial
from .processing import Morphology, image_value
from .selection import SELECTION_INPUTS, selection_mask

MODEL = "binary_mask_runs_v1"
MAX_DIMENSION = 1_000_000
MAX_RUNS = 250_000
MAX_WORK = 2_000_000
MAX_DENSE_PIXELS = 64_000_000


def read_runs(value):
    if not isinstance(value, dict) or value.get("model") != MODEL:
        raise ValidationError("Connect an encoded binary mask")
    size, runs = value.get("image_size"), value.get("runs")
    if not isinstance(size, list) or len(size) != 2 or any(
        type(n) is not int or not 1 <= n <= MAX_DIMENSION for n in size
    ):
        raise ValidationError(f"Encoded mask dimensions must be integers from 1 to {MAX_DIMENSION:,}")
    if not isinstance(runs, list) or len(runs) > MAX_RUNS:
        raise ValidationError(f"Encoded masks support at most {MAX_RUNS:,} runs")
    width, height = size
    rows = {}
    previous_y, previous_end = -1, -1
    for item in runs:
        if not isinstance(item, list) or len(item) != 3 or any(type(n) is not int for n in item):
            raise ValidationError("Each run must be [row, start, exclusive end] in whole pixels")
        y, start, end = item
        if not 0 <= y < height or not 0 <= start < end <= width:
            raise ValidationError("Run coordinates must be inside the encoded mask")
        if y < previous_y or (y == previous_y and start <= previous_end):
            raise ValidationError("Runs must be sorted and merged within each row")
        rows.setdefault(y, []).append((start, end))
        previous_y, previous_end = y, end
    return size, rows


def mask_result(size, rows, **info):
    runs = [[y, start, end] for y in sorted(rows) for start, end in rows[y]]
    foreground = sum(end - start for _, start, end in runs)
    return Result(
        {
            "runs": {"model": MODEL, "image_size": size, "runs": runs},
            "report": {
                **info,
                "run_count": len(runs),
                "foreground_pixels": foreground,
                "packed_run_bytes": 12 * len(runs),
                "dense_uint8_bytes": size[0] * size[1],
            },
        },
        message=f"{len(runs):,} runs, {foreground:,} foreground pixels",
    )


def encode_rows(image, threshold, invert, selected):
    rows = {}
    count = 0
    for y, row in enumerate(image):
        foreground = (row <= threshold) if invert else (row > threshold)
        if selected is not None:
            foreground &= selected[y] != 0
        edges = np.flatnonzero(np.diff(foreground, prepend=False, append=False)).tolist()
        if edges:
            count += len(edges) // 2
            if count > MAX_RUNS:
                raise ValidationError(f"Encoded mask exceeds {MAX_RUNS:,} runs. Use dense morphology.")
            rows[y] = list(zip(edges[::2], edges[1::2], strict=True))
    return rows


def kernel_spans(parameters):
    try:
        kernel = Morphology.kernel_matrix(parameters)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            str(exc), "shape", "kernel", "kernel_y", "shape_anchor", "custom_kernel"
        ) from exc
    height, width = kernel.shape
    try:
        ax, ay = anchor_value(parameters["anchor"], width, height)
    except ValueError as exc:
        raise ValidationError(str(exc), "anchor") from exc
    ax = width // 2 if ax == -1 else ax
    ay = height // 2 if ay == -1 else ay
    spans = []
    for y, row in enumerate(kernel):
        active = np.flatnonzero(row)
        if active.size:
            left, right = int(active[0]), int(active[-1])
            if active.size != right - left + 1:
                raise ValidationError("Each active kernel row must be contiguous", "custom_kernel")
            spans.append((y - ay, left - ax, right - ax))
    return spans


def merged(intervals):
    result = []
    for start, end in intervals:
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def intersect(a, b):
    i = j = 0
    while i < len(a) and j < len(b):
        left, right = a[i], b[j]
        start, end = max(left[0], right[0]), min(left[1], right[1])
        if start < end:
            yield start, end
        if left[1] <= right[1]:
            i += 1
        else:
            j += 1


def subtract(a, b):
    j = 0
    for start, end in a:
        while j < len(b) and b[j][0] < end and start < end:
            lo, hi = b[j]
            if start < lo:
                yield start, lo
            if hi >= end:
                start = end
            else:
                start = max(start, hi)
                j += 1
        if start < end:
            yield start, end


class RunMorphology:
    def __init__(self, size, spans, foreground_border):
        self.width, self.height = size
        self.spans = spans
        self.foreground_border = foreground_border
        self.work = 0

    def spend(self, count):
        self.work += count
        if self.work > MAX_WORK:
            raise ValidationError("Sparse morphology work limit exceeded. Use dense morphology or smaller tiles.")

    def dilate(self, rows):
        output = {}
        self.spend(sum(map(len, rows.values())) * len(self.spans))
        for y, intervals in rows.items():
            for dy, left, right in self.spans:
                if not 0 <= y - dy < self.height:
                    continue
                shifted = output.setdefault(y - dy, [])
                for start, end in intervals:
                    lo, hi = max(0, start - right), min(self.width, end - left)
                    if lo < hi:
                        shifted.append((lo, hi))
        return {y: merged(sorted(row)) for y, row in output.items() if row}

    def erosion_candidates(self, rows, y, span):
        dy, left, right = span
        source_y = y + dy
        if not 0 <= source_y < self.height:
            return [(0, self.width)] if self.foreground_border else []
        intervals = rows.get(source_y, ())
        self.spend(len(intervals) + 1)
        if self.foreground_border:
            # Add only the exterior pixels a kernel sample can reach, merging at the edges.
            reach = max(abs(left), abs(right)) + 1
            intervals = merged([(-reach, 0), *intervals, (self.width, self.width + reach)])
        allowed = []
        for start, end in intervals:
            lo, hi = max(0, start - left), min(self.width, end - right)
            if lo < hi:
                allowed.append((lo, hi))
        return allowed

    def erode(self, rows):
        # A span covering the anchor column cannot find horizontal foreground in empty rows.
        # Start there when possible to avoid visiting the full height of a sparse image.
        seed = min(self.spans, key=lambda s: (not s[1] <= 0 <= s[2], abs(s[0])))
        dy, left, right = seed
        if self.foreground_border and not left <= 0 <= right:
            candidates = range(self.height)
        else:
            candidates = [y - dy for y in rows if 0 <= y - dy < self.height]
            if self.foreground_border:
                if dy > 0:
                    candidates.extend(range(max(0, self.height - dy), self.height))
                elif dy < 0:
                    candidates.extend(range(min(-dy, self.height)))
        self.spend(len(candidates))
        output = {}
        run_count = 0
        for y in candidates:
            current = self.erosion_candidates(rows, y, seed)
            for span in self.spans:
                if not current:
                    break
                if span != seed:
                    self.spend(len(current))
                    current = list(intersect(current, self.erosion_candidates(rows, y, span)))
            if current:
                run_count += len(current)
                if run_count > MAX_RUNS:
                    raise ValidationError(f"Morphology exceeds {MAX_RUNS:,} runs. Use smaller tiles.")
                output[y] = current
        return output

    def repeat(self, primitive, rows, iterations):
        for _ in range(iterations):
            rows = primitive(rows)
            if sum(map(len, rows.values())) > MAX_RUNS:
                raise ValidationError(f"Morphology exceeds {MAX_RUNS:,} runs. Use smaller tiles.")
        return rows

    def apply(self, rows, operation, iterations):
        if operation == "erode":
            return self.repeat(self.erode, rows, iterations)
        if operation == "dilate":
            return self.repeat(self.dilate, rows, iterations)
        if operation in ("open", "tophat"):
            eroded = self.repeat(self.erode, rows, iterations)
            opened = self.repeat(self.dilate, eroded, iterations)
            if operation == "open":
                return opened
            positive, negative = rows, opened
        elif operation in ("close", "blackhat"):
            dilated = self.repeat(self.dilate, rows, iterations)
            closed = self.repeat(self.erode, dilated, iterations)
            if operation == "close":
                return closed
            positive, negative = closed, rows
        else:
            positive = self.repeat(self.dilate, rows, iterations)
            negative = self.repeat(self.erode, rows, iterations)
        self.spend(sum(map(len, positive.values())) + sum(map(len, negative.values())))
        output = {y: list(subtract(row, negative.get(y, ()))) for y, row in positive.items()}
        if sum(map(len, output.values())) > MAX_RUNS:
            raise ValidationError(f"Morphology exceeds {MAX_RUNS:,} runs. Use smaller tiles.")
        return {y: row for y, row in output.items() if row}


class EncodeBinaryMaskRuns(Block):
    type_id, title = "mask.encode_runs", "Encode binary mask runs"
    category, block_type, subtype = "Image processing", "Image processing", "Morphology"
    inputs = SELECTION_INPUTS
    outputs = (Port("runs", "data"), Port("report", "data"))
    payload_reference = "image"
    parallel_safe = preview_safe = process_inputs_readonly = True
    defaults = {"threshold": 0.0, "polarity": "bright"}
    parameter_ranges = {"threshold": NumericRange()}
    parameter_choices = {"polarity": ("bright", "dark")}
    parameter_help = {"threshold": "Bright selects samples above this value. Dark includes equality."}

    def process(self, inputs):
        image = inputs["image"]
        if image.ndim != 2:
            raise ValidationError("Encode a monochrome image or binary mask")
        height, width = image.shape
        if width > MAX_DIMENSION or height > MAX_DIMENSION or image.size > MAX_DENSE_PIXELS:
            raise ValidationError("Image exceeds the dense mask encoding limit")
        image = image_value(image)
        selected = selection_mask(inputs, image)
        rows = encode_rows(
            image, float(self.parameters["threshold"]), self.parameters["polarity"] == "dark", selected
        )
        return mask_result([width, height], rows)


class MorphologyMaskRuns(Block):
    type_id, title = "mask.morphology_runs", "Run-length binary morphology"
    category, block_type, subtype = "Image processing", "Image processing", "Morphology"
    inputs = (Port("runs", "data"),)
    outputs = (Port("runs", "data"), Port("report", "data"))
    payload_reference = "runs"
    parallel_safe = preview_safe = process_inputs_readonly = True
    supports_kernel_editor = True
    defaults = {
        key: value for key, value in Morphology.defaults.items() if key not in ("border", "border_value")
    } | {"erosion_border": "foreground"}
    parameter_choices = {
        "operation": Morphology.parameter_choices["operation"],
        "shape": Morphology.parameter_choices["shape"],
        "erosion_border": ("foreground", "background"),
    }
    parameter_ranges = {
        "kernel": NumericRange(1, 255, whole=True),
        "kernel_y": NumericRange(0, 255, whole=True),
        "iterations": Morphology.parameter_ranges["iterations"],
    }
    parameter_help = {
        **{key: Morphology.parameter_help[key] for key in ("shape_anchor", "custom_kernel")},
        "erosion_border": "Foreground matches the neutral OpenCV border. Dilation always uses background.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        kernel_spans(self.parameters if parameters is None else parameters)

    def process(self, inputs):
        size, rows = read_runs(inputs["runs"])
        check_spatial(self.input_payloads["runs"], size)
        p = self.parameters
        morphology = RunMorphology(size, kernel_spans(p), p["erosion_border"] == "foreground")
        output = morphology.apply(rows, p["operation"], int(p["iterations"]))
        return mask_result(size, output, operation=p["operation"], work_items=morphology.work)


class DecodeBinaryMaskRuns(Block):
    type_id, title = "mask.decode_runs", "Decode binary mask runs"
    category, block_type, subtype = "Image processing", "Image processing", "Morphology"
    inputs = (Port("runs", "data"),)
    outputs = (Port("image"),)
    payload_reference = "runs"
    parallel_safe = preview_safe = process_inputs_readonly = True
    defaults = {"foreground": 255, "max_pixels": MAX_DENSE_PIXELS}
    parameter_ranges = {
        "foreground": NumericRange(1, 255, whole=True),
        "max_pixels": NumericRange(1, MAX_DENSE_PIXELS, whole=True),
    }
    parameter_help = {"max_pixels": "Maximum number of pixels to allocate for the dense uint8 output."}

    def process(self, inputs):
        size, rows = read_runs(inputs["runs"])
        check_spatial(self.input_payloads["runs"], size)
        width, height = size
        if width * height > self.parameters["max_pixels"]:
            raise ValidationError("Encoded mask exceeds the configured dense decode limit", "max_pixels")
        image = np.zeros((height, width), np.uint8)
        for y, intervals in rows.items():
            for start, end in intervals:
                image[y, start:end] = self.parameters["foreground"]
        return Result({"image": image})
