"""Native one-window HOG descriptors with cell-orientation graphics."""

import copy

import cv2
import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .images import image_format
from .parameter_validation import NumericRange, ValidationError
from .processing import image_value

_SIZE_FIELDS = ("cell_size", "block_size", "block_stride")
_SIZE_RANGE = NumericRange(2, 2048, whole=True)


def _sizes(parameters):
    sizes = []
    for field in _SIZE_FIELDS:
        value = parameters[field]
        label = field.replace("_", " ").capitalize()
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValidationError(f"{label} must be [width, height] in pixels", field)
        sizes.append(tuple(_SIZE_RANGE.check(n, label, field) for n in value))
    cell, block, stride = sizes
    for field, dimensions in (("block_size", block), ("block_stride", stride)):
        if any(n % c for n, c in zip(dimensions, cell)):
            raise ValidationError("Block dimensions and stride must use whole cells", field)
    if any(s > b for s, b in zip(stride, block)):
        raise ValidationError("Block stride must not exceed the block size", "block_stride")
    if block[0] * block[1] > 65_536:
        raise ValidationError("A HOG block is limited to 65,536 pixels", "block_size")
    return cell, block, stride


def _cell_votes(descriptor, blocks, block_cells, cell_stride, cells, bins):
    # OpenCV stores both blocks and cells column-first, with orientation bins contiguous.
    windows = descriptor.reshape(*blocks, *block_cells, bins)
    votes = np.zeros((*cells, bins), np.float32)
    visits = np.zeros(cells, np.uint16)
    step_x, step_y = cell_stride
    for x in range(block_cells[0]):
        for y in range(block_cells[1]):
            footprint = np.s_[x : x + blocks[0] * step_x : step_x,
                              y : y + blocks[1] * step_y : step_y]
            votes[footprint] += windows[:, :, x, y]
            visits[footprint] += 1
    return votes / visits[..., None]


def _orientation_glyphs(votes, cell, signed):
    magnitude = votes.max(axis=2)
    peak = float(magnitude.max())
    if peak == 0:
        return ()
    bins = votes.shape[2]
    angles = (np.arange(bins) + 0.5) * ((2 if signed else 1) * np.pi / bins)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    strongest = votes.argmax(axis=2)
    half_length = 0.42 * min(cell) * magnitude / peak
    offsets = directions[strongest] * half_length[..., None]
    kind = "arrow" if signed else "polyline"
    objects = []
    for x, y in zip(*np.nonzero(magnitude > 0)):
        center = np.array([(x + 0.5) * cell[0] - 0.5, (y + 0.5) * cell[1] - 0.5])
        offset = offsets[x, y]
        objects.append(graphic_object(
            kind, (center - offset, center + offset), (40, 210, 255), thickness=1,
        ))
    return tuple(objects)


class HOGFeatures(Block):
    type_id, title = "inspect.hog", "Histogram of oriented gradients"
    category, block_type, subtype = "Inspection", "Inspection", "Features"
    inputs = (Port("image"),)
    outputs = (Port("descriptor", "data"), Port("info", "data"))
    defaults = {
        "cell_size": [8, 8], "block_size": [16, 16], "block_stride": [8, 8], "bins": 9,
        "win_sigma": -1.0, "l2_hys_threshold": 0.2,
        "gamma_correction": True, "signed_gradient": False,
    }
    parameter_ranges = {
        "bins": NumericRange(2, 36, whole=True),
        "win_sigma": NumericRange(-1, 4096),
        "l2_hys_threshold": NumericRange(0.0001, 1),
    }
    parameter_help = {
        "cell_size": "[Width, height] of a cell in pixels.",
        "block_size": "[Width, height] in pixels. Both dimensions must be multiples of cell size.",
        "block_stride": "[X, Y] stride in pixels, in whole cells no larger than the block.",
        "bins": "Gradient orientation bins per cell.",
        "win_sigma": "Gaussian weighting sigma in pixels. −1 uses OpenCV's automatic value.",
        "l2_hys_threshold": "Clipping threshold before the second block normalization.",
        "gamma_correction": "Apply square-root gamma correction before measuring gradients.",
        "signed_gradient": "Use 360° directions. Off treats opposite directions as equivalent.",
    }
    input_help = {
        "image": "One uint8 grayscale, BGR or RGB window. Crop first to measure a region.",
    }
    process_inputs_readonly = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        _sizes(p)
        sigma = float(p["win_sigma"])
        if sigma != -1 and sigma < 0.0001:
            raise ValidationError("Sigma must be −1 or at least 0.0001", "win_sigma")
        for key in ("gamma_correction", "signed_gradient"):
            if type(p[key]) is not bool:
                raise ValidationError(f"{key.replace('_', ' ').capitalize()} must be boolean", key)

    def process(self, inputs):
        source, p = inputs["image"], self.parameters
        if image_format(source) not in ("gray", "bgr", "rgb"):
            raise ValidationError("Use a uint8 grayscale, BGR or RGB image")
        height, width = source.shape[:2]
        if width * height > 4_194_304:
            raise ValidationError("HOG accepts windows up to 4,194,304 pixels")
        cell, block, stride = _sizes(p)
        size = (width, height)
        if any(n < b or (n - b) % s for n, b, s in zip(size, block, stride)):
            raise ValidationError("Crop or resize the window so whole HOG blocks fit the stride")
        cells = tuple(n // c for n, c in zip(size, cell))
        blocks = tuple((n - b) // s + 1 for n, b, s in zip(size, block, stride))
        block_cells = tuple(b // c for b, c in zip(block, cell))
        cell_stride = tuple(s // c for s, c in zip(stride, cell))
        bins = int(p["bins"])
        length = blocks[0] * blocks[1] * block_cells[0] * block_cells[1] * bins
        if cells[0] * cells[1] > 4096 or length > 262_144:
            raise ValidationError("HOG is limited to 4,096 cells and 262,144 descriptor values")
        image = image_value(source, (np.uint8,), canonical=True)
        hog = cv2.HOGDescriptor(
            _winSize=size, _blockSize=block, _blockStride=stride, _cellSize=cell, _nbins=bins,
            _winSigma=float(p["win_sigma"]), _L2HysThreshold=float(p["l2_hys_threshold"]),
            _gammaCorrection=p["gamma_correction"], _signedGradient=p["signed_gradient"],
        )
        native = hog.compute(image)
        if native is None or native.size != length or not np.isfinite(native).all():
            raise ValidationError("OpenCV returned an invalid HOG descriptor")
        descriptor = native.reshape(-1)
        drawings = {}
        if self.generate_graphics:
            votes = _cell_votes(descriptor, blocks, block_cells, cell_stride, cells, bins)
            objects = _orientation_glyphs(votes, cell, p["signed_gradient"])
            if objects:
                drawings["Dominant cell gradients"] = GraphicDrawing(objects)
        info = {
            "model": "opencv_hog_window_v1", "opencv_version": cv2.__version__,
            "working_encoding": "gray" if source.ndim == 2 else "bgr",
            "input_depth": str(source.dtype),
            "window_size": list(size),
            "cell_grid": list(cells),
            "block_grid": list(blocks),
            "descriptor_length": length,
            "order": ["block_x", "block_y", "cell_x", "cell_y", "bin"],
            "settings": p.copy(),
        }
        return Result(
            {"descriptor": descriptor.tolist(), "info": info},
            payloads={"descriptor": {**copy.deepcopy(self.input_payloads["image"]),
                                     "feature_contract": copy.deepcopy(info)}},
            drawings=drawings,
            message=f"{blocks[0]} × {blocks[1]} blocks · {length} descriptor values",
        )
