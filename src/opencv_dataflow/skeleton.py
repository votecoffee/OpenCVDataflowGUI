"""Scikit-image skeleton extraction and length-based terminal spur pruning."""

import numpy as np

from .core import Block, Port, Result
from .graphics_layers import GraphicDrawing, graphic_object
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .selection import SELECTION_INPUTS, selection_mask

SKIMAGE_INSTALL = "Install scikit-image with pip install .[morphology]"
REMOVED_COLOR = (65, 95, 245)


def binary_image(inputs):
    image = inputs["image"]
    if image.ndim != 2 or image.dtype not in (np.uint8, np.bool_):
        raise ValidationError("Use a monochrome uint8 or boolean mask")
    return np.asarray(image) != 0


class Skeletonize(Block):
    type_id, title = "inspect.skeletonize", "Skeletonize mask (scikit-image)"
    category, block_type, subtype = "Inspection", "Inspection", "Segmentation"
    inputs = SELECTION_INPUTS
    outputs = (Port("mask"), Port("distance", required=False), Port("pixel_count", "number"))
    defaults = {"method": "zhang", "max_iterations": 0, "medial_axis_seed": 0}
    parameter_choices = {"method": ("zhang", "lee", "guo_hall", "medial_axis")}
    parameter_ranges = {
        "max_iterations": NumericRange(0, 10000, whole=True),
        "medial_axis_seed": NumericRange(0, 2**32 - 1, whole=True),
    }
    parameter_help = {
        "max_iterations": "Guo–Hall only. Zero runs until stable.",
        "medial_axis_seed": "Local random seed for breaking medial-axis ties.",
    }

    def process(self, inputs):
        try:
            from skimage.morphology import medial_axis, skeletonize, thin
        except ImportError as exc:
            raise RuntimeError(SKIMAGE_INSTALL) from exc
        mask = binary_image(inputs)
        selection = selection_mask(inputs, mask)
        if selection is not None:
            mask &= selection != 0
        padded = np.pad(mask, 1, constant_values=False)
        p = self.parameters
        method = p["method"]
        distance = NO_VALUE
        if method == "medial_axis":
            skeleton, radii = medial_axis(
                padded, return_distance=True, rng=int(p["medial_axis_seed"])
            )
            distance = radii[1:-1, 1:-1].astype(np.float32)
        elif method == "guo_hall":
            skeleton = thin(padded, max_num_iter=int(p["max_iterations"]) or None)
        else:
            skeleton = skeletonize(padded, method=method)
        skeleton = skeleton[1:-1, 1:-1]
        pixels = int(np.count_nonzero(skeleton))
        return Result(
            {"mask": skeleton.astype(np.uint8) * 255, "distance": distance, "pixel_count": pixels},
            message=f"{pixels} skeleton pixels · {method.replace('_', ' ')}",
        )


def remove_corner_shortcuts(adjacency, nodes, skeleton):
    # Skip a diagonal when a two-edge cardinal path already joins its pixels.
    rows = np.repeat(np.arange(len(nodes)), np.diff(adjacency.indptr))
    y1, x1 = np.divmod(nodes[rows], skeleton.shape[1])
    y2, x2 = np.divmod(nodes[adjacency.indices], skeleton.shape[1])
    shortcuts = (x1 != x2) & (y1 != y2) & (skeleton[y1, x2] | skeleton[y2, x1])
    adjacency.data[shortcuts] = 0
    adjacency.eliminate_zeros()


def terminal_spurs(adjacency, nodes, limit, selection):
    offsets = adjacency.indptr
    degree = np.diff(offsets)
    for endpoint in np.flatnonzero(degree == 1):
        previous, current = -1, endpoint
        path, length = [], 0.0
        while True:
            pixel = nodes[current]
            if selection is not None and not selection[pixel]:
                break
            path.append(pixel)
            edge = offsets[current]
            if adjacency.indices[edge] == previous:
                edge += 1
            following = adjacency.indices[edge]
            length += adjacency.data[edge]
            if length > limit:
                break
            if degree[following] > 2:
                yield path, nodes[following]
                break
            if degree[following] != 2:
                break
            previous, current = current, following


class PruneSkeleton(Block):
    type_id, title = "inspect.prune_skeleton", "Prune skeleton spurs"
    category, block_type, subtype = "Inspection", "Inspection", "Segmentation"
    inputs = SELECTION_INPUTS
    outputs = (
        Port("mask"),
        Port("removed", visual_type="region"),
        Port("preview"),
        Port("removed_branches", "number"),
        Port("removed_pixels", "number"),
    )
    defaults = {"max_length": 5.0, "max_passes": 4, "connectivity": "8"}
    parameter_choices = {"connectivity": ("4", "8")}
    parameter_ranges = {
        "max_length": NumericRange(0, 4096),
        "max_passes": NumericRange(1, 32, whole=True),
    }
    parameter_help = {
        "max_length": "Endpoint-to-junction length in pixels. Zero keeps all branches.",
        "max_passes": "Recompute branches after each pass. Stops early when no spurs qualify.",
        "connectivity": "Eight includes diagonals that have no equivalent cardinal path.",
    }

    def process(self, inputs):
        try:
            from skimage.graph import pixel_graph
        except ImportError as exc:
            raise RuntimeError(SKIMAGE_INSTALL) from exc
        original = binary_image(inputs)
        selection = selection_mask(inputs, original)
        if selection is not None:
            selection = selection.ravel()
        mask = original.copy()
        length = float(self.parameters["max_length"])
        diagonals = self.parameters["connectivity"] == "8"
        if length and np.count_nonzero(mask) > 250_000:
            raise ValidationError("Pruning accepts at most 250,000 foreground skeleton pixels")
        branches, objects = 0, []
        draw = self.generate_graphics
        for _ in range(int(self.parameters["max_passes"])):
            if not length or not mask.any() or self._cancelled():
                break
            adjacency, nodes = pixel_graph(mask, connectivity=2 if diagonals else 1)
            if diagonals:
                remove_corner_shortcuts(adjacency, nodes, mask)
            spurs = list(terminal_spurs(adjacency, nodes, length, selection))
            if not spurs:
                break
            branches += len(spurs)
            flat = mask.ravel()
            for path, junction in spurs:
                flat[path] = False
                if draw:
                    y, x = np.divmod([*path, junction], mask.shape[1])
                    objects.append(
                        graphic_object("polyline", zip(x, y, strict=True), REMOVED_COLOR)
                    )
        removed = original & ~mask
        count = int(np.count_nonzero(removed))
        preview = np.zeros((*mask.shape, 3), np.uint8)
        preview[mask] = (80, 230, 140)
        preview[removed] = REMOVED_COLOR
        return Result(
            {
                "mask": mask.astype(np.uint8) * 255,
                "removed": removed.astype(np.uint8) * 255,
                "preview": preview,
                "removed_branches": branches,
                "removed_pixels": count,
            },
            drawings={"removed spurs": GraphicDrawing(tuple(objects))} if objects else {},
            message=f"Removed {branches} terminal spurs and {count} pixels",
        )
